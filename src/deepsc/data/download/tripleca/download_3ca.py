import gzip
import logging
import os
import os.path as osp
import subprocess
import tarfile
import zipfile

import pandas as pd
import requests
from dask import compute, delayed
from dask.diagnostics import ProgressBar

import argparse
import multiprocessing
import time
from deepsc.data.download.tripleca.crawl_3ca import data_crawl
from deepsc.utils import setup_logging


def download_file(url, folder_path, filename, log_path):
    """Download a single file in a worker process.

    Args:
        url (str): File URL.
        folder_path (str): Target folder to save the file.
        filename (str): Output filename.
        log_path (str): Directory where worker logs are stored.

    Returns:
        dict | None: Error info dict if failed; otherwise None.
    """
    process_id = multiprocessing.current_process().pid
    process_log_file = osp.join(log_path, f"worker_{process_id}.log")

    # Configure per-process log file
    logging.basicConfig(
        filename=process_log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="a",
    )

    logging.info(
        "Process %s starts downloading: %s -> %s",
        process_id,
        filename,
        folder_path,
    )

    # Validate URL by HEAD request
    try:
        response = requests.head(url, timeout=5)
        if response.status_code not in (200, 301, 302):
            raise requests.RequestException(
                f"Invalid status code: {response.status_code}"
            )
    except requests.RequestException as exc:
        error_message = f"Invalid URL, skipped: {url}, error: {exc}"
        logging.error(error_message)
        return {
            "url": url,
            "filename": filename,
            "folderpath": folder_path,
            "error": str(exc),
        }

    # Download body
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        file_path = osp.join(folder_path, filename)

        with open(file_path, "wb") as fp:
            for chunk in response.iter_content(chunk_size=8192):
                fp.write(chunk)

        logging.info("Download finished: %s", file_path)
        return None
    except requests.exceptions.RequestException as exc:
        error_message = f"Download failed: {url}, error: {exc}"
        logging.error(error_message)
        return {
            "url": url,
            "filename": filename,
            "folderpath": folder_path,
            "error": str(exc),
        }


def is_valid_url(url):
    """Check whether a URL-like field is valid (non-empty string)."""
    if not isinstance(url, str) or pd.isna(url):
        return False
    return url.strip() != ""


def extract_gzip_tar(file_path, extract_to):
    """Extract a gzip-compressed tar archive.

    Args:
        file_path (str): Path to .tar.gz (or gzipped tar).
        extract_to (str): Output directory.

    Returns:
        tuple[bool, str | None]: (success, error_message)
    """
    try:
        with gzip.open(file_path, "rb") as gz_file:
            with tarfile.open(fileobj=gz_file, mode="r") as tar:
                tar.extractall(path=extract_to)
        return True, None
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def merge_logs(log_path, main_log_file):
    """Append worker logs into the main log file and remove worker logs."""
    with open(main_log_file, "a") as main_log:
        for log_filename in os.listdir(log_path):
            if log_filename.startswith("worker_") and log_filename.endswith(".log"):
                worker_log_file = osp.join(log_path, log_filename)
                with open(worker_log_file, "r") as worker_log:
                    main_log.write(worker_log.read())
                os.remove(worker_log_file)


def detect_file_type(file_path):
    """Detect actual file type using `file` command.

    Returns:
        str: One of {"tar.gz", "zip", "tar", "unknown"}.
    """
    try:
        result = subprocess.run(
            ["file", file_path],
            capture_output=True,
            text=True,
            check=True,
        )
        file_type = result.stdout.lower()

        if "gzip compressed" in file_type and "tar" in file_type:
            return "tar.gz"
        if "zip archive" in file_type:
            return "zip"
        if "tar archive" in file_type:
            return "tar"
        return "unknown"
    except subprocess.CalledProcessError:
        return "unknown"


def extract_file(file_path, extract_folder):
    """Extract an archive based on detected type.

    Args:
        file_path (str): Path to the archive (may be misnamed).
        extract_folder (str): Output directory.

    Returns:
        tuple[bool, str]: (success, message)
    """
    file_type = detect_file_type(file_path)

    if file_type == "zip":
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(extract_folder)
        return True, "Extracted ZIP using zipfile"

    if file_type == "tar.gz":
        with tarfile.open(file_path, "r:gz") as tar_ref:
            tar_ref.extractall(extract_folder)
        return True, "Extracted gzip-compressed tar using tarfile"

    if file_type == "tar":
        with tarfile.open(file_path, "r") as tar_ref:
            tar_ref.extractall(extract_folder)
        return True, "Extracted TAR using tarfile"

    return False, f"Unknown file type: {file_type}"


def extract_and_delete_zips(root_folder, csv_path):
    """Extract all `.zip`-named files under `root_folder` and delete originals.

    This function auto-detects actual file formats (even if misnamed as .zip),
    extracts into a folder named after the file stem, and records failures
    to a CSV.

    Args:
        root_folder (str): Root directory to scan recursively.
        csv_path (str): Path to CSV file for failed extractions.
    """
    logging.info("Start extracting archives...")
    failed_extractions = []
    success_count = 0
    failed_count = 0

    for folder_path, _, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(".zip"):
                zip_path = os.path.join(folder_path, file)
                extract_folder = os.path.join(folder_path, file[:-4])

                logging.info("Processing: %s", zip_path)

                try:
                    os.makedirs(extract_folder, exist_ok=True)

                    success, message = extract_file(zip_path, extract_folder)

                    if success:
                        logging.info(
                            "Extracted: %s -> %s (%s)",
                            zip_path,
                            extract_folder,
                            message,
                        )
                        os.remove(zip_path)
                        logging.info("Removed: %s", zip_path)
                        success_count += 1
                    else:
                        logging.error(
                            "Extraction failed: %s, error: %s", zip_path, message
                        )
                        failed_extractions.append([folder_path, file, message])
                        failed_count += 1

                except Exception as exc:  # noqa: BLE001
                    error_message = str(exc)
                    logging.error(
                        "Extraction failed: %s, error: %s", zip_path, error_message
                    )
                    failed_extractions.append([folder_path, file, error_message])
                    failed_count += 1

    logging.info(
        "Extraction finished. Success: %s, Failed: %s", success_count, failed_count
    )

    if failed_extractions:
        df = pd.DataFrame(
            failed_extractions,
            columns=["folder_path", "filename", "error_message"],
        )
        df.to_csv(
            csv_path,
            mode="a",
            index=False,
            encoding="utf-8-sig",
            header=not os.path.exists(csv_path),
        )
        logging.info("Failed extraction log saved to: %s", csv_path)
    else:
        logging.info("All archives extracted successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download TripleCA (3CA) dataset using Dask with multiprocessing.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Directory where files will be saved.",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        required=True,
        help="Directory for logs.",
    )

    parser.add_argument(
        "--num_processes",
        type=int,
        default=min(4, os.cpu_count()),
        help="Number of parallel processes.",
    )
    args = parser.parse_args()

    download_url_table = data_crawl()
    main_log_file = setup_logging("download", args.log_path)
    failed_output_csv = osp.join(args.log_path, "failed_downloads.csv")

    logging.info("Start download tasks.")
    os.makedirs(args.output_path, exist_ok=True)

    retry_count = 0
    max_retries = 10

    while retry_count < max_retries:
        tasks = []

        # If previous failures exist, retry them first
        if osp.exists(failed_output_csv) and osp.getsize(failed_output_csv) > 0:
            logging.info("Retry #%d for previously failed tasks...", retry_count + 1)
            df_failed = pd.read_csv(failed_output_csv)

            for _, row in df_failed.iterrows():
                url = row["url"]
                filename = row["filename"]
                folder_path = row["folderpath"]
                os.makedirs(folder_path, exist_ok=True)

                logging.info("Downloading %s to %s", filename, folder_path)
                task = delayed(download_file)(url, folder_path, filename, args.log_path)
                tasks.append(task)

        # Otherwise, read from the original URL table
        else:
            logging.info("Reading the original URL table...")
            df = download_url_table
            logging.info("Downloading all files from the dataset...")

            for _, row in df.iterrows():
                uuid_val = row["Study_uuid"]
                url_dataset = row["Data_link"]
                url_metadata = row["Meta data_link"]

                if pd.isna(uuid_val) or not isinstance(uuid_val, str):
                    logging.error("Dataset UUID is empty. Skipping this record.")
                    continue

                if any(not is_valid_url(u) for u in (url_dataset, url_metadata)):
                    logging.error(
                        "Invalid URLs for dataset %s: %s, %s. Skipping.",
                        uuid_val,
                        url_dataset,
                        url_metadata,
                    )
                    continue

                folder_path = os.path.join(args.output_path, uuid_val)
                os.makedirs(folder_path, exist_ok=True)

                for file_type, url in (
                    ("data_set", url_dataset),
                    ("meta_data", url_metadata),
                ):
                    filename = f"{file_type}_{uuid_val}.zip"
                    logging.info("Downloading %s to %s", filename, folder_path)
                    task = delayed(download_file)(
                        url, folder_path, filename, args.log_path
                    )
                    tasks.append(task)

        with ProgressBar():
            results = compute(
                *tasks,
                scheduler="processes",
                num_workers=args.num_processes,
            )

        failed_downloads = [res for res in results if res is not None]

        if failed_downloads:
            failed_df = pd.DataFrame(failed_downloads)
            failed_df.to_csv(failed_output_csv, index=False)
            logging.info(
                "Some files failed to download. Saved to: %s",
                failed_output_csv,
            )
            retry_count += 1
            logging.info(
                "Sleeping 10 seconds before retry (%d/%d)...",
                retry_count,
                max_retries,
            )
            time.sleep(10)
        else:
            logging.info("All files downloaded successfully. No failed tasks.")
            if osp.exists(failed_output_csv):
                os.remove(failed_output_csv)
            break

    merge_logs(args.log_path, main_log_file)

    if retry_count >= max_retries:
        logging.error(
            "Reached maximum retries. Some files still failed. "
            "Please check failed_downloads.csv manually."
        )
    failed_extract_csv = osp.join(args.output_path, "extract_failed.csv")
    extract_and_delete_zips(args.output_path, failed_extract_csv)
