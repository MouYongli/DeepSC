import logging
import os
from functools import partial

import pandas as pd
from tqdm import tqdm

import argparse
from multiprocessing import Pool, cpu_count
from src.deepsc.data.preprocessing.preprocess_datasets import (
    process_h5ad_to_sparse_tensor,
)
from src.deepsc.utils import setup_logging


def process_one_sample(row_dict, output_dir, gene_map_path):
    """Process a single 3CA dataset sample to convert h5ad to sparse tensor format.

    Converts an h5ad file from the 3CA dataset to sparse tensor format (.npz)
    for efficient loading during model training. Handles missing files gracefully
    by logging warnings.

    Args:
        row_dict (dict): Dictionary containing dataset metadata with keys:
            - "dataset_id": Unique identifier for the dataset
            - "path": Directory path where the h5ad file is located
        output_dir (str): Directory path where the processed sparse tensor
            files will be saved.

    Returns:
        str: Status message indicating completion or warning about missing files.
    """
    dataset_id = row_dict["dataset_id"]
    h5ad_path = os.path.join(row_dict["path"], f"{dataset_id}.h5ad")

    if not os.path.isfile(h5ad_path):
        logging.info(f"warning : File not found {h5ad_path}")
        return f"Warning: File not found {h5ad_path}"

    basename = os.path.basename(h5ad_path).replace(".h5ad", "_sparse.npz")
    output_path = os.path.join(output_dir, basename)

    # logging.info(f"Processing {h5ad_path} -> {output_path}")
    res = process_h5ad_to_sparse_tensor(h5ad_path, output_path, gene_map_path)
    logging.info(res)
    return f"Done: {h5ad_path}"


def preprocess_datasets_3ca(
    input_dir: str, gene_map_path: str, output_dir: str, num_processes: int = None
):
    """Preprocess all 3CA datasets using parallel processing.

    Reads metadata from a CSV file and processes multiple h5ad files in parallel
    to convert them to sparse tensor format. Uses multiprocessing to efficiently
    handle large numbers of datasets.

    Args:
        input_dir (str): Directory containing the metadata CSV file and where
            processed sparse tensor files will be saved.
        output_dir (str): Directory containing the metadata CSV file and where
            processed sparse tensor files will be saved.
        num_processes (int, optional): Number of parallel processes to use.
            Defaults to the number of CPU cores available.

    Returns:
        None: Results are written to files in the output directory.
    """
    metadata_path = os.path.join(input_dir, "updated_target_dataset_with_datasetid.csv")

    os.makedirs(output_dir, exist_ok=True)

    logfile = setup_logging("preprocessing", "./logs")
    logging.info("Start preprocessing datasets...")

    meta_df = pd.read_csv(metadata_path)
    rows = meta_df.to_dict(orient="records")

    num_processes = num_processes or cpu_count()

    with Pool(processes=num_processes) as pool:
        worker = partial(
            process_one_sample, output_dir=output_dir, gene_map_path=gene_map_path
        )
        results = list(tqdm(pool.imap_unordered(worker, rows), total=len(rows)))

    for r in results:
        if r:
            print(r)

    logging.info("All done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess h5ad files from 3CA datasets using multiprocessing."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to directory containing 3CA datasets.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to directory where sparse tensor .pth files will be saved.",
    )
    parser.add_argument(
        "--gene_map_path",
        type=str,
        required=True,
        help="Path to gene map CSV file.",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=min(4, cpu_count()),
        help="Number of parallel processes to use (default: min(4, cpu cores)).",
    )

    args = parser.parse_args()
    preprocess_datasets_3ca(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        num_processes=args.num_processes,
        gene_map_path=args.gene_map_path,
    )
