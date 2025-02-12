import pandas as pd
import os
import os.path as osp
import requests
import dask
from dask import delayed, compute
from dask.diagnostics import ProgressBar
import argparse
import logging
from datetime import datetime
import mimetypes
from urllib.parse import urlparse
import multiprocessing
import time  # 用于延迟重试
import zipfile


def get_parse():
    parser = argparse.ArgumentParser(description="Download files using Dask with multiprocessing.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV file containing download links")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the directory where files will be saved")
    parser.add_argument("--log_path", type=str, required=True, help="Path to the log directory")
    parser.add_argument("--num_files", type=int, default=3, help="Number of files to download")
    parser.add_argument("--num_processes", type=int, default=min(4, os.cpu_count()), help="Number of parallel processes")
    return parser.parse_args()

def setup_logging(type, log_path):
    os.makedirs(log_path, exist_ok=True)  # 确保日志目录存在

    # 生成带时间戳的日志文件名
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = osp.join(log_path, f"{type}_{timestamp}.log")

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="w"
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)

    logging.info(f"日志文件: {log_file}")
    
    return log_file

def download_file(url, folder_path, filename, log_path):
    """ 子进程下载文件，返回失败信息（如果失败） """
    process_id = multiprocessing.current_process().pid
    process_log_file = osp.join(log_path, f"worker_{process_id}.log")
    
    # 设置单独的日志文件
    logging.basicConfig(
        filename=process_log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="a"
    )
    
    logging.info(f"进程 {process_id} 开始下载: {filename} -> {folder_path}")
    

    # 检查 URL 是否有效
    try:
        response = requests.head(url, timeout=5)
        if response.status_code not in [200, 301, 302]:  
            raise requests.RequestException(f"Invalid status code: {response.status_code}")  
    except requests.RequestException as e:
        error_message = f"无效的 URL，跳过下载: {url}, 错误: {e}"
        logging.error(error_message)
        return {"url": url, "filename": filename, "folderpath": folder_path, "error": str(e)}

    # 下载文件
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()  # 确保请求成功
        file_path = osp.join(folder_path, filename)

        with open(file_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        logging.info(f"下载完成: {file_path}")
        return None  # 成功下载，返回 None
    except requests.exceptions.RequestException as e:
        error_message = f"下载失败: {url}, 错误: {e}"
        logging.error(error_message)
        return {"url": url, "filename": filename, "folderpath": folder_path, "error": str(e)}  

def is_valid_url(url):
    """检查 URL 是否有效"""
    if not isinstance(url, str) or pd.isna(url):
        return False

    url = url.strip()
    if url == "":
        return False  
    return True

def merge_logs(log_path, main_log_file):
    """合并子进程的日志到主日志文件"""
    with open(main_log_file, "a") as main_log:
        for log_filename in os.listdir(log_path):
            if log_filename.startswith("worker_") and log_filename.endswith(".log"):
                worker_log_file = osp.join(log_path, log_filename)
                with open(worker_log_file, "r") as worker_log:
                    main_log.write(worker_log.read()) 
                os.remove(worker_log_file)


def extract_and_delete_zips(root_folder, csv_path):
    """
    遍历 root_folder 下的所有子文件夹，找到所有 .zip 文件并解压到以 .zip 文件名命名的文件夹中，之后删除 .zip 文件。
    失败的解压记录会存入 pandas DataFrame，并写入 csv_path (CSV) 文件中。
    """
    logging.error(f"开始解压缩......")
    failed_extractions = []  # 存储失败的 .zip 文件信息

    for folder_path, _, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(".zip"):  # 仅处理 .zip 文件
                zip_path = os.path.join(folder_path, file)
                extract_folder = os.path.join(folder_path, file[:-4])  # 创建去掉 .zip 后的文件夹
                
                try:
                    # 创建解压目标文件夹（如果不存在）
                    os.makedirs(extract_folder, exist_ok=True)

                    # 解压 ZIP 文件
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_folder)
                    logging.info(f"解压成功: {zip_path} -> {extract_folder}")

                    # 删除 ZIP 文件
                    os.remove(zip_path)
                    logging.info(f"已删除 ZIP 文件: {zip_path}")

                except zipfile.BadZipFile as e:
                    error_message = str(e)
                    logging.error(f"解压失败: {zip_path}, 错误: {error_message}")

                    # 记录失败信息
                    failed_extractions.append([folder_path, file, error_message])

    # 如果有解压失败的文件，存入 DataFrame 并写入 CSV
    if failed_extractions:
        df = pd.DataFrame(failed_extractions, columns=["文件夹路径", "文件名", "错误信息"])
        df.to_csv(csv_path, mode="a", index=False, encoding="utf-8-sig", header=not os.path.exists(csv_path))
        logging.info(f"失败的解压记录已保存至: {csv_path}")
    else:
        logging.info("全部文件解压成功")

if __name__ == "__main__":
    args = get_parse()

    main_log_file = setup_logging('download',args.log_path)
    failed_output_csv = osp.join(args.log_path, "failed_downloads.csv")

    logging.info("开始下载任务")

    os.makedirs(args.output_path, exist_ok=True)

    retry_count = 0  # 记录重试次数
    max_retries = 10  # 最大重试次数

    while retry_count < max_retries:
        # 准备下载任务
        tasks = []
        failed_downloads = []  # 存储新的失败任务

        # 检查是否有未下载成功的记录
        if osp.exists(failed_output_csv) and osp.getsize(failed_output_csv) > 0:
            logging.info(f"第 {retry_count+1} 次尝试下载失败任务...")
            df = pd.read_csv(failed_output_csv)  # 读取之前失败的任务
            for _, row in df.iterrows():
                url = row["url"]
                filename = row["filename"]
                folder_path = row["folderpath"]
                os.makedirs(folder_path, exist_ok=True)

                logging.info(f"正在将文件 {filename} 下载到路径 {folder_path}")

                task = delayed(download_file)(url, folder_path, filename, args.log_path)
                tasks.append(task)
        # 读取原始csv文件
        else:
            logging.info("读取原始 CSV 文件...")
            try:
                df = pd.read_csv(args.csv_path)
            except Exception as e:
                logging.error(f"无法读取 CSV 文件: {args.csv_path}, 错误: {e}")
                exit(1)
            logging.info(f"选择下载文件数目为 {args.num_files}")
            for _, row in df.head(args.num_files).iterrows():

                uuid = row["Study_uuid"]
                url_dataset = row["Data_link"]
                url_metadata=row["Meta data_link"]

                if pd.isna(uuid) or not isinstance(uuid, str):
                    logging.error(f"数据集 UUID 为空，跳过该条记录")
                    continue

                if any(not is_valid_url(url) for url in [url_dataset, url_metadata]):
                    logging.error(f"数据集 {uuid} 的下载链接无效: {url_dataset}, {url_metadata}，跳过该任务")
                    continue

                folder_path = os.path.join(args.output_path, uuid)

                os.makedirs(folder_path, exist_ok=True)

                for file_type, url in [("data_set", url_dataset), ("meta_data", url_metadata)]:
                    filename = f"{file_type}_{uuid}.zip"
                    logging.info(f"正在将文件 {filename} 下载到路径 {folder_path}")

                    task = delayed(download_file)(url, folder_path, filename, args.log_path)
                    tasks.append(task)



        with ProgressBar():
            results = compute(*tasks, scheduler="processes", num_workers=args.num_processes)


        failed_downloads = [res for res in results if res is not None]  # 过滤出失败的下载任务
        if failed_downloads:
            failed_df = pd.DataFrame(failed_downloads)
            failed_df.to_csv(failed_output_csv, index=False)
            logging.info(f"仍有未成功下载的文件，失败任务已保存到: {failed_output_csv}")
            retry_count += 1  # 增加重试次数
            logging.info(f"等待 10 秒后重试 (重试次数: {retry_count}/{max_retries})...")
            time.sleep(10)  # 让服务器缓冲一会再尝试
        else:
            logging.info("所有文件下载完成，没有失败任务")
            if osp.exists(failed_output_csv):
                os.remove(failed_output_csv)  
            break
    merge_logs(args.log_path, main_log_file)

    if retry_count >= max_retries:
        logging.error("达到最大重试次数，仍有文件下载失败，请手动检查 failed_downloads.csv")
    
    extract_and_delete_zips(args.output_path, "./extract_failed.csv")