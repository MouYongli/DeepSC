import pandas as pd
import os
import os.path as osp
import requests
import dask
from dask import delayed, compute, config
from dask.diagnostics import ProgressBar
import argparse
import logging
from datetime import datetime
import mimetypes
from urllib.parse import urlparse

def get_parse():
    parser = argparse.ArgumentParser(description="Download files using Dask with multiprocessing.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV file containing download links")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the directory where files will be saved")
    parser.add_argument("--log_path", type=str, required=True, help="Path to the log directory or log file")
    parser.add_argument("--num_files", type=int, default=3, help="Number of files to download")
    parser.add_argument("--num_processes", type=int, default=min(4, os.cpu_count()), help="Number of parallel processes")
    return parser.parse_args()

def setup_logging(log_path):
    os.makedirs(log_path, exist_ok=True)  # 确保日志目录存在

    # 生成带时间戳的日志文件名
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = osp.join(log_path, f"download_{timestamp}.log")

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

def download_file(url, folder_path, filename):
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()  # 确保请求成功
        file_path = osp.join(folder_path, filename)

        with open(file_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        with open(log_file, "a") as log:
            log.write(f"下载完成: {file_path}\n")
        return file_path
    except requests.exceptions.RequestException as e:
        with open(log_file, "a") as log:
            log.write(f"下载失败: {url}, 错误: {e}\n")
        return None
    
def is_valid_url(url):
    """检查 URL 是否有效"""
    if not isinstance(url, str) or url.strip() == "":
        return False  # 空值或非字符串，直接返回 False
    
    try:
        response = requests.head(url, timeout=5)
        return response.status_code in [200, 301, 302]  # 仅接受 200、301、302 状态码
    except requests.RequestException:
        return False  # 请求失败，认为 URL 无效
    
def get_filename_from_url(url, default_ext=".bin"):
    """从 URL 提取文件名，并自动补全扩展名（如果缺失）"""
    parsed_url = urlparse(url)
    filename = osp.basename(parsed_url.path).strip().strip('"').strip("'")
    ext='.zip'
    return filename + ext

if __name__ == "__main__":
    args = get_parse()

    # 设置日志系统，并获取最终的日志文件路径
    log_file = setup_logging(args.log_path)

    logging.info("开始下载任务")

    # 确保输出目录存在
    os.makedirs(args.output_path, exist_ok=True)

    # 读取数据文件
    try:
        df = pd.read_csv(args.csv_path)
    except Exception as e:
        logging.error(f"无法读取 CSV 文件: {args.csv_path}, 错误: {e}")
        exit(1)

    # 准备下载任务
    tasks = []
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
            filename = f"{file_type}_{get_filename_from_url(url)}"
            logging.info(f"正在将文件 {filename} 下载到路径 {folder_path}")

            task = delayed(download_file)(url, folder_path, filename)
            tasks.append(task)

    with ProgressBar():
        results = compute(*tasks, scheduler="processes", num_workers=args.num_processes)

    logging.info("所有文件下载完成")