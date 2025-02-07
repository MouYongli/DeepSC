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

        logging.info(f"下载完成: {file_path}")
        return file_path
    except requests.exceptions.RequestException as e:
        logging.error(f"下载失败: {url}, 错误: {e}")
        return None

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
    for _, row in df.head(args.num_files).iterrows():
        uuid = row["Study_uuid"]
        url = row["Data_link"]
        folder_path = os.path.join(args.output_path, uuid)  # 定义存储路径
        os.makedirs(folder_path, exist_ok=True)

        filename = osp.basename(url)  # 从 URL 获取文件名
        task = delayed(download_file)(url, folder_path, filename)
        tasks.append(task)

    # 并行下载
    with ProgressBar():
        results = compute(*tasks, scheduler="processes", num_workers=args.num_processes)

    logging.info("所有文件下载完成")