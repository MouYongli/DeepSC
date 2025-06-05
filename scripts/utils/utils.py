import logging
import os
import os.path as osp
from pathlib import Path

from datetime import datetime


def path_of_file(file_path, file_name):
    if file_name == "cell":
        searchKey1 = "cell"
        searchKey2 = ".csv"

    if file_name == "gene":
        searchKey1 = "gene"
        searchKey2 = ".txt"

    files_in_directory = {
        f.name.lower(): f.name for f in file_path.iterdir() if f.is_file()
    }
    lower_files = list(files_in_directory.keys())
    search_file_path = Path("")

    search_files = [
        f for f in lower_files if f.startswith(searchKey1) and f.endswith(searchKey2)
    ]
    if search_files:
        if not len(search_files) > 1:
            # print(f"find {file_name} file: {search_files[0]} in path {file_path}")
            original_file_name = files_in_directory[search_files[0]]
            search_file_path = file_path / original_file_name
            return search_file_path
        else:
            print(f"Multiple files found in path {file_path}")
    else:
        parent_folder = file_path.parent
        files_in_parent_directory = {
            f.name.lower(): f.name for f in parent_folder.iterdir() if f.is_file()
        }
        lower_files_in_parent_directory = list(files_in_parent_directory.keys())
        search_files = [
            f
            for f in lower_files_in_parent_directory
            if f.startswith(searchKey1) and f.endswith(searchKey2)
        ]
        if search_files:
            if not len(search_files) > 1:
                original_file_name = files_in_parent_directory[search_files[0]]
                search_file_path = parent_folder / original_file_name
                # print(f"find gene file: {search_files[0]} in path {parent_folder}")
                return search_file_path
            else:
                print(f"Multiple files found in path {file_path}")
        else:
            print(f"Corresponding file not found in path {file_path}")


def setup_logging(type, log_path):
    os.makedirs(log_path, exist_ok=True)  # 确保日志目录存在

    # 生成带时间戳的日志文件名
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = osp.join(log_path, f"{type}_{timestamp}.log")

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="w",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)

    logging.info(f"日志文件: {log_file}")

    return log_file
