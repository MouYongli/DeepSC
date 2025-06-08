import os

import pandas as pd
from scipy.io import mmread

import argparse
import re


def get_parse():
    parser = argparse.ArgumentParser(description="Process data from the dataset folder")
    parser.add_argument(
        "--dataset_path", type=str, required=True, help="Path to the datasets"
    )
    return parser.parse_args()


def getTargetDatasets(root_folder):
    """
    Find the dataset that is suitable for DeepSC project
    """
    i = 0  # int64 类型文件计数
    j = 0  # float64 类型文件计数
    x = 0  # 处理的总文件数

    pattern = r"([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})"

    # 用于存储数据的 DataFrame
    excluded_files_df = pd.DataFrame(
        columns=["Study_uuid", "filename", "total_sum", "floored_sum", "path"]
    )
    target_dataset_files_df = pd.DataFrame(columns=["Study_uuid", "filename", "path"])
    for current_path, dirs, files in os.walk(root_folder):
        for filename in files:
            if filename.endswith(".mtx"):
                x += 1
                matrix = mmread(os.path.join(current_path, filename))  # 读取 .mtx 文件
                study_uuid_match = re.search(pattern, current_path)  # 提取 study_uuid
                study_uuid = (
                    study_uuid_match.group(0) if study_uuid_match else "Unknown"
                )

                if matrix.dtype == "int64":
                    i += 1

                elif matrix.dtype == "float64":
                    j += 1

                    with open(os.path.join(current_path, filename), "r") as f:
                        lines = f.readlines()

                    if len(lines) > 152:  # 确保有足够的行 maybe not needed
                        values1 = [float(line.split()[2]) for line in lines[2:52]]
                        values2 = [float(line.split()[2]) for line in lines[102:152]]
                        total_sum = sum(values1) + sum(values2)
                        floored_sum = int(total_sum)

                        if total_sum - floored_sum > 0.00001:
                            print("Detail of excluded file:")
                            print(f"Found in {current_path}")
                            print(f"Total sum: {total_sum}")
                            print(f"Floored sum: {floored_sum}")

                            excluded_dataset = pd.DataFrame(
                                [
                                    {
                                        "Study_uuid": study_uuid,
                                        "filename": filename,
                                        "total_sum": total_sum,
                                        "floored_sum": floored_sum,
                                        "path": current_path,
                                    }
                                ]
                            )
                            excluded_files_df = pd.concat(
                                [excluded_files_df, excluded_dataset], ignore_index=True
                            )
                            continue
                targetDataset = pd.DataFrame(
                    [
                        {
                            "Study_uuid": study_uuid,
                            "filename": filename,
                            "path": current_path,
                        }
                    ]
                )
                target_dataset_files_df = pd.concat(
                    [target_dataset_files_df, targetDataset], ignore_index=True
                )

    if not excluded_files_df.empty:
        excluded_files_df.to_csv("excluded_files.csv", index=False)
    if not target_dataset_files_df.empty:
        target_dataset_files_df.to_csv("target_dataset_files.csv", index=False)

    print(f"Totally {x} files.")
    print(f"Number of int64 type: {i}")
    print(f"Number of float64 type: {j}")
    print("Excluded files saved to 'excluded_files.csv'.")

    data_info_df = pd.read_csv("data_info.csv")
    target_dataset_files_df = pd.read_csv("target_dataset_files.csv")

    target_study_uuids = set(target_dataset_files_df["Study_uuid"])

    processed_data_info_df = data_info_df[
        data_info_df["Study_uuid"].isin(target_study_uuids)
    ]

    processed_data_info_df.to_csv("processed_data_info.csv", index=False)

    print("Saved target study information to processed_data_info.csv")


if __name__ == "__main__":
    args = get_parse()
    dataset_path = args.dataset_path
    getTargetDatasets(dataset_path)
