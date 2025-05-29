import os
from pathlib import Path

import pandas as pd
import scanpy as sc
import scipy.io
import scipy.sparse
from scipy.io import mmread

import argparse
import re
from scripts.utils.utils import path_of_file


def get_parse():
    parser = argparse.ArgumentParser(
        description="Download files using Dask with multiprocessing."
    )
    parser.add_argument(
        "--dataset_root_path",
        type=str,
        required=True,
        help="Root path to the datasets",
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
    counter = {}

    dataset_ids = []
    for uuid in target_dataset_files_df["Study_uuid"]:
        if uuid not in counter:
            counter[uuid] = 1
            dataset_ids.append(uuid)
        else:
            counter[uuid] += 1
            dataset_ids.append(f"{uuid}_{counter[uuid]}")

    target_dataset_files_df["dataset_id"] = dataset_ids
    print(target_dataset_files_df.head(10))
    return target_dataset_files_df


def anndata_generate(target_datasets):
    valid_dataset_ids = set(target_datasets["dataset_id"])
    invalid_dataset_ids = set()
    if all(
        col in target_datasets.columns for col in ["path", "filename", "dataset_id"]
    ):
        for row in target_datasets.itertuples(index=False):
            file_path = Path(row.path)
            file_name = row.filename
            uuid = row.Study_uuid
            path_of_mtx_file = file_path / file_name
            print(file_path)
            if not file_path.exists():
                print(f"File {file_path} does not exist")
                continue

            files_in_directory = [f.name for f in file_path.iterdir() if f.is_file()]
            parent_folder = file_path.parent
            files_in_parent_directory = [
                f.name for f in parent_folder.iterdir() if f.is_file()
            ]
            lower_files = [f.lower() for f in files_in_directory]
            lower_files_in_parent_directory = [
                f.lower() for f in files_in_parent_directory
            ]
            mtx_files = [f for f in files_in_directory if f.endswith(".mtx")]
            if not mtx_files:
                print("Not found matrix file")
                continue

            path_of_gene_file = path_of_file(file_path, "gene")
            path_of_cell_file = path_of_file(file_path, "cell")

            genes = pd.read_csv(path_of_gene_file, header=None, names=["feature_name"])
            genes["feature_name"] = genes["feature_name"].str.replace(
                '"', ""
            )  # 有的feature name是带有双引号的，需要去掉
            genes.index = range(1, len(genes) + 1)
            cells = pd.read_csv(path_of_cell_file)
            cells.index = range(1, len(cells) + 1)
            X = scipy.io.mmread(path_of_mtx_file)
            X = X.transpose()
            X = scipy.sparse.csr_matrix(X)

            adata = sc.AnnData(X=X, obs=cells, var=genes)
            print(type(adata.X))

            adata.obs.index = adata.obs.index.astype(str)
            adata.obs = adata.obs.astype(str)

            adata.obs = adata.obs.fillna("")

            if adata.shape[0] < 2000:
                print(f"Dataset {uuid} has less than 2000 cells")
                invalid_dataset_ids.add(uuid)
                continue
            if adata.shape[1] < 10000:
                print(f"Dataset {uuid} has less than 10000 genes")
                invalid_dataset_ids.add(uuid)
                continue

            sc.pp.filter_cells(adata, min_genes=200)
            sc.pp.filter_genes(adata, min_cells=3)
            unique_filename = uuid + ".h5ad"
            output_file_path = Path(file_path) / unique_filename
            adata.write(output_file_path)
    print(invalid_dataset_ids)
    valid_dataset_ids.difference_update(invalid_dataset_ids)
    target_datasets = target_datasets[
        target_datasets["dataset_id"].isin(valid_dataset_ids)
    ]  # 仅保留有效的 dataset_id
    target_datasets.to_csv(
        "newly_updated_target_dataset_with_datasetid.csv", index=False
    )
    print(
        f"已更新 CSV 文件并保存至 updated_target_dataset_with_datasetid.csv ，删除了 {len(invalid_dataset_ids)} 个无效的数据集！"
    )


if __name__ == "__main__":
    args = get_parse()
    target_datasets = getTargetDatasets(args.dataset_root_path)
    anndata_generate(target_datasets)
