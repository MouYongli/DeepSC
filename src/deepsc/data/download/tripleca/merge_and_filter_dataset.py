import logging
import os
from pathlib import Path

import pandas as pd
import scanpy as sc
import scipy.io
import scipy.sparse
from scipy.io import mmread

import argparse
import re
from deepsc.utils import path_of_file, setup_logging

def get_target_datasets(root_folder):
    """Find candidate datasets and filter invalid ones.

    Args:
        root_folder (str): Path to the root directory of datasets.

    Returns:
        pd.DataFrame: Target dataset file information.
    """
    int64_count = 0
    float64_count = 0
    total_files = 0

    pattern = r"([a-f0-9]{8}-[a-f0-9]{4}-" r"[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})"

    excluded_files_df = pd.DataFrame(
        columns=["Study_uuid", "filename", "total_sum", "floored_sum", "path"]
    )
    target_dataset_files_df = pd.DataFrame(columns=["Study_uuid", "filename", "path"])

    for current_path, _, files in os.walk(root_folder):
        for filename in files:
            if filename.endswith(".mtx"):
                total_files += 1
                matrix = mmread(os.path.join(current_path, filename))
                study_uuid_match = re.search(pattern, current_path)
                study_uuid = (
                    study_uuid_match.group(0) if study_uuid_match else "Unknown"
                )

                if matrix.dtype == "int64":
                    int64_count += 1

                elif matrix.dtype == "float64":
                    float64_count += 1

                    with open(os.path.join(current_path, filename), "r") as f:
                        lines = f.readlines()

                    if len(lines) > 152:
                        values1 = [float(line.split()[2]) for line in lines[2:52]]
                        values2 = [float(line.split()[2]) for line in lines[102:152]]
                        total_sum = sum(values1) + sum(values2)
                        floored_sum = int(total_sum)

                        if total_sum - floored_sum > 0.00001:
                            logging.warning("Excluded file detail:")
                            logging.warning("Found in %s", current_path)
                            logging.warning("Total sum: %s", total_sum)
                            logging.warning("Floored sum: %s", floored_sum)

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
                                [excluded_files_df, excluded_dataset],
                                ignore_index=True,
                            )
                            continue

                target_dataset = pd.DataFrame(
                    [
                        {
                            "Study_uuid": study_uuid,
                            "filename": filename,
                            "path": current_path,
                        }
                    ]
                )
                target_dataset_files_df = pd.concat(
                    [target_dataset_files_df, target_dataset],
                    ignore_index=True,
                )
    excluded_file = os.path.join(
        args.dataset_root_path, "excluded_files.csv"
    )
    target_file = os.path.join(
        args.dataset_root_path, "target_dataset_files.csv"
    )
    if not excluded_files_df.empty:
        excluded_files_df.to_csv(excluded_file, index=False)
        logging.info(f"Excluded files saved to {excluded_file}.")
    if not target_dataset_files_df.empty:
        target_dataset_files_df.to_csv(target_file, index=False)
        logging.info(f"Target dataset files saved to {target_file}.")

    logging.info("Total files: %s", total_files)
    logging.info("Number of int64 type: %s", int64_count)
    logging.info("Number of float64 type: %s", float64_count)

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
    logging.info("Preview of target datasets:\n%s", target_dataset_files_df.head(10))
    return target_dataset_files_df


def anndata_generate(target_datasets):
    """Generate AnnData objects and filter invalid datasets."""
    valid_dataset_ids = set(target_datasets["dataset_id"])
    invalid_dataset_ids = set()

    if all(
        col in target_datasets.columns for col in ["path", "filename", "dataset_id"]
    ):

        for row in target_datasets.itertuples(index=False):
            file_path = Path(row.path)
            file_name = row.filename
            uuid = row.dataset_id
            path_of_mtx_file = file_path / file_name
            logging.info("Processing dataset at: %s", file_path)

            if not file_path.exists():
                logging.error("File %s does not exist", file_path)
                continue

            mtx_files = [
                f.name
                for f in file_path.iterdir()
                if f.is_file() and f.name.endswith(".mtx")
            ]
            if not mtx_files:
                logging.error("Matrix file not found in %s", file_path)
                continue

            path_of_gene_file = path_of_file(file_path, "gene")
            path_of_cell_file = path_of_file(file_path, "cell")

            genes = pd.read_csv(path_of_gene_file, header=None, names=["feature_name"])
            genes["feature_name"] = genes["feature_name"].str.replace('"', "")
            genes.index = range(1, len(genes) + 1)

            cells = pd.read_csv(path_of_cell_file)
            cells.index = range(1, len(cells) + 1)

            X = scipy.io.mmread(path_of_mtx_file)
            X = X.transpose()
            X = scipy.sparse.csr_matrix(X)

            adata = sc.AnnData(X=X, obs=cells, var=genes)
            logging.info("AnnData created with shape %s", adata.shape)

            adata.obs.index = adata.obs.index.astype(str)
            adata.obs = adata.obs.astype(str).fillna("")

            if adata.shape[0] < 2000:
                logging.warning("Dataset %s has less than 2000 cells", uuid)
                invalid_dataset_ids.add(uuid)
                continue
            if adata.shape[1] < 10000:
                logging.warning("Dataset %s has less than 10000 genes", uuid)
                invalid_dataset_ids.add(uuid)
                continue

            sc.pp.filter_cells(adata, min_genes=200)
            sc.pp.filter_genes(adata, min_cells=3)
            unique_filename = uuid + ".h5ad"
            output_file_path = Path(file_path) / unique_filename
            adata.write(output_file_path)
            logging.info("Saved AnnData to %s", output_file_path)

    logging.info("Invalid dataset IDs: %s", invalid_dataset_ids)
    valid_dataset_ids.difference_update(invalid_dataset_ids)

    target_datasets = target_datasets[
        target_datasets["dataset_id"].isin(valid_dataset_ids)
    ]

    output_file = os.path.join(
        args.dataset_root_path, "updated_target_dataset_with_datasetid.csv"
    )
    target_datasets.to_csv(
        output_file,
        index=False,
    )
    logging.info(
        "CSV updated and saved. Removed %d invalid datasets.",
        len(invalid_dataset_ids),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find and process dataset files for DeepSC."
    )
    parser.add_argument(
        "--dataset_root_path",
        type=str,
        required=True,
        help="Root path to the datasets.",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        required=True,
        help="Path to the log directory.",
    )
    args = parser.parse_args()
    main_log_file = setup_logging("dataset_processing", args.log_path)

    target_datasets = get_target_datasets(args.dataset_root_path)
    anndata_generate(target_datasets)

    logging.info("All tasks completed. Logs saved to %s", main_log_file)
