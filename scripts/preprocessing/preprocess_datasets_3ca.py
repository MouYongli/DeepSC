import logging
import os
from functools import partial

import pandas as pd
from tqdm import tqdm

from multiprocessing import Pool, cpu_count
from scripts.preprocessing.preprocess_datasets import process_h5ad_to_sparse_tensor
from scripts.utils.utils import setup_logging


def process_one_sample(row_dict, output_dir):
    dataset_id = row_dict["dataset_id"]
    h5ad_path = os.path.join(row_dict["path"], f"{dataset_id}.h5ad")

    if not os.path.isfile(h5ad_path):
        logging.info(f"warning : File not found {h5ad_path}")
        return f"Warning: File not found {h5ad_path}"

    basename = os.path.basename(h5ad_path).replace(".h5ad", "_sparse.pth")
    output_path = os.path.join(output_dir, basename)

    # logging.info(f"Processing {h5ad_path} -> {output_path}")
    res = process_h5ad_to_sparse_tensor(h5ad_path, output_path)
    logging.info(res)
    return f"Done: {h5ad_path}"


def preprocess_datasets_3ca(output_dir: str, num_processes: int = None):
    metadata_path = "/home/angli/DeepSC/scripts/download/tripleca/updated_target_dataset_with_datasetid.csv"
    os.makedirs(output_dir, exist_ok=True)

    logfile = setup_logging("preprocessing", "./logs")
    logging.info("Start preprocessing datasets...")

    meta_df = pd.read_csv(metadata_path)
    rows = meta_df.to_dict(orient="records")

    num_processes = num_processes or cpu_count()

    with Pool(processes=num_processes) as pool:
        worker = partial(process_one_sample, output_dir=output_dir)
        results = list(tqdm(pool.imap_unordered(worker, rows), total=len(rows)))

    for r in results:
        if r:
            print(r)

    logging.info("All done.")


# Command-line argument parsing and main entry point.
import argparse


def get_parse():
    parser = argparse.ArgumentParser(
        description="Preprocess h5ad files from 3CA datasets using multiprocessing."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to directory where sparse tensor .pth files will be saved.",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=min(4, cpu_count()),
        help="Number of parallel processes to use (default: min(4, cpu cores)).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_parse()
    preprocess_datasets_3ca(
        output_dir=args.output_dir, num_processes=args.num_processes
    )
