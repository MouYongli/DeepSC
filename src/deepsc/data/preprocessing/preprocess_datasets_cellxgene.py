import logging
import os
from functools import partial

from tqdm import tqdm

from multiprocessing import Pool, cpu_count
from deepsc.data.preprocessing.preprocess_datasets import process_h5ad_to_sparse_tensor
from deepsc.utils import setup_logging

def find_all_h5ad_files(root_dir: str):
    h5ad_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".h5ad"):
                full_path = os.path.join(dirpath, filename)
                h5ad_files.append(full_path)
                logging.info(f"Found h5ad file: {full_path}")
    return h5ad_files


def process_one_file(h5ad_path: str, output_dir: str):
    if not os.path.isfile(h5ad_path):
        logging.info(f"Warning: File not found {h5ad_path}")
        return f"Warning: File not found {h5ad_path}"

    basename = os.path.basename(h5ad_path).replace(".h5ad", "_sparse.pth")
    output_path = os.path.join(output_dir, basename)

    try:
        res = process_h5ad_to_sparse_tensor(h5ad_path, output_path)
        logging.info(res)
        return f"Done: {h5ad_path}"
    except Exception as e:
        logging.error(f"Error processing {h5ad_path}: {e}")
        return f"Error: {h5ad_path}"


def preprocess_cellxgene_folder(
    input_dir: str, output_dir: str, num_processes: int = None
):
    os.makedirs(output_dir, exist_ok=True)
    logfile = setup_logging("preprocessing_cellxgene", "./logs")
    logging.info("Start preprocessing cellxgene datasets...")

    h5ad_files = find_all_h5ad_files(input_dir)
    num_processes = num_processes or cpu_count()

    with Pool(processes=num_processes) as pool:
        worker = partial(process_one_file, output_dir=output_dir)
        results = list(
            tqdm(pool.imap_unordered(worker, h5ad_files), total=len(h5ad_files))
        )

    for r in results:
        if r:
            print(r)

    logging.info("All done.")
