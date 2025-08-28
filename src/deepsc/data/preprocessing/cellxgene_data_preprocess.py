import logging
from functools import partial

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse, save_npz
from tqdm import tqdm

import argparse
from multiprocessing import Pool, cpu_count

# GPU加速相关导入
try:
    import cupy as cp
    from cupyx.scipy.sparse import csr_matrix as cupy_csr_matrix

    GPU_AVAILABLE = True
    print("GPU (CuPy) is available")
except ImportError:
    GPU_AVAILABLE = False
    print("GPU (CuPy) is not available, using CPU")

import os

import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
import os.path as osp

from datetime import datetime


def setup_logging(
    log_path: str = "./logs",
    log_name: str = "deepsc",
    rank: int = -1,
    add_timestamp: bool = True,
    log_level: str = "INFO",
) -> str:
    """
    Setup unified logging configuration.

    Args:
        log_path: Directory to store log files
        log_name: Base name for the log file
        rank: Process rank for distributed training (-1 for single process)
        add_timestamp: Whether to add timestamp to log filename
        log_level: Logging level

    Returns:
        str: Path to the created log file
    """
    os.makedirs(log_path, exist_ok=True)

    # Build log filename
    if add_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{log_name}_{timestamp}.log"
    else:
        time_now = datetime.now()
        log_filename = f"{log_name}_{time_now.month}_{time_now.day}_{time_now.hour}_{time_now.minute}.log"

    log_file = osp.join(log_path, log_filename)

    # Set logging level based on rank
    if rank in [-1, 0]:
        level = getattr(logging, log_level.upper())
    else:
        level = logging.WARN

    # Configure logging
    logging.basicConfig(
        level=level,
        format="[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s",
        datefmt="[%X]",
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()],
        force=True,  # Reset any existing logging configuration
    )

    logger = logging.getLogger()
    logger.info(f"Log file initialized: {log_file}")

    return log_file


def process_h5ad_to_sparse_tensor_gpu(
    h5ad_path: str,
    output_path: str,
    gene_map_path: str,
    use_gpu: bool = True,
    normalize: bool = True,
) -> dict:
    """
    集成的处理函数：读取h5ad文件，转换为稀疏矩阵，并可选择性地进行normalization

    Args:
        h5ad_path: h5ad文件路径
        output_path: 输出npz文件路径
        gene_map_path: 基因映射文件路径
        use_gpu: 是否使用GPU加速
        normalize: 是否进行normalization
    """
    # 读取基因映射
    gene_map_df = pd.read_csv(gene_map_path)
    gene_map_df["id"] = gene_map_df["id"].astype(int)
    gene_map = dict(zip(gene_map_df["feature_name"], gene_map_df["id"]))
    max_gene_id = gene_map_df["id"].max()

    # 读取h5ad文件
    adata = ad.read_h5ad(h5ad_path)
    feature_names = adata.var["feature_name"].values
    X = adata.X.tocsr() if issparse(adata.X) else csr_matrix(adata.X)

    print(f"Original matrix shape: {X.shape}")

    # 只保留有映射的基因列
    valid_mask = np.array([f in gene_map for f in feature_names])
    valid_feature_names = feature_names[valid_mask]
    valid_gene_ids = np.array([gene_map[f] for f in valid_feature_names])

    X_valid = X[:, valid_mask]
    print(f"Valid genes matrix shape: {X_valid.shape}")

    # 按 gene_id 排序
    sort_idx = np.argsort(valid_gene_ids)
    X_valid_sorted = X_valid[:, sort_idx]
    valid_gene_ids_sorted = valid_gene_ids[sort_idx]

    n_cells = X.shape[0]
    n_genes = max_gene_id + 1

    # 用三元组(row, col, data)构造目标稀疏矩阵
    X_valid_sorted = X_valid_sorted.tocoo()
    rows = X_valid_sorted.row
    cols = valid_gene_ids_sorted[X_valid_sorted.col]
    data = X_valid_sorted.data
    X_final = csr_matrix((data, (rows, cols)), shape=(n_cells, n_genes), dtype=X.dtype)

    print(f"Final matrix shape: {X_final.shape}")

    if normalize:
        print("Performing normalization...")
        X_final = normalize_tensor_gpu(X_final, use_gpu=use_gpu)

    save_npz(output_path, X_final)
    print(f"Saved processed data to: {output_path}")

    return {"status": "saved", "path": output_path, "shape": X_final.shape}


def normalize_tensor_gpu(csr_matrix, use_gpu: bool = True, min_genes: int = 200):
    """
    对稀疏矩阵进行normalization，可选择使用GPU加速

    Args:
        csr_matrix: 输入的CSR稀疏矩阵
        use_gpu: 是否使用GPU加速
        min_genes: 每个细胞最少基因数量的阈值
    """
    # 过滤掉基因数量少于阈值的细胞
    valid_cells = np.diff(csr_matrix.indptr) >= min_genes
    csr_filtered = csr_matrix
    print(f"Valid cells after filtering: {valid_cells.sum()}")

    if use_gpu and GPU_AVAILABLE:
        print("Using GPU for normalization...")
        try:
            # 转换到GPU
            gpu_matrix = cupy_csr_matrix(csr_filtered)

            # 在GPU上进行log2(1+x)变换
            gpu_matrix.data = cp.log2(1 + gpu_matrix.data)

            # 转换回CPU
            cpu_data = cp.asnumpy(gpu_matrix.data)
            cpu_indices = cp.asnumpy(gpu_matrix.indices)
            cpu_indptr = cp.asnumpy(gpu_matrix.indptr)

            normalized_matrix = csr_matrix(
                (cpu_data, cpu_indices, cpu_indptr), shape=gpu_matrix.shape
            )

            print("GPU normalization completed")
            return normalized_matrix

        except Exception as e:
            print(f"GPU normalization failed: {e}, falling back to CPU")
            use_gpu = False

    if not use_gpu or not GPU_AVAILABLE:
        print("Using CPU for normalization...")
        # CPU normalization
        csr_filtered.data = np.log2(1 + csr_filtered.data)
        print("CPU normalization completed")
        return csr_filtered


def find_all_h5ad_files(root_dir: str):
    """递归查找所有h5ad文件"""
    h5ad_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".h5ad"):
                full_path = os.path.join(dirpath, filename)
                h5ad_files.append(full_path)
                logging.info(f"Found h5ad file: {full_path}")
    return h5ad_files


def process_one_file(
    h5ad_path: str,
    output_dir: str,
    gene_map_path: str,
    use_gpu: bool = True,
    normalize: bool = True,
):
    """处理单个h5ad文件"""
    if not os.path.isfile(h5ad_path):
        logging.warning(f"File not found: {h5ad_path}")
        return f"Warning: File not found {h5ad_path}"

    basename = os.path.basename(h5ad_path).replace(".h5ad", "_processed.npz")
    output_path = os.path.join(output_dir, basename)
    print(f"Processing {h5ad_path} to {output_path}")

    try:
        res = process_h5ad_to_sparse_tensor_gpu(
            h5ad_path, output_path, gene_map_path, use_gpu, normalize
        )
        logging.info(f"Successfully processed: {h5ad_path}, result: {res}")
        return f"Done: {h5ad_path} -> {res['shape']}"
    except Exception as e:
        logging.error(f"Error processing {h5ad_path}: {e}")
        return f"Error: {h5ad_path} - {str(e)}"


def preprocess_cellxgene_folder(
    input_dir: str,
    output_dir: str,
    gene_map_path: str,
    num_processes: int = None,
    use_gpu: bool = True,
    normalize: bool = True,
):
    """批量处理文件夹中的所有h5ad文件"""
    os.makedirs(output_dir, exist_ok=True)
    logfile = setup_logging("integrated_preprocessing", "./logs")
    logging.info("Start integrated preprocessing...")

    print(f"Processing {input_dir} to {output_dir}")
    print(f"Using GPU: {use_gpu and GPU_AVAILABLE}")
    print(f"Normalize: {normalize}")

    h5ad_files = find_all_h5ad_files(input_dir)
    print(f"Found {len(h5ad_files)} h5ad files")

    if not h5ad_files:
        print("No h5ad files found!")
        return

    num_processes = num_processes or min(4, cpu_count())
    print(f"Using {num_processes} processes")

    with Pool(processes=num_processes) as pool:
        worker = partial(
            process_one_file,
            output_dir=output_dir,
            gene_map_path=gene_map_path,
            use_gpu=use_gpu,
            normalize=normalize,
        )
        results = list(
            tqdm(pool.imap_unordered(worker, h5ad_files), total=len(h5ad_files))
        )

    print("\nProcessing Results:")
    for r in results:
        if r:
            print(r)

    logging.info("All processing completed.")


def main():
    parser = argparse.ArgumentParser(
        description="Integrated preprocessing of h5ad files with optional GPU acceleration and normalization."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to directory containing h5ad files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to directory where processed .npz files will be saved.",
    )
    parser.add_argument(
        "--gene_map_path",
        type=str,
        default="/hpcwork/p0021245/Data/gene_map.csv",
        help="Path to gene mapping CSV file.",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=min(4, cpu_count()),
        help="Number of parallel processes to use (default: min(4, cpu cores)).",
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Use GPU acceleration for processing (requires CuPy).",
    )
    parser.add_argument(
        "--no_normalize",
        action="store_true",
        help="Skip normalization step.",
    )
    parser.add_argument(
        "--min_genes",
        type=int,
        default=200,
        help="Minimum number of genes per cell for filtering (default: 200).",
    )

    args = parser.parse_args()

    # 检查GPU可用性
    if args.use_gpu and not GPU_AVAILABLE:
        print("Warning: GPU requested but CuPy not available, using CPU")
        args.use_gpu = False

    preprocess_cellxgene_folder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        gene_map_path=args.gene_map_path,
        num_processes=args.num_processes,
        use_gpu=args.use_gpu,
        normalize=not args.no_normalize,
    )


if __name__ == "__main__":
    main()
