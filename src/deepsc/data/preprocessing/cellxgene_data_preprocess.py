"""CellxGene data preprocessing module.

This module provides functionality to preprocess h5ad files from CellxGene,
converting them to sparse tensors with optional normalization for downstream
analysis in the DeepSC framework.
"""

import logging
import os
from functools import partial

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags, issparse, save_npz
from tqdm import tqdm

import argparse
from deepsc.utils.utils import setup_logging
from multiprocessing import Pool, cpu_count

# CPU-only processing configuration
GPU_AVAILABLE = False


def process_h5ad_to_sparse_tensor(
    h5ad_path: str,
    output_path: str,
    gene_map_path: str,
    normalize: bool = True,
    scale_factor: float = 1e4,
    min_genes: int = 200,
    apply_log1p: bool = True,
) -> dict:
    """Process h5ad file to sparse tensor with optional TP10K normalization.

    Reads an h5ad file, converts it to a sparse matrix format, maps genes
    according to the provided gene mapping, and optionally applies TP10K/CPM
    normalization with log1p transformation.
    The resulting sparse matrix is saved as an npz file.

    Args:
        h5ad_path: Path to the input h5ad file.
        output_path: Path where the processed npz file will be saved.
        gene_map_path: Path to the CSV file containing gene mapping information.
        normalize: Whether to apply normalization to the data.
        scale_factor: Scaling factor for normalization (1e4 for TP10K, 1e6 for CPM).
        min_genes: Minimum number of genes per cell for filtering.
        apply_log1p: Whether to apply log1p transformation.

    Returns:
        dict: Processing status information including save path and matrix shape.
    """
    # Load gene mapping from CSV file
    gene_map_df = pd.read_csv(gene_map_path)
    gene_map_df["id"] = gene_map_df["id"].astype(int)
    gene_map = dict(zip(gene_map_df["feature_name"], gene_map_df["id"]))
    max_gene_id = gene_map_df["id"].max()

    # Load h5ad file and extract gene expression matrix
    adata = ad.read_h5ad(h5ad_path)
    feature_names = adata.var["feature_name"].values
    X = adata.X.tocsr() if issparse(adata.X) else csr_matrix(adata.X)

    print(f"Original matrix shape: {X.shape}")

    # Filter to keep only genes that have valid mappings
    valid_mask = np.array([f in gene_map for f in feature_names])
    valid_feature_names = feature_names[valid_mask]
    valid_gene_ids = np.array([gene_map[f] for f in valid_feature_names])

    X_valid = X[:, valid_mask]
    print(f"Valid genes matrix shape: {X_valid.shape}")

    # Sort genes by gene_id for consistent indexing
    sort_idx = np.argsort(valid_gene_ids)
    X_valid_sorted = X_valid[:, sort_idx]
    valid_gene_ids_sorted = valid_gene_ids[sort_idx]

    n_cells = X.shape[0]
    n_genes = max_gene_id + 1

    # Construct target sparse matrix using triplet format (row, col, data)
    X_valid_sorted = X_valid_sorted.tocoo()
    rows = X_valid_sorted.row
    cols = valid_gene_ids_sorted[X_valid_sorted.col]
    data = X_valid_sorted.data
    X_final = csr_matrix((data, (rows, cols)), shape=(n_cells, n_genes), dtype=X.dtype)

    print(f"Final matrix shape: {X_final.shape}")

    if normalize:
        print("Performing TP10K normalization...")
        X_final = normalize_tensor(
            X_final,
            min_genes=min_genes,
            scale_factor=scale_factor,
            apply_log1p=apply_log1p,
        )

    save_npz(output_path, X_final)
    print(f"Saved processed data to: {output_path}")

    return {"status": "saved", "path": output_path, "shape": X_final.shape}


def normalize_tensor(
    csr_matrix,
    min_genes: int = 200,
    scale_factor: float = 1e4,
    apply_log1p: bool = True,
):
    """Apply TP10K/CPM normalization to sparse matrix using CPU processing.

    单细胞测序数据归一化流程：
    1. 过滤低质量细胞（少于 min_genes 个基因表达）
    2. 对每个细胞进行归一化：x_ij_norm = (x_ij / sum(x_i)) * scale_factor
       - 消除测序深度（library size）差异
       - 使得细胞间表达量可比
    3. 可选的 log1p 变换，稳定方差

    使用对角稀疏矩阵左乘实现按行缩放，避免广播成稠密矩阵，保持内存高效。

    Args:
        csr_matrix: Input CSR sparse matrix containing gene expression data (cells × genes).
        min_genes: Minimum number of genes per cell threshold for filtering.
        scale_factor: Scaling factor. 1e4 for TP10K, 1e6 for CPM (counts per million).
        apply_log1p: Whether to apply log1p transformation after normalization.

    Returns:
        csr_matrix: Normalized sparse matrix with TP10K/CPM + optional log1p.
    """
    # 1) 过滤低质量细胞（表达基因数 < min_genes）
    valid_cells = np.diff(csr_matrix.indptr) >= min_genes
    csr_filtered = csr_matrix[valid_cells]
    print(f"Valid cells after filtering (>= {min_genes} genes): {valid_cells.sum()}")

    print("Using CPU for normalization...")

    # 2) 计算每个细胞的 library size（总 UMI counts）
    library_sizes = np.array(csr_filtered.sum(axis=1)).flatten()

    # 避免除以 0
    library_sizes[library_sizes == 0] = 1

    # 3) 用对角稀疏矩阵左乘实现按行缩放（内存高效，保持稀疏性）
    scale_factors = scale_factor / library_sizes
    D = diags(scale_factors, format="csr")
    csr_filtered = D @ csr_filtered

    print(f"Applied TP{int(scale_factor)} normalization")

    # 4) 可选的 log1p 变换
    if apply_log1p:
        csr_filtered.data = np.log1p(csr_filtered.data)
        print("Applied log1p transformation")

    print("CPU normalization completed")
    return csr_filtered


def find_all_h5ad_files(root_dir: str):
    """Recursively find all h5ad files in the given directory.

    Args:
        root_dir: Root directory to search for h5ad files.

    Returns:
        list: List of full paths to all found h5ad files.
    """
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
    normalize: bool = True,
    scale_factor: float = 1e4,
    min_genes: int = 200,
    apply_log1p: bool = True,
):
    """Process a single h5ad file to sparse tensor format.

    Args:
        h5ad_path: Path to the h5ad file to process.
        output_dir: Directory where the processed file will be saved.
        gene_map_path: Path to the gene mapping CSV file.
        normalize: Whether to apply normalization.
        scale_factor: Scaling factor for normalization.
        min_genes: Minimum number of genes per cell.
        apply_log1p: Whether to apply log1p transformation.

    Returns:
        str: Processing status message.
    """
    if not os.path.isfile(h5ad_path):
        logging.warning(f"File not found: {h5ad_path}")
        return f"Warning: File not found {h5ad_path}"

    basename = os.path.basename(h5ad_path).replace(".h5ad", "_processed.npz")
    output_path = os.path.join(output_dir, basename)
    print(f"Processing {h5ad_path} to {output_path}")

    try:
        res = process_h5ad_to_sparse_tensor(
            h5ad_path,
            output_path,
            gene_map_path,
            normalize,
            scale_factor,
            min_genes,
            apply_log1p,
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
    normalize: bool = True,
    scale_factor: float = 1e4,
    min_genes: int = 200,
    apply_log1p: bool = True,
):
    """Batch process all h5ad files in a folder using multiprocessing.

    Args:
        input_dir: Input directory containing h5ad files.
        output_dir: Output directory for processed npz files.
        gene_map_path: Path to the gene mapping CSV file.
        num_processes: Number of parallel processes (defaults to min(4, cpu_count)).
        normalize: Whether to apply normalization to the data.
        scale_factor: Scaling factor for normalization (1e4 for TP10K, 1e6 for CPM).
        min_genes: Minimum number of genes per cell for filtering.
        apply_log1p: Whether to apply log1p transformation.
    """
    os.makedirs(output_dir, exist_ok=True)
    logfile = setup_logging("integrated_preprocessing", "./logs")
    logging.info("Start integrated preprocessing...")

    print("=" * 60)
    print(f"Processing {input_dir} to {output_dir}")
    print("Using CPU for processing")
    print(f"Normalize: {normalize}")
    if normalize:
        print(f"  Method:        TP{int(scale_factor)} normalization")
        print(f"  Scale factor:  {scale_factor}")
        print(f"  Min genes:     {min_genes}")
        print(f"  Log1p:         {apply_log1p}")
    print("=" * 60)

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
            normalize=normalize,
            scale_factor=scale_factor,
            min_genes=min_genes,
            apply_log1p=apply_log1p,
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
        description="Integrated preprocessing of h5ad files with TP10K normalization and log1p transformation."
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
        "--no_normalize",
        action="store_true",
        help="Skip normalization step.",
    )
    parser.add_argument(
        "--scale_factor",
        type=float,
        default=1e4,
        help="Scaling factor: 1e4 for TP10K (default), 1e6 for CPM",
    )
    parser.add_argument(
        "--min_genes",
        type=int,
        default=200,
        help="Minimum number of genes per cell for filtering (default: 200).",
    )
    parser.add_argument(
        "--no_log1p",
        action="store_true",
        help="Disable log1p transformation after normalization",
    )

    args = parser.parse_args()

    preprocess_cellxgene_folder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        gene_map_path=args.gene_map_path,
        num_processes=args.num_processes,
        normalize=not args.no_normalize,
        scale_factor=args.scale_factor,
        min_genes=args.min_genes,
        apply_log1p=not args.no_log1p,
    )


if __name__ == "__main__":
    main()
