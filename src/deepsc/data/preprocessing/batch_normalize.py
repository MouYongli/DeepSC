import os

import numpy as np
from scipy import sparse
from scipy.sparse import diags

import argparse


def normalize_with_tp10k(
    csr: sparse.csr_matrix,
    scale_factor: float = 1e4,
    min_genes: int = 200,
    apply_log1p: bool = True,
) -> sparse.csr_matrix:
    """
    TP10K (or CPM) normalization + optional log1p transformation.

    单细胞测序数据归一化流程：
    1. 过滤低质量细胞（少于 min_genes 个基因表达）
    2. 对每个细胞进行归一化：x_ij_norm = (x_ij / sum(x_i)) * scale_factor
       - 消除测序深度（library size）差异
       - 使得细胞间表达量可比
    3. 可选的 log1p 变换，稳定方差

    使用对角稀疏矩阵左乘实现按行缩放，避免广播成稠密矩阵，保持内存高效。

    Args:
        csr (sparse.csr_matrix): Input sparse CSR matrix (cells × genes)
        scale_factor (float): Scaling factor. 1e4 for TP10K, 1e6 for CPM (counts per million)
        min_genes (int): Minimum number of genes expressed per cell to keep
        apply_log1p (bool): Whether to apply log1p transformation after normalization

    Returns:
        sparse.csr_matrix: Normalized sparse matrix with TP10K/CPM + optional log1p

    Mathematical formula:
        x_ij_norm = (x_ij / Σ_k x_ik) × scale_factor
        where x_ij is the count for gene j in cell i
    """
    # 1) 过滤低质量细胞（表达基因数 < min_genes）
    valid_cells = np.diff(csr.indptr) >= min_genes
    csr = csr[valid_cells]
    print(f"Valid cells after filtering (>= {min_genes} genes): {valid_cells.sum()}")

    # 2) 计算每个细胞的 library size（总 UMI counts）
    library_sizes = np.array(csr.sum(axis=1)).flatten()  # shape: (n_cells,)

    # 避免除以 0
    library_sizes[library_sizes == 0] = 1

    # 3) 用对角稀疏矩阵左乘实现按行缩放（内存高效，保持稀疏性）
    # D @ csr: 对角矩阵左乘 = 对每一行乘以对应的缩放因子
    scale_factors = scale_factor / library_sizes
    D = diags(scale_factors, format="csr")  # 创建对角稀疏矩阵
    csr = D @ csr  # 稀疏矩阵乘法，不会稠密化

    print(f"Applied TP{int(scale_factor)} normalization")

    # 4) 可选的 log1p 变换
    if apply_log1p:
        csr.data = np.log1p(csr.data)
        print("Applied log1p transformation")

    return csr


def normalize_tensor_no_scale(csr: sparse.csr_matrix) -> sparse.csr_matrix:
    """
    Normalize a sparse matrix without scaling, applying log1p transformation.

    ⚠️  DEPRECATED: 建议使用 normalize_with_tp10k() 以消除测序深度差异

    Filters out cells with fewer than 200 valid entries and applies log1p
    transformation to the remaining data.

    Args:
        csr (sparse.csr_matrix): Input sparse CSR matrix to be normalized

    Returns:
        sparse.csr_matrix: Normalized sparse matrix with log1p transformation applied
    """
    valid_cells = np.diff(csr.indptr) >= 200
    csr = csr[valid_cells]
    print(f"Valid cells: {valid_cells.sum()}")
    csr.data = np.log1p(csr.data)
    print(f"Normalized data: {csr.data}")
    return csr


def process_file(
    filepath: str,
    out_dir: str,
    scale_factor: float = 1e4,
    min_genes: int = 200,
    apply_log1p: bool = True,
    use_tp10k: bool = True,
) -> None:
    """
    Process a single sparse matrix file and save the normalized version.

    Loads a sparse matrix from either .npz or .mtx format, applies normalization,
    and saves the result as a .npz file with '_norm' suffix.

    Args:
        filepath (str): Path to the input file (.npz or .mtx format)
        out_dir (str): Output directory where normalized file will be saved
        scale_factor (float): Scaling factor for TP10K/CPM normalization
        min_genes (int): Minimum number of genes per cell
        apply_log1p (bool): Whether to apply log1p transformation
        use_tp10k (bool): Use TP10K normalization (True) or legacy method (False)

    Returns:
        None
    """
    filename = os.path.basename(filepath)
    name, ext = os.path.splitext(filename)
    if ext == ".npz":
        csr = sparse.load_npz(filepath)
        print(f"Loaded {filepath}")
    elif ext == ".mtx":
        csr = sparse.csr_matrix(sparse.mmread(filepath))
        print(f"Loaded {filepath}")
    else:
        print(f"Skipping unsupported file: {filename}")
        return

    # 选择归一化方法
    if use_tp10k:
        norm_csr = normalize_with_tp10k(
            csr, scale_factor=scale_factor, min_genes=min_genes, apply_log1p=apply_log1p
        )
    else:
        print("⚠️  Using legacy normalization (no TP10K scaling)")
        norm_csr = normalize_tensor_no_scale(csr)

    out_path = os.path.join(out_dir, name + "_norm.npz")
    sparse.save_npz(out_path, norm_csr)
    print(f"Saved normalized: {out_path}")


def main() -> None:
    """
    Main function for batch normalization of sparse matrices.

    Parses command line arguments and processes all .npz and .mtx files
    in the input directory, applying normalization and saving results
    to the output directory.

    Command line arguments:
        --input_dir: Input folder containing .npz or .mtx files
        --output_dir: Output folder (defaults to input_dir if not specified)
        --scale_factor: Scaling factor (1e4 for TP10K, 1e6 for CPM)
        --min_genes: Minimum number of genes per cell
        --no_log1p: Disable log1p transformation
        --no_tp10k: Use legacy normalization without TP10K scaling

    Returns:
        None
    """
    parser = argparse.ArgumentParser(
        description="Batch normalize all sparse matrices in a folder with TP10K normalization."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input folder containing .npz or .mtx files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output folder (default: input_dir)",
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
        help="Minimum number of genes per cell (default: 200)",
    )
    parser.add_argument(
        "--no_log1p",
        action="store_true",
        help="Disable log1p transformation after normalization",
    )
    parser.add_argument(
        "--no_tp10k",
        action="store_true",
        help="Use legacy normalization without TP10K scaling (not recommended)",
    )

    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Normalization Configuration:")
    print(f"  Input dir:     {input_dir}")
    print(f"  Output dir:    {output_dir}")
    if not args.no_tp10k:
        print(f"  Method:        TP{int(args.scale_factor)} normalization")
        print(f"  Scale factor:  {args.scale_factor}")
        print(f"  Min genes:     {args.min_genes}")
        print(f"  Log1p:         {not args.no_log1p}")
    else:
        print("  Method:        Legacy (no TP10K scaling)")
    print("=" * 60)

    for fname in os.listdir(input_dir):
        fpath = os.path.join(input_dir, fname)
        if os.path.isfile(fpath):
            print(f"\nProcessing: {fname}")
            process_file(
                fpath,
                output_dir,
                scale_factor=args.scale_factor,
                min_genes=args.min_genes,
                apply_log1p=not args.no_log1p,
                use_tp10k=not args.no_tp10k,
            )


if __name__ == "__main__":
    main()
