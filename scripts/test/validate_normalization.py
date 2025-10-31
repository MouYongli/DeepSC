"""Validate TP10K normalization on actual data."""

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse

import argparse


def validate_normalization(
    input_path: str, output_path: str = None, create_plots: bool = True
):
    """
    验证归一化后的数据是否正确。

    检查项：
    1. 每个细胞的 library size 是否都是 10000
    2. 稀疏性是否保持
    3. 数据分布是否合理
    """
    print("=" * 60)
    print("Normalization Validation Report")
    print("=" * 60)

    # 加载归一化后的数据
    print(f"\n1. Loading normalized data from: {input_path}")
    data = sparse.load_npz(input_path)
    print(f"   Shape: {data.shape} (cells × genes)")
    print(f"   Non-zero elements: {data.nnz:,} ({data.nnz / data.size * 100:.2f}%)")

    # 检查1: Library size (对 log1p 变换后的数据，需要先 expm1)
    print("\n2. Checking library sizes...")

    # 假设数据已经经过 log1p，先还原
    data_restored = data.copy()
    data_restored.data = np.expm1(data_restored.data)

    library_sizes = np.array(data_restored.sum(axis=1)).flatten()

    print(f"   Mean library size:   {library_sizes.mean():.2f}")
    print(f"   Std library size:    {library_sizes.std():.2f}")
    print(f"   Min library size:    {library_sizes.min():.2f}")
    print(f"   Max library size:    {library_sizes.max():.2f}")

    # 所有细胞的 library size 应该接近 10000
    expected_size = 10000
    tolerance = expected_size * 0.01  # 1% 容差

    if np.abs(library_sizes.mean() - expected_size) < tolerance:
        print(f"   ✅ Library sizes are close to {expected_size}")
    else:
        print(f"   ⚠️  Library sizes deviate from {expected_size}")

    # 检查2: 稀疏性
    print("\n3. Checking sparsity...")
    print(f"   Sparsity: {100 - data.nnz / data.size * 100:.2f}% zeros")

    # 检查3: 值的范围
    print("\n4. Checking value ranges (after log1p)...")
    print(f"   Min value:  {data.data.min():.4f}")
    print(f"   Max value:  {data.data.max():.4f}")
    print(f"   Mean value: {data.data.mean():.4f}")
    print(f"   Median:     {np.median(data.data):.4f}")

    # log1p 后的值应该 >= 0
    if data.data.min() >= 0:
        print("   ✅ All values are non-negative (correct for log1p)")
    else:
        print("   ❌ Found negative values (unexpected after log1p)")

    # 检查4: 每个细胞的基因表达数
    print("\n5. Checking genes per cell...")
    genes_per_cell = np.diff(data.indptr)
    print(f"   Mean genes/cell: {genes_per_cell.mean():.0f}")
    print(f"   Min genes/cell:  {genes_per_cell.min()}")
    print(f"   Max genes/cell:  {genes_per_cell.max()}")

    # 可视化
    if create_plots and output_path:
        print(f"\n6. Creating validation plots...")
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot 1: Library size 分布
        axes[0, 0].hist(library_sizes, bins=50, edgecolor="black")
        axes[0, 0].axvline(
            expected_size, color="r", linestyle="--", label=f"Expected: {expected_size}"
        )
        axes[0, 0].set_xlabel("Library Size (after expm1)")
        axes[0, 0].set_ylabel("Count")
        axes[0, 0].set_title("Library Size Distribution")
        axes[0, 0].legend()

        # Plot 2: 表达值分布（log1p 后）
        axes[0, 1].hist(data.data, bins=100, edgecolor="black")
        axes[0, 1].set_xlabel("Expression Value (after log1p)")
        axes[0, 1].set_ylabel("Count")
        axes[0, 1].set_title("Expression Value Distribution")
        axes[0, 1].set_yscale("log")

        # Plot 3: 每个细胞的基因数分布
        axes[1, 0].hist(genes_per_cell, bins=50, edgecolor="black")
        axes[1, 0].set_xlabel("Number of Genes")
        axes[1, 0].set_ylabel("Count")
        axes[1, 0].set_title("Genes per Cell Distribution")

        # Plot 4: 稀疏性模式 (前1000个细胞，前1000个基因)
        sample = data[:1000, :1000].toarray()
        axes[1, 1].imshow(
            sample > 0, aspect="auto", cmap="binary", interpolation="nearest"
        )
        axes[1, 1].set_xlabel("Genes (first 1000)")
        axes[1, 1].set_ylabel("Cells (first 1000)")
        axes[1, 1].set_title("Sparsity Pattern (white = expressed)")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"   ✅ Saved validation plot to: {output_path}")
        plt.close()

    print("\n" + "=" * 60)
    print("Validation complete!")
    print("=" * 60)

    return {
        "library_size_mean": library_sizes.mean(),
        "library_size_std": library_sizes.std(),
        "sparsity": 100 - data.nnz / data.size * 100,
        "min_value": data.data.min(),
        "max_value": data.data.max(),
        "mean_genes_per_cell": genes_per_cell.mean(),
    }


def compare_before_after(before_path: str, after_path: str):
    """比较归一化前后的数据。"""
    print("\n" + "=" * 60)
    print("Before/After Comparison")
    print("=" * 60)

    before = sparse.load_npz(before_path)
    after = sparse.load_npz(after_path)

    # 归一化前的 library sizes
    lib_before = np.array(before.sum(axis=1)).flatten()
    # 归一化后的 library sizes (需要先 expm1)
    after_restored = after.copy()
    after_restored.data = np.expm1(after_restored.data)
    lib_after = np.array(after_restored.sum(axis=1)).flatten()

    print(f"\nBefore normalization:")
    print(f"  Mean library size: {lib_before.mean():.0f}")
    print(f"  Std library size:  {lib_before.std():.0f}")
    print(
        f"  CV (coefficient of variation): {lib_before.std() / lib_before.mean():.4f}"
    )

    print(f"\nAfter TP10K normalization:")
    print(f"  Mean library size: {lib_after.mean():.0f}")
    print(f"  Std library size:  {lib_after.std():.0f}")
    print(f"  CV (coefficient of variation): {lib_after.std() / lib_after.mean():.4f}")

    # CV 应该显著降低
    cv_reduction = (
        ((lib_before.std() / lib_before.mean()) - (lib_after.std() / lib_after.mean()))
        / (lib_before.std() / lib_before.mean())
        * 100
    )

    print(f"\n✅ CV reduced by {cv_reduction:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Validate TP10K normalization on actual data."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to normalized .npz file",
    )
    parser.add_argument(
        "--output_plot",
        type=str,
        default="validation_plot.png",
        help="Path to save validation plot",
    )
    parser.add_argument(
        "--before",
        type=str,
        default=None,
        help="Path to pre-normalization data for comparison",
    )
    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="Skip plot generation",
    )

    args = parser.parse_args()

    # 验证归一化数据
    results = validate_normalization(
        args.input,
        output_path=args.output_plot if not args.no_plot else None,
        create_plots=not args.no_plot,
    )

    # 如果提供了归一化前的数据，做对比
    if args.before:
        compare_before_after(args.before, args.input)


if __name__ == "__main__":
    main()
