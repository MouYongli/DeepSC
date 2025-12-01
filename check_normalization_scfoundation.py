#!/usr/bin/env python3
"""
快速检查scfoundation数据集是否进行了log归一化
"""

from pathlib import Path

import numpy as np
import scanpy as sc


def check_normalization(h5ad_path):
    """
    检查数据集是否已log归一化

    判断依据:
    - 如果最大值 > 50: 可能是原始count
    - 如果最大值 < 20: 可能是log归一化
    - 如果值包含很多整数: 可能是count
    """
    print(f"\n{'='*80}")
    print(f"检查文件: {Path(h5ad_path).name}")
    print(f"{'='*80}")

    # 读取数据
    adata = sc.read_h5ad(h5ad_path)

    # 获取表达矩阵
    if hasattr(adata.X, "toarray"):
        expr_matrix = adata.X.toarray()
    else:
        expr_matrix = adata.X

    # 基本统计
    min_val = expr_matrix.min()
    max_val = expr_matrix.max()
    mean_val = expr_matrix.mean()
    median_val = np.median(expr_matrix)

    print(f"\n数据集形状: {adata.shape} (cells × genes)")
    print("\n基本统计:")
    print(f"  - 最小值: {min_val:.4f}")
    print(f"  - 最大值: {max_val:.4f}")
    print(f"  - 平均值: {mean_val:.4f}")
    print(f"  - 中位数: {median_val:.4f}")

    # 获取非零值
    nonzero_values = expr_matrix[expr_matrix > 0]
    zero_percentage = (expr_matrix == 0).sum() / expr_matrix.size * 100

    print("\n稀疏性:")
    print(f"  - 零值比例: {zero_percentage:.2f}%")

    if len(nonzero_values) > 0:
        print("\n非零值统计:")
        print(f"  - 非零值数量: {len(nonzero_values):,}")
        print(f"  - 非零值最小值: {nonzero_values.min():.4f}")
        print(f"  - 非零值最大值: {nonzero_values.max():.4f}")
        print(f"  - 非零值平均值: {nonzero_values.mean():.4f}")
        print(f"  - 非零值中位数: {np.median(nonzero_values):.4f}")

        # 检查是否是整数(count数据的特征)
        sample_size = min(10000, len(nonzero_values))
        sample = np.random.choice(nonzero_values, sample_size, replace=False)
        is_integer = np.allclose(sample, np.round(sample))
        integer_percentage = (
            (np.isclose(sample, np.round(sample))).sum() / len(sample) * 100
        )

        print("\n整数特征:")
        print(f"  - 采样大小: {sample_size:,}")
        print(f"  - 整数占比: {integer_percentage:.2f}%")

    # 判断是否归一化
    print(f"\n{'='*80}")
    print("归一化状态判断:")
    print(f"{'='*80}")

    is_normalized = None
    reasons = []

    if max_val > 100:
        is_normalized = False
        reasons.append(f"❌ 最大值 {max_val:.2f} > 100 (明显是原始count)")
    elif max_val > 50:
        is_normalized = False
        reasons.append(f"⚠️  最大值 {max_val:.2f} > 50 (可能是count数据)")
    elif max_val < 20:
        is_normalized = True
        reasons.append(f"✅ 最大值 {max_val:.2f} < 20 (符合log归一化范围)")

    if len(nonzero_values) > 0:
        if integer_percentage > 90:
            is_normalized = False
            reasons.append(
                f"❌ {integer_percentage:.1f}% 的非零值是整数 (count数据特征)"
            )
        elif integer_percentage < 10:
            is_normalized = True
            reasons.append(
                f"✅ 仅{integer_percentage:.1f}% 的非零值是整数 (log归一化特征)"
            )

    if mean_val > 10:
        is_normalized = False
        reasons.append(f"❌ 平均值 {mean_val:.2f} > 10 (count数据特征)")

    # 输出结论
    print()
    for reason in reasons:
        print(f"  {reason}")

    print("\n结论: ", end="")
    if is_normalized is True:
        print("✅ 数据已进行log归一化")
    elif is_normalized is False:
        print("❌ 数据是原始count,未进行log归一化")
    else:
        print("⚠️  无法确定,需要人工检查")

    return {
        "file": Path(h5ad_path).name,
        "shape": adata.shape,
        "min": min_val,
        "max": max_val,
        "mean": mean_val,
        "median": median_val,
        "zero_pct": zero_percentage,
        "is_normalized": is_normalized,
    }


def main():
    base_dir = Path("/home/angli/DeepSC/data/processed/baseline/scfoundation")

    datasets = [
        "hPancreas_merged.h5ad",
        "myeloid_merged.h5ad",
        "segerstolpe_merged.h5ad",
        "zheng_merged.h5ad",
    ]

    results = []

    for dataset in datasets:
        dataset_path = base_dir / dataset
        if dataset_path.exists():
            result = check_normalization(str(dataset_path))
            results.append(result)
        else:
            print(f"警告: 文件不存在 - {dataset_path}")

    # 打印汇总表
    print(f"\n\n{'='*100}")
    print("归一化状态汇总")
    print(f"{'='*100}\n")

    print(
        f"{'数据集':<30} {'形状':<20} {'最大值':<12} {'平均值':<12} {'归一化状态':<15}"
    )
    print("-" * 100)

    for result in results:
        dataset_name = result["file"].replace("_merged.h5ad", "")
        shape_str = f"{result['shape'][0]:,} × {result['shape'][1]:,}"
        status = "✅ 已归一化" if result["is_normalized"] else "❌ 未归一化"

        print(
            f"{dataset_name:<30} {shape_str:<20} {result['max']:<12.2f} "
            f"{result['mean']:<12.4f} {status:<15}"
        )

    print(f"\n{'='*100}\n")


if __name__ == "__main__":
    main()
