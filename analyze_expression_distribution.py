#!/usr/bin/env python3
"""
分析scRNA-seq数据集的表达值分布
"""

from pathlib import Path

import numpy as np
import scanpy as sc


def analyze_expression_distribution(h5ad_path):
    """
    分析单个h5ad文件的表达值分布

    Args:
        h5ad_path: h5ad文件路径

    Returns:
        dict: 包含各种统计信息的字典
    """
    print(f"\n{'='*80}")
    print(f"分析数据集: {Path(h5ad_path).name}")
    print(f"{'='*80}")

    # 读取数据
    adata = sc.read_h5ad(h5ad_path)

    # 获取表达矩阵 (可能是sparse或dense)
    if hasattr(adata.X, "toarray"):
        # Sparse matrix
        expr_matrix = adata.X.toarray()
    else:
        # Dense matrix
        expr_matrix = adata.X

    # 展平为一维数组
    expr_values = expr_matrix.flatten()
    total_values = len(expr_values)

    print(f"数据集形状: {adata.shape} (cells × genes)")
    print(f"总表达值数量: {total_values:,}")
    print("\n基本统计:")
    print(f"  - 最小值: {expr_values.min():.4f}")
    print(f"  - 最大值: {expr_values.max():.4f}")
    print(f"  - 平均值: {expr_values.mean():.4f}")
    print(f"  - 中位数: {np.median(expr_values):.4f}")
    print(f"  - 标准差: {expr_values.std():.4f}")

    # 计算各个区间的百分比
    ranges = {
        "等于0": (expr_values == 0).sum(),
        "(0, 1]": ((expr_values > 0) & (expr_values <= 1)).sum(),
        "(1, 10]": ((expr_values > 1) & (expr_values <= 10)).sum(),
        "(10, 50]": ((expr_values > 10) & (expr_values <= 50)).sum(),
        "(50, 100]": ((expr_values > 50) & (expr_values <= 100)).sum(),
        "(100, 200]": ((expr_values > 100) & (expr_values <= 200)).sum(),
        "(200, 300]": ((expr_values > 200) & (expr_values <= 300)).sum(),
        ">300": (expr_values > 300).sum(),
    }

    print("\n表达值分布:")
    print(f"{'区间':<15} {'数量':>15} {'百分比':>10}")
    print("-" * 42)

    results = {}
    for range_name, count in ranges.items():
        percentage = (count / total_values) * 100
        print(f"{range_name:<15} {count:>15,} {percentage:>9.2f}%")
        results[range_name] = {"count": count, "percentage": percentage}

    # 额外的稀疏性统计
    zero_percentage = (expr_values == 0).sum() / total_values * 100
    nonzero_percentage = 100 - zero_percentage

    print("\n稀疏性统计:")
    print(f"  - 零值比例: {zero_percentage:.2f}%")
    print(f"  - 非零值比例: {nonzero_percentage:.2f}%")
    print(f"  - 数据稀疏度: {zero_percentage:.2f}%")

    # 非零值的统计
    nonzero_values = expr_values[expr_values > 0]
    if len(nonzero_values) > 0:
        print("\n非零值统计:")
        print(f"  - 非零值数量: {len(nonzero_values):,}")
        print(f"  - 非零值平均值: {nonzero_values.mean():.4f}")
        print(f"  - 非零值中位数: {np.median(nonzero_values):.4f}")
        print(f"  - 非零值标准差: {nonzero_values.std():.4f}")
        print(f"  - 非零值最小值: {nonzero_values.min():.4f}")
        print(f"  - 非零值最大值: {nonzero_values.max():.4f}")

    return {
        "dataset": Path(h5ad_path).name,
        "n_cells": adata.n_obs,
        "n_genes": adata.n_vars,
        "total_values": total_values,
        "min": float(expr_values.min()),
        "max": float(expr_values.max()),
        "mean": float(expr_values.mean()),
        "median": float(np.median(expr_values)),
        "std": float(expr_values.std()),
        "zero_percentage": zero_percentage,
        "nonzero_percentage": nonzero_percentage,
        "distribution": results,
    }


def create_summary_table(all_results):
    """
    创建汇总表格
    """
    print(f"\n\n{'='*100}")
    print("所有数据集汇总")
    print(f"{'='*100}\n")

    # 创建数据集基本信息表
    print("数据集基本信息:")
    print(f"{'数据集':<20} {'细胞数':>10} {'基因数':>10} {'总值数':>15} {'稀疏度':>10}")
    print("-" * 68)
    for result in all_results:
        print(
            f"{result['dataset']:<20} {result['n_cells']:>10,} {result['n_genes']:>10,} "
            f"{result['total_values']:>15,} {result['zero_percentage']:>9.2f}%"
        )

    # 创建表达值范围分布对比表
    print("\n\n表达值区间分布对比 (百分比):")

    # 获取所有区间名称
    ranges = list(all_results[0]["distribution"].keys())

    # 打印表头
    header = f"{'区间':<15}"
    for result in all_results:
        dataset_name = result["dataset"].replace("_train.h5ad", "")
        header += f" {dataset_name:>15}"
    print(header)
    print("-" * (15 + 16 * len(all_results)))

    # 打印每个区间的数据
    for range_name in ranges:
        row = f"{range_name:<15}"
        for result in all_results:
            percentage = result["distribution"][range_name]["percentage"]
            row += f" {percentage:>14.2f}%"
        print(row)

    # 创建统计信息对比表
    print("\n\n基本统计对比:")
    print(f"{'统计量':<15}", end="")
    for result in all_results:
        dataset_name = result["dataset"].replace("_train.h5ad", "")
        print(f" {dataset_name:>15}", end="")
    print()
    print("-" * (15 + 16 * len(all_results)))

    stats = ["min", "max", "mean", "median", "std"]
    stat_names = {
        "min": "最小值",
        "max": "最大值",
        "mean": "平均值",
        "median": "中位数",
        "std": "标准差",
    }

    for stat in stats:
        print(f"{stat_names[stat]:<15}", end="")
        for result in all_results:
            print(f" {result[stat]:>15.4f}", end="")
        print()


def main():
    # 数据集路径
    base_dir = Path("/home/angli/DeepSC/data/processed/baseline/scgpt")
    datasets = [
        "hPancreas_train.h5ad",
        "myeloid_train.h5ad",
        "segerstolpe_train.h5ad",
        "zheng_train.h5ad",
    ]

    all_results = []

    # 分析每个数据集
    for dataset in datasets:
        dataset_path = base_dir / dataset
        if dataset_path.exists():
            result = analyze_expression_distribution(str(dataset_path))
            all_results.append(result)
        else:
            print(f"警告: 文件不存在 - {dataset_path}")

    # 创建汇总表
    if all_results:
        create_summary_table(all_results)

    print(f"\n{'='*100}")
    print("分析完成!")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    main()
