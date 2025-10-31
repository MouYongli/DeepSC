#!/usr/bin/env python3
"""
比较两个CSR格式npz文件的数值分布
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import sparse, stats


def load_csr_from_npz(filepath):
    """从npz文件加载CSR矩阵"""
    data = np.load(filepath)
    print(f"\n文件: {filepath}")
    print(f"包含的键: {list(data.keys())}")

    # 尝试不同的键名
    if "data" in data.keys():
        # 如果是完整的CSR格式
        if (
            "indices" in data.keys()
            and "indptr" in data.keys()
            and "shape" in data.keys()
        ):
            csr_matrix = sparse.csr_matrix(
                (data["data"], data["indices"], data["indptr"]),
                shape=tuple(data["shape"]),
            )
            return csr_matrix
        else:
            # 可能只是普通数组
            return data["data"]
    elif "arr_0" in data.keys():
        return data["arr_0"]
    else:
        # 返回第一个键对应的数据
        first_key = list(data.keys())[0]
        return data[first_key]


def get_nonzero_values(matrix):
    """获取矩阵的非零值"""
    if sparse.issparse(matrix):
        return matrix.data
    else:
        return matrix[matrix != 0]


def compare_distributions(file1, file2):
    """比较两个文件的数值分布"""
    print("=" * 80)
    print("比较CSR数据文件的数值分布")
    print("=" * 80)

    # 加载数据
    print("\n加载数据...")
    matrix1 = load_csr_from_npz(file1)
    matrix2 = load_csr_from_npz(file2)

    # 获取非零值
    values1 = get_nonzero_values(matrix1)
    values2 = get_nonzero_values(matrix2)

    print(f"\n数据1形状: {matrix1.shape if hasattr(matrix1, 'shape') else 'N/A'}")
    print(f"数据2形状: {matrix2.shape if hasattr(matrix2, 'shape') else 'N/A'}")

    # 基本统计信息
    print("\n" + "=" * 80)
    print("基本统计信息")
    print("=" * 80)

    print(f"\n文件1: {file1}")
    print(f"  非零元素数量: {len(values1)}")
    print(f"  均值: {np.mean(values1):.6f}")
    print(f"  中位数: {np.median(values1):.6f}")
    print(f"  标准差: {np.std(values1):.6f}")
    print(f"  最小值: {np.min(values1):.6f}")
    print(f"  最大值: {np.max(values1):.6f}")
    print(f"  25%分位数: {np.percentile(values1, 25):.6f}")
    print(f"  75%分位数: {np.percentile(values1, 75):.6f}")

    print(f"\n文件2: {file2}")
    print(f"  非零元素数量: {len(values2)}")
    print(f"  均值: {np.mean(values2):.6f}")
    print(f"  中位数: {np.median(values2):.6f}")
    print(f"  标准差: {np.std(values2):.6f}")
    print(f"  最小值: {np.min(values2):.6f}")
    print(f"  最大值: {np.max(values2):.6f}")
    print(f"  25%分位数: {np.percentile(values2, 25):.6f}")
    print(f"  75%分位数: {np.percentile(values2, 75):.6f}")

    # 统计检验
    print("\n" + "=" * 80)
    print("统计检验")
    print("=" * 80)

    # Kolmogorov-Smirnov检验
    ks_stat, ks_pvalue = stats.ks_2samp(values1, values2)
    print(f"\nKolmogorov-Smirnov检验:")
    print(f"  KS统计量: {ks_stat:.6f}")
    print(f"  p值: {ks_pvalue:.6e}")
    print(
        f"  结论: {'分布显著不同' if ks_pvalue < 0.05 else '分布无显著差异'} (α=0.05)"
    )

    # Mann-Whitney U检验 (非参数检验)
    if len(values1) > 0 and len(values2) > 0:
        u_stat, u_pvalue = stats.mannwhitneyu(values1, values2, alternative="two-sided")
        print(f"\nMann-Whitney U检验:")
        print(f"  U统计量: {u_stat:.6f}")
        print(f"  p值: {u_pvalue:.6e}")
        print(
            f"  结论: {'中位数显著不同' if u_pvalue < 0.05 else '中位数无显著差异'} (α=0.05)"
        )

    # 可视化
    print("\n" + "=" * 80)
    print("生成可视化图表...")
    print("=" * 80)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. 直方图对比
    ax1 = axes[0, 0]
    ax1.hist(
        values1,
        bins=50,
        alpha=0.5,
        label="文件1 (processed)",
        density=True,
        color="blue",
    )
    ax1.hist(
        values2, bins=50, alpha=0.5, label="文件2 (new)", density=True, color="red"
    )
    ax1.set_xlabel("数值")
    ax1.set_ylabel("密度")
    ax1.set_title("数值分布直方图")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 累积分布函数
    ax2 = axes[0, 1]
    sorted_vals1 = np.sort(values1)
    sorted_vals2 = np.sort(values2)
    cdf1 = np.arange(1, len(sorted_vals1) + 1) / len(sorted_vals1)
    cdf2 = np.arange(1, len(sorted_vals2) + 1) / len(sorted_vals2)
    ax2.plot(sorted_vals1, cdf1, label="文件1 (processed)", alpha=0.7, color="blue")
    ax2.plot(sorted_vals2, cdf2, label="文件2 (new)", alpha=0.7, color="red")
    ax2.set_xlabel("数值")
    ax2.set_ylabel("累积概率")
    ax2.set_title("累积分布函数 (CDF)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 箱线图对比
    ax3 = axes[1, 0]
    bp = ax3.boxplot(
        [values1, values2],
        labels=["文件1\n(processed)", "文件2\n(new)"],
        patch_artist=True,
        showmeans=True,
    )
    bp["boxes"][0].set_facecolor("lightblue")
    bp["boxes"][1].set_facecolor("lightcoral")
    ax3.set_ylabel("数值")
    ax3.set_title("箱线图对比")
    ax3.grid(True, alpha=0.3)

    # 4. Q-Q图
    ax4 = axes[1, 1]
    # 对两个分布进行采样以便绘制Q-Q图
    sample_size = min(len(values1), len(values2), 10000)
    sample1 = (
        np.random.choice(values1, size=sample_size, replace=False)
        if len(values1) > sample_size
        else values1
    )
    sample2 = (
        np.random.choice(values2, size=sample_size, replace=False)
        if len(values2) > sample_size
        else values2
    )

    quantiles1 = np.percentile(sample1, np.linspace(0, 100, 100))
    quantiles2 = np.percentile(sample2, np.linspace(0, 100, 100))

    ax4.scatter(quantiles1, quantiles2, alpha=0.5, s=20)
    min_val = min(quantiles1.min(), quantiles2.min())
    max_val = max(quantiles1.max(), quantiles2.max())
    ax4.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="y=x")
    ax4.set_xlabel("文件1分位数")
    ax4.set_ylabel("文件2分位数")
    ax4.set_title("Q-Q图")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = (
        "/home/angli/baseline/DeepSC/tests/test_output/distribution_comparison.png"
    )
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n图表已保存至: {output_path}")

    # 额外的比较信息
    print("\n" + "=" * 80)
    print("差异分析")
    print("=" * 80)
    print(f"\n均值差异: {np.mean(values1) - np.mean(values2):.6f}")
    print(
        f"均值相对差异: {(np.mean(values1) - np.mean(values2)) / np.mean(values2) * 100:.2f}%"
    )
    print(f"标准差差异: {np.std(values1) - np.std(values2):.6f}")
    print(f"非零元素数量差异: {len(values1) - len(values2)}")

    return values1, values2


if __name__ == "__main__":
    file1 = "/home/angli/baseline/DeepSC/data/processed/npz_data_before_shuffel/3ca/batch_19_norm.npz"
    file2 = "/home/angli/baseline/DeepSC/data/npz_data_before_shuffel/3ca_new/batch_19_norm.npz"

    values1, values2 = compare_distributions(file1, file2)

    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)
