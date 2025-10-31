"""Compare expression value distributions between old and new normalization methods."""

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse

# 加载两个数据集
print("Loading data files...")
old_norm = sparse.load_npz(
    "/home/angli/baseline/DeepSC/data/processed/npz_data_before_shuffel/3ca/batch_19_norm.npz"
)
new_norm = sparse.load_npz(
    "/home/angli/baseline/DeepSC/data/npz_data_before_shuffel/3ca_new/batch_19_norm.npz"
)

print(f"\nOld normalization:")
print(f"  Shape: {old_norm.shape}")
print(f"  Non-zero elements: {old_norm.nnz:,}")
print(f"  Sparsity: {100 - old_norm.nnz / old_norm.size * 100:.2f}%")

print(f"\nNew normalization:")
print(f"  Shape: {new_norm.shape}")
print(f"  Non-zero elements: {new_norm.nnz:,}")
print(f"  Sparsity: {100 - new_norm.nnz / new_norm.size * 100:.2f}%")

# 提取非零表达值
print("\n" + "=" * 60)
print("Expression Value Statistics")
print("=" * 60)

old_values = old_norm.data
new_values = new_norm.data

print(f"\nOld normalization:")
print(f"  Min:     {old_values.min():.6f}")
print(f"  Max:     {old_values.max():.6f}")
print(f"  Mean:    {old_values.mean():.6f}")
print(f"  Median:  {np.median(old_values):.6f}")
print(f"  Std:     {old_values.std():.6f}")

print(f"\nNew normalization (TP10K + log1p):")
print(f"  Min:     {new_values.min():.6f}")
print(f"  Max:     {new_values.max():.6f}")
print(f"  Mean:    {new_values.mean():.6f}")
print(f"  Median:  {np.median(new_values):.6f}")
print(f"  Std:     {new_values.std():.6f}")

# 计算百分位数
print(f"\nPercentiles comparison:")
percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
print(f"{'Percentile':<12} {'Old':<12} {'New':<12} {'Ratio':<12}")
print("-" * 50)
for p in percentiles:
    old_p = np.percentile(old_values, p)
    new_p = np.percentile(new_values, p)
    ratio = new_p / old_p if old_p > 0 else 0
    print(f"{p}%{'':<9} {old_p:>11.4f}  {new_p:>11.4f}  {ratio:>11.4f}")

# 检查 library sizes (需要先 expm1 还原)
print("\n" + "=" * 60)
print("Library Size Analysis (after expm1 to restore)")
print("=" * 60)

old_restored = old_norm.copy()
old_restored.data = np.expm1(old_restored.data)
old_lib_sizes = np.array(old_restored.sum(axis=1)).flatten()

new_restored = new_norm.copy()
new_restored.data = np.expm1(new_restored.data)
new_lib_sizes = np.array(new_restored.sum(axis=1)).flatten()

print(f"\nOld normalization library sizes:")
print(f"  Mean: {old_lib_sizes.mean():.2f}")
print(f"  Std:  {old_lib_sizes.std():.2f}")
print(f"  CV:   {old_lib_sizes.std() / old_lib_sizes.mean():.6f}")
print(f"  Min:  {old_lib_sizes.min():.2f}")
print(f"  Max:  {old_lib_sizes.max():.2f}")

print(f"\nNew normalization library sizes:")
print(f"  Mean: {new_lib_sizes.mean():.2f}")
print(f"  Std:  {new_lib_sizes.std():.2f}")
print(f"  CV:   {new_lib_sizes.std() / new_lib_sizes.mean():.6f}")
print(f"  Min:  {new_lib_sizes.min():.2f}")
print(f"  Max:  {new_lib_sizes.max():.2f}")

# 创建对比图
print("\n" + "=" * 60)
print("Creating visualization...")
print("=" * 60)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle(
    "Normalization Methods Comparison\nOld vs New (TP10K + log1p)",
    fontsize=16,
    fontweight="bold",
)

# Row 1: Old normalization
# Plot 1: Expression value distribution (old)
axes[0, 0].hist(old_values, bins=100, edgecolor="black", alpha=0.7, color="coral")
axes[0, 0].set_xlabel("Expression Value (after log)")
axes[0, 0].set_ylabel("Count")
axes[0, 0].set_title("Old Normalization: Expression Distribution")
axes[0, 0].set_yscale("log")
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Library size distribution (old)
axes[0, 1].hist(old_lib_sizes, bins=100, edgecolor="black", alpha=0.7, color="coral")
axes[0, 1].set_xlabel("Library Size (after expm1)")
axes[0, 1].set_ylabel("Count")
axes[0, 1].set_title(
    f"Old: Library Size Distribution\nCV={old_lib_sizes.std()/old_lib_sizes.mean():.4f}"
)
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Cumulative distribution (old)
sorted_old = np.sort(old_values)
cumulative_old = np.arange(1, len(sorted_old) + 1) / len(sorted_old)
axes[0, 2].plot(sorted_old, cumulative_old, linewidth=2, color="coral")
axes[0, 2].set_xlabel("Expression Value")
axes[0, 2].set_ylabel("Cumulative Probability")
axes[0, 2].set_title("Old: Cumulative Distribution")
axes[0, 2].grid(True, alpha=0.3)

# Row 2: New normalization (TP10K + log1p)
# Plot 4: Expression value distribution (new)
axes[1, 0].hist(new_values, bins=100, edgecolor="black", alpha=0.7, color="skyblue")
axes[1, 0].set_xlabel("Expression Value (after log1p)")
axes[1, 0].set_ylabel("Count")
axes[1, 0].set_title("New (TP10K + log1p): Expression Distribution")
axes[1, 0].set_yscale("log")
axes[1, 0].grid(True, alpha=0.3)

# Plot 5: Library size distribution (new)
axes[1, 1].hist(new_lib_sizes, bins=100, edgecolor="black", alpha=0.7, color="skyblue")
axes[1, 1].axvline(
    10000, color="red", linestyle="--", linewidth=2, label="Expected: 10000"
)
axes[1, 1].set_xlabel("Library Size (after expm1)")
axes[1, 1].set_ylabel("Count")
axes[1, 1].set_title(
    f"New: Library Size Distribution\nCV={new_lib_sizes.std()/new_lib_sizes.mean():.6f}"
)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# Plot 6: Cumulative distribution (new)
sorted_new = np.sort(new_values)
cumulative_new = np.arange(1, len(sorted_new) + 1) / len(sorted_new)
axes[1, 2].plot(sorted_new, cumulative_new, linewidth=2, color="skyblue")
axes[1, 2].set_xlabel("Expression Value")
axes[1, 2].set_ylabel("Cumulative Probability")
axes[1, 2].set_title("New: Cumulative Distribution")
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
output_path = (
    "/home/angli/baseline/DeepSC/tests/test_output/normalization_comparison.png"
)
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\n✅ Saved comparison plot to: {output_path}")

# 创建叠加对比图
fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle("Direct Comparison: Old vs New", fontsize=14, fontweight="bold")

# Overlaid histograms
axes2[0].hist(
    old_values, bins=100, alpha=0.5, label="Old", color="coral", edgecolor="black"
)
axes2[0].hist(
    new_values,
    bins=100,
    alpha=0.5,
    label="New (TP10K+log1p)",
    color="skyblue",
    edgecolor="black",
)
axes2[0].set_xlabel("Expression Value")
axes2[0].set_ylabel("Count")
axes2[0].set_title("Expression Value Distribution")
axes2[0].set_yscale("log")
axes2[0].legend()
axes2[0].grid(True, alpha=0.3)

# Overlaid CDFs
axes2[1].plot(sorted_old, cumulative_old, linewidth=2, label="Old", color="coral")
axes2[1].plot(
    sorted_new, cumulative_new, linewidth=2, label="New (TP10K+log1p)", color="skyblue"
)
axes2[1].set_xlabel("Expression Value")
axes2[1].set_ylabel("Cumulative Probability")
axes2[1].set_title("Cumulative Distribution Comparison")
axes2[1].legend()
axes2[1].grid(True, alpha=0.3)

plt.tight_layout()
output_path2 = "/home/angli/baseline/DeepSC/tests/test_output/normalization_overlay.png"
plt.savefig(output_path2, dpi=150, bbox_inches="tight")
print(f"✅ Saved overlay plot to: {output_path2}")

print("\n" + "=" * 60)
print("Analysis complete!")
print("=" * 60)
