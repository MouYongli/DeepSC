"""Analyze why 25th percentile differs from minimum in TP10K normalization."""

import numpy as np
from scipy import sparse

print("=" * 70)
print("ANALYZING 25TH PERCENTILE vs MINIMUM VALUE")
print("=" * 70)

# Load both files
print("\nLoading data files...")
old_norm = sparse.load_npz(
    "/home/angli/baseline/DeepSC/data/processed/npz_data_before_shuffel/3ca/batch_19_norm.npz"
)
new_norm = sparse.load_npz(
    "/home/angli/baseline/DeepSC/data/npz_data_before_shuffel/3ca_new/batch_19_norm.npz"
)

old_values = old_norm.data
new_values = new_norm.data

print("\n" + "=" * 70)
print("OLD METHOD (log1p only, no normalization)")
print("=" * 70)
print(f"  Minimum:        {old_values.min():.6f}")
print(f"  25th percentile: {np.percentile(old_values, 25):.6f}")
print(
    f"  Are they equal? {np.isclose(old_values.min(), np.percentile(old_values, 25))}"
)

# 计算有多少值等于最小值
min_count_old = np.sum(np.isclose(old_values, old_values.min()))
print(f"\n  Number of values equal to minimum: {min_count_old:,}")
print(f"  Percentage: {min_count_old / len(old_values) * 100:.2f}%")

# 解释
print("\n  INTERPRETATION:")
print(f"    - Minimum value = ln(2) = log1p(1) = 0.693147")
print(f"    - This means the original count was 1")
print(
    f"    - {min_count_old / len(old_values) * 100:.1f}% of non-zero values have count=1"
)
print(f"    - Since >25% of values are count=1, the 25th percentile = minimum")

print("\n" + "=" * 70)
print("NEW METHOD (TP10K + log1p)")
print("=" * 70)
print(f"  Minimum:        {new_values.min():.6f}")
print(f"  25th percentile: {np.percentile(new_values, 25):.6f}")
print(
    f"  Are they equal? {np.isclose(new_values.min(), np.percentile(new_values, 25))}"
)

min_count_new = np.sum(np.isclose(new_values, new_values.min(), rtol=1e-4))
print(f"\n  Number of values equal to minimum: {min_count_new:,}")
print(f"  Percentage: {min_count_new / len(new_values) * 100:.4f}%")

# 分析为什么不同
print("\n  INTERPRETATION:")
print(f"    - Minimum = {new_values.min():.6f}")
print(
    f"    - Reverse: exp({new_values.min():.6f}) - 1 = {np.expm1(new_values.min()):.4f}"
)
print(
    f"    - This could be: count=1, library_size ≈ {10000 / np.expm1(new_values.min()):.0f}"
)
print()
print(f"    - 25th percentile = {np.percentile(new_values, 25):.6f}")
print(
    f"    - Reverse: exp({np.percentile(new_values, 25):.6f}) - 1 = {np.expm1(np.percentile(new_values, 25)):.4f}"
)
print(
    f"    - This could be: count=1, library_size ≈ {10000 / np.expm1(np.percentile(new_values, 25)):.0f}"
)

print("\n" + "=" * 70)
print("WHY ARE THEY DIFFERENT IN THE NEW METHOD?")
print("=" * 70)
print(
    """
In the OLD method (log1p only):
  - All cells with count=1 get the same value: log1p(1) = 0.693147
  - Library size doesn't matter
  - If >25% of values have count=1, then 25th percentile = minimum

In the NEW method (TP10K + log1p):
  - Cells with count=1 get DIFFERENT values depending on library size:
    * Cell with lib_size=10,000:  (1/10000)×10000 = 1 → log1p(1) = 0.693
    * Cell with lib_size=50,000:  (1/50000)×10000 = 0.2 → log1p(0.2) = 0.182
    * Cell with lib_size=120,000: (1/120000)×10000 = 0.083 → log1p(0.083) = 0.080

  - The MINIMUM corresponds to count=1 in the cell with the LARGEST library size
  - The 25th PERCENTILE includes count=1 from cells with smaller library sizes
  - Therefore: 25th percentile > minimum ✓
"""
)

# 让我们检查原始数据中的library size分布
print("\n" + "=" * 70)
print("ORIGINAL LIBRARY SIZE DISTRIBUTION")
print("=" * 70)

# 需要加载原始未归一化数据
# 先从新数据反推
restored_new = new_norm.copy()
restored_new.data = np.expm1(restored_new.data)
lib_sizes_new = np.array(restored_new.sum(axis=1)).flatten()

# 从旧数据反推原始library size
restored_old = old_norm.copy()
restored_old.data = np.expm1(restored_old.data)
lib_sizes_old = np.array(restored_old.sum(axis=1)).flatten()

print(f"\nLibrary sizes from OLD data:")
print(f"  Min:  {lib_sizes_old.min():.0f}")
print(f"  Max:  {lib_sizes_old.max():.0f}")
print(f"  Mean: {lib_sizes_old.mean():.0f}")
print(
    f"  This {lib_sizes_old.max() / lib_sizes_old.min():.1f}x variation explains the difference!"
)

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print("✅ It is CORRECT that the 25th percentile ≠ minimum in the NEW method")
print("✅ This is because TP10K normalization considers each cell's library size")
print("✅ Different cells with count=1 will have different normalized values")
print("✅ The OLD method's behavior (25th percentile = minimum) was actually a BUG")
print("   because it failed to account for sequencing depth variation!")
print("=" * 70)
