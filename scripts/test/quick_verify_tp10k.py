"""Quick verification of TP10K normalization."""

import numpy as np
from scipy import sparse

print("Loading new normalized data...")
new_norm = sparse.load_npz(
    "/home/angli/baseline/DeepSC/data/npz_data_before_shuffel/3ca_new/batch_19_norm.npz"
)

print(f"Shape: {new_norm.shape}")
print(f"Non-zero elements: {new_norm.nnz:,}")

# 还原: expm1 to reverse log1p
print("\nRestoring original values (expm1)...")
restored = new_norm.copy()
restored.data = np.expm1(restored.data)

# 计算每个细胞的 library size
print("\nCalculating library sizes per cell...")
lib_sizes = np.array(restored.sum(axis=1)).flatten()

print(f"\n{'='*60}")
print("LIBRARY SIZE ANALYSIS (Should be ~10,000 after TP10K)")
print(f"{'='*60}")
print(f"  Mean:   {lib_sizes.mean():.2f}")
print(f"  Median: {np.median(lib_sizes):.2f}")
print(f"  Std:    {lib_sizes.std():.2f}")
print(f"  CV:     {lib_sizes.std() / lib_sizes.mean():.6f}")
print(f"  Min:    {lib_sizes.min():.2f}")
print(f"  Max:    {lib_sizes.max():.2f}")

# 检查是否接近10000
print(f"\n{'='*60}")
if 9500 <= lib_sizes.mean() <= 10500:
    print("✅ PASS: Mean library size is close to 10,000")
    print("✅ TP10K normalization appears CORRECT")
else:
    print("❌ FAIL: Mean library size is NOT close to 10,000")
    print("❌ TP10K normalization may be INCORRECT")
print(f"{'='*60}")

# 检查变异系数
if lib_sizes.std() / lib_sizes.mean() < 0.01:
    print("✅ PASS: Low coefficient of variation (CV < 1%)")
    print("✅ All cells have similar library sizes after normalization")
else:
    print(f"⚠️  WARNING: CV = {lib_sizes.std() / lib_sizes.mean():.4f}")
    print("⚠️  Library sizes vary across cells")
print(f"{'='*60}\n")
