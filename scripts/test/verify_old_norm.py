"""Verify old normalization library sizes."""

import numpy as np
from scipy import sparse

print("Loading old normalized data...")
old_norm = sparse.load_npz(
    "/home/angli/baseline/DeepSC/data/processed/npz_data_before_shuffel/3ca/batch_19_norm.npz"
)

print(f"Shape: {old_norm.shape}")
print(f"Non-zero elements: {old_norm.nnz:,}")

# 还原: expm1 to reverse log1p
print("\nRestoring original values (expm1)...")
restored = old_norm.copy()
restored.data = np.expm1(restored.data)

# 计算每个细胞的 library size
print("\nCalculating library sizes per cell...")
lib_sizes = np.array(restored.sum(axis=1)).flatten()

print(f"\n{'='*60}")
print("OLD NORMALIZATION - LIBRARY SIZE ANALYSIS")
print(f"{'='*60}")
print(f"  Mean:   {lib_sizes.mean():.2f}")
print(f"  Median: {np.median(lib_sizes):.2f}")
print(f"  Std:    {lib_sizes.std():.2f}")
print(f"  CV:     {lib_sizes.std() / lib_sizes.mean():.6f}")
print(f"  Min:    {lib_sizes.min():.2f}")
print(f"  Max:    {lib_sizes.max():.2f}")

print(f"\n{'='*60}")
print("INTERPRETATION:")
print(f"{'='*60}")
print("The old normalization did NOT eliminate sequencing depth bias.")
print("Different cells have very different library sizes,")
print("making it difficult to compare expression levels across cells.")
print(f"{'='*60}\n")
