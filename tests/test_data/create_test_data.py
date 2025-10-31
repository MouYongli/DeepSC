"""Create synthetic test data for normalization validation."""

import numpy as np
from scipy import sparse

# 设置随机种子以保证可重复性
np.random.seed(42)

# 创建一个模拟的单细胞数据集
# 1000 个细胞，5000 个基因
n_cells = 1000
n_genes = 5000

# 模拟不同测序深度的细胞
# 有些细胞测序深度高（library size ~10000），有些低（~2000）
library_sizes = np.random.choice([2000, 3000, 5000, 8000, 10000], size=n_cells)

# 创建稀疏矩阵
# 每个细胞平均表达 300 个基因（6% 的基因）
rows = []
cols = []
data = []

for cell_idx in range(n_cells):
    # 每个细胞随机选择 250-400 个基因表达
    n_expressed_genes = np.random.randint(250, 400)
    expressed_genes = np.random.choice(n_genes, size=n_expressed_genes, replace=False)

    # 生成符合负二项分布的 counts（符合单细胞数据特征）
    counts = np.random.negative_binomial(5, 0.3, size=n_expressed_genes)

    # 缩放到目标 library size
    current_sum = counts.sum()
    if current_sum > 0:
        counts = counts * library_sizes[cell_idx] / current_sum
        counts = counts.astype(np.float32)

    # 记录非零元素
    for gene_idx, count in zip(expressed_genes, counts):
        rows.append(cell_idx)
        cols.append(gene_idx)
        data.append(count)

# 创建 CSR 稀疏矩阵
csr_matrix = sparse.csr_matrix(
    (data, (rows, cols)), shape=(n_cells, n_genes), dtype=np.float32
)

print("Created test data:")
print(f"  Shape: {csr_matrix.shape}")
print(f"  Non-zero elements: {csr_matrix.nnz:,}")
print(f"  Sparsity: {100 - csr_matrix.nnz / (n_cells * n_genes) * 100:.2f}%")

# 计算一些统计信息
library_sizes_actual = np.array(csr_matrix.sum(axis=1)).flatten()
print(f"\nLibrary sizes before normalization:")
print(f"  Mean: {library_sizes_actual.mean():.0f}")
print(f"  Std:  {library_sizes_actual.std():.0f}")
print(f"  Min:  {library_sizes_actual.min():.0f}")
print(f"  Max:  {library_sizes_actual.max():.0f}")
print(f"  CV:   {library_sizes_actual.std() / library_sizes_actual.mean():.4f}")

# 保存测试数据
import os

os.makedirs("tests/test_data", exist_ok=True)
sparse.save_npz("tests/test_data/synthetic_test_data.npz", csr_matrix)
print(f"\n✅ Saved test data to: tests/test_data/synthetic_test_data.npz")
