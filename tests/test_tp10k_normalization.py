"""Test TP10K normalization implementation."""

import numpy as np
import pytest
from scipy import sparse

from deepsc.data.preprocessing.batch_normalize import normalize_with_tp10k


class TestTP10KNormalization:
    """Test suite for TP10K normalization."""

    def test_basic_normalization(self):
        """Test basic TP10K normalization math."""
        # 创建一个简单的测试矩阵：3个细胞 × 5个基因
        # 细胞1: [100, 200, 0, 300, 400] -> library size = 1000
        # 细胞2: [50, 50, 100, 0, 0] -> library size = 200
        # 细胞3: [200, 200, 200, 200, 200] -> library size = 1000
        data = np.array(
            [
                [100, 200, 0, 300, 400],
                [50, 50, 100, 0, 0],
                [200, 200, 200, 200, 200],
            ],
            dtype=np.float32,
        )
        csr = sparse.csr_matrix(data)

        # 应用 TP10K 归一化（不做 log1p）
        normalized = normalize_with_tp10k(
            csr, scale_factor=1e4, min_genes=2, apply_log1p=False
        )

        # 转为稠密矩阵方便检查
        result = normalized.toarray()

        # 验证：每个细胞的总和应该是 10000
        row_sums = result.sum(axis=1)
        expected_sums = np.array([10000, 10000, 10000])
        np.testing.assert_allclose(row_sums, expected_sums, rtol=1e-5)

        # 验证具体的归一化值
        # 细胞1的第一个基因: 100 / 1000 * 10000 = 1000
        assert abs(result[0, 0] - 1000) < 0.01
        # 细胞2的第一个基因: 50 / 200 * 10000 = 2500
        assert abs(result[1, 0] - 2500) < 0.01
        # 细胞3的第一个基因: 200 / 1000 * 10000 = 2000
        assert abs(result[2, 0] - 2000) < 0.01

    def test_log1p_transformation(self):
        """Test log1p transformation after normalization."""
        data = np.array([[100, 200, 300]], dtype=np.float32)
        csr = sparse.csr_matrix(data)

        # 应用 TP10K + log1p
        normalized = normalize_with_tp10k(
            csr, scale_factor=1e4, min_genes=1, apply_log1p=True
        )

        result = normalized.toarray()

        # 手动计算期望值
        # 1. 归一化: [100/600*10000, 200/600*10000, 300/600*10000]
        #           = [1666.67, 3333.33, 5000]
        # 2. log1p: [log(1+1666.67), log(1+3333.33), log(1+5000)]
        expected = np.log1p(
            np.array([100 / 600 * 1e4, 200 / 600 * 1e4, 300 / 600 * 1e4])
        )

        np.testing.assert_allclose(result[0], expected, rtol=1e-5)

    def test_sparse_preservation(self):
        """Test that sparsity is preserved during normalization."""
        # 创建一个稀疏矩阵：1000个细胞，10000个基因，只有1%非零
        n_cells, n_genes = 1000, 10000
        density = 0.01

        # 使用 random sparse matrix
        from scipy.sparse import random

        csr = random(n_cells, n_genes, density=density, format="csr")
        csr.data = np.abs(csr.data) * 100  # 确保正值

        original_nnz = csr.nnz

        # 应用归一化
        normalized = normalize_with_tp10k(csr, min_genes=50, apply_log1p=True)

        # 稀疏性应该保持
        assert normalized.nnz == original_nnz
        assert normalized.format == "csr"

    def test_filtering_cells(self):
        """Test that cells with too few genes are filtered out."""
        # 创建矩阵：
        # 细胞1: 只有50个基因表达
        # 细胞2: 有300个基因表达
        # 细胞3: 有250个基因表达
        n_genes = 1000
        data1 = sparse.random(1, n_genes, density=0.05, format="csr")  # 50 genes
        data2 = sparse.random(1, n_genes, density=0.30, format="csr")  # 300 genes
        data3 = sparse.random(1, n_genes, density=0.25, format="csr")  # 250 genes

        csr = sparse.vstack([data1, data2, data3])
        csr.data = np.abs(csr.data) * 100

        # min_genes=200，应该过滤掉细胞1
        normalized = normalize_with_tp10k(csr, min_genes=200, apply_log1p=False)

        # 应该只剩2个细胞
        assert normalized.shape[0] == 2

    def test_zero_library_size_handling(self):
        """Test handling of cells with zero library size."""
        # 创建包含全零行的矩阵
        data = np.array(
            [
                [100, 200, 300],
                [0, 0, 0],  # 全零细胞
                [50, 50, 50],
            ],
            dtype=np.float32,
        )
        csr = sparse.csr_matrix(data)

        # 应该不会抛出除零错误
        normalized = normalize_with_tp10k(csr, min_genes=1, apply_log1p=False)

        # 验证没有 NaN 或 Inf
        assert not np.isnan(normalized.data).any()
        assert not np.isinf(normalized.data).any()

    def test_scale_factor_variants(self):
        """Test different scale factors (TP10K vs CPM)."""
        data = np.array([[100, 200, 300, 400]], dtype=np.float32)
        csr = sparse.csr_matrix(data)

        # TP10K (scale_factor = 1e4)
        tp10k = normalize_with_tp10k(
            csr, scale_factor=1e4, min_genes=1, apply_log1p=False
        )

        # CPM (scale_factor = 1e6)
        cpm = normalize_with_tp10k(
            csr, scale_factor=1e6, min_genes=1, apply_log1p=False
        )

        # CPM 的值应该是 TP10K 的 100 倍
        np.testing.assert_allclose(cpm.toarray(), tp10k.toarray() * 100, rtol=1e-5)


def test_diagonal_matrix_optimization():
    """Test that diagonal matrix method gives same result as naive approach."""
    from scipy.sparse import diags

    # 创建测试数据
    data = np.array(
        [
            [100, 200, 300],
            [50, 150, 200],
            [200, 400, 400],
        ],
        dtype=np.float32,
    )
    csr = sparse.csr_matrix(data)

    # 方法1：对角矩阵优化（我们的实现）
    library_sizes = np.array(csr.sum(axis=1)).flatten()
    scale_factors = 1e4 / library_sizes
    D = diags(scale_factors, format="csr")
    result_optimized = (D @ csr).toarray()

    # 方法2：朴素循环实现
    result_naive = np.zeros_like(data)
    for i in range(csr.shape[0]):
        lib_size = csr[i].sum()
        result_naive[i] = csr[i].toarray() / lib_size * 1e4

    # 两种方法结果应该一致
    np.testing.assert_allclose(result_optimized, result_naive, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
