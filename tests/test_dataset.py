"""Tests for dataset functions."""

import numpy as np
import pytest
from scipy import sparse

from deepsc.data.dataset import normalize_tensor


class TestNormalizeTensor:
    """Test the normalize_tensor function."""
    
    def test_normalize_tensor_valid_input(self):
        """Test normalize_tensor with valid sparse matrix."""
        # Create a simple sparse matrix with cells having >= 200 genes
        row = np.array([0, 0, 0, 1, 1, 1])
        col = np.array([0, 1, 2, 0, 1, 2])
        data = np.array([100, 200, 300, 150, 250, 350])
        
        # Create matrix with 2 cells, each having 3 genes (need at least 200 for test)
        # We'll create a larger matrix to meet the >= 200 genes requirement
        n_genes = 250
        row_expanded = []
        col_expanded = []
        data_expanded = []
        
        for cell in range(2):
            for gene in range(n_genes):
                row_expanded.append(cell)
                col_expanded.append(gene)
                data_expanded.append(np.random.randint(1, 100))
        
        csr = sparse.csr_matrix(
            (data_expanded, (row_expanded, col_expanded)), 
            shape=(2, n_genes)
        )
        
        result = normalize_tensor(csr)
        
        # Check that result is still a CSR matrix
        assert sparse.isspmatrix_csr(result)
        
        # Check that data is log-transformed (all values should be log2(1 + original))
        assert np.all(result.data >= 0)  # log2(1 + x) should be >= 0 for x >= 0
    
    def test_normalize_tensor_invalid_input_type(self):
        """Test that non-sparse input raises TypeError."""
        dense_matrix = np.array([[1, 2, 3], [4, 5, 6]])
        
        with pytest.raises(TypeError, match="Input must be a sparse matrix"):
            normalize_tensor(dense_matrix)
    
    def test_normalize_tensor_non_csr_matrix(self):
        """Test that non-CSR sparse matrix raises TypeError."""
        # Create a COO matrix instead of CSR
        row = np.array([0, 1, 2])
        col = np.array([0, 1, 2]) 
        data = np.array([1, 2, 3])
        coo = sparse.coo_matrix((data, (row, col)), shape=(3, 3))
        
        with pytest.raises(TypeError, match="Input must be a CSR matrix"):
            normalize_tensor(coo)
    
    def test_normalize_tensor_empty_matrix(self):
        """Test that empty matrix raises ValueError."""
        empty_matrix = sparse.csr_matrix((0, 0))
        
        with pytest.raises(ValueError, match="Input matrix cannot have zero dimensions"):
            normalize_tensor(empty_matrix)
    
    def test_normalize_tensor_no_valid_cells(self):
        """Test that matrix with no cells having >= 200 genes raises ValueError."""
        # Create a small matrix where no cell has >= 200 genes
        row = np.array([0, 0, 1, 1])
        col = np.array([0, 1, 0, 1])
        data = np.array([10, 20, 30, 40])
        csr = sparse.csr_matrix((data, (row, col)), shape=(2, 2))
        
        with pytest.raises(ValueError, match="No cells with >= 200 genes found"):
            normalize_tensor(csr)
    
    def test_normalize_tensor_negative_values(self):
        """Test that negative values raise ValueError."""
        # Create matrix with negative values
        n_genes = 250
        row = list(range(n_genes))
        col = list(range(n_genes))
        data = [100] * (n_genes - 1) + [-50]  # One negative value
        
        csr = sparse.csr_matrix((data, (row, col)), shape=(1, n_genes))
        
        with pytest.raises(ValueError, match="Negative values found in expression data"):
            normalize_tensor(csr)


if __name__ == "__main__":
    pytest.main([__file__])