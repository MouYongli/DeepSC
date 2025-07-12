"""Tests for utility functions."""

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from deepsc.utils.utils import (
    path_of_file,
    setup_logging,
    seed_all
)


class TestPathOfFile:
    """Test the path_of_file function."""
    
    def test_path_of_file_finds_cell_file(self):
        """Test finding a cell file in directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create a test cell file
            cell_file = temp_path / "cell_metadata.csv"
            cell_file.touch()
            
            result = path_of_file(temp_path, "cell")
            assert result == cell_file
    
    def test_path_of_file_finds_gene_file(self):
        """Test finding a gene file in directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create a test gene file
            gene_file = temp_path / "gene_names.txt"
            gene_file.touch()
            
            result = path_of_file(temp_path, "gene")
            assert result == gene_file
    
    def test_path_of_file_invalid_file_type(self):
        """Test that invalid file type raises ValueError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            with pytest.raises(ValueError, match="Invalid file_name"):
                path_of_file(temp_path, "invalid")
    
    def test_path_of_file_directory_not_exists(self):
        """Test that non-existent directory raises FileNotFoundError."""
        non_existent = Path("/non/existent/directory")
        
        with pytest.raises(FileNotFoundError, match="Directory does not exist"):
            path_of_file(non_existent, "cell")
    
    def test_path_of_file_multiple_files(self):
        """Test that multiple matching files raises ValueError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create multiple cell files
            (temp_path / "cell_1.csv").touch()
            (temp_path / "cell_2.csv").touch()
            
            with pytest.raises(ValueError, match="Multiple cell files found"):
                path_of_file(temp_path, "cell")
    
    def test_path_of_file_no_matching_files(self):
        """Test that no matching files raises FileNotFoundError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create unrelated file
            (temp_path / "other_file.txt").touch()
            
            with pytest.raises(FileNotFoundError, match="No cell file found"):
                path_of_file(temp_path, "cell")


class TestSetupLogging:
    """Test the setup_logging function."""
    
    def test_setup_logging_creates_log_file(self):
        """Test that setup_logging creates a log file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = setup_logging(log_path=temp_dir, log_name="test")
            
            assert os.path.exists(log_path)
            assert "test_" in os.path.basename(log_path)
            assert log_path.endswith(".log")
    
    def test_setup_logging_with_rank(self):
        """Test setup_logging with different ranks."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test rank 0 (should get INFO level)
            log_path = setup_logging(log_path=temp_dir, log_name="test_rank0", rank=0)
            assert os.path.exists(log_path)
            
            # Test rank > 0 (should get WARN level)
            log_path = setup_logging(log_path=temp_dir, log_name="test_rank1", rank=1)
            assert os.path.exists(log_path)


class TestSeedAll:
    """Test the seed_all function."""
    
    @patch('numpy.random.seed')
    @patch('random.seed')
    @patch('torch.manual_seed')
    def test_seed_all_sets_all_seeds(self, mock_torch_seed, mock_random_seed, mock_numpy_seed):
        """Test that seed_all sets all random seeds."""
        seed_value = 42
        seed_all(seed_value)
        
        mock_torch_seed.assert_called_once_with(seed_value)
        mock_random_seed.assert_called_once_with(seed_value)
        mock_numpy_seed.assert_called_once_with(seed_value)
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.manual_seed_all')
    def test_seed_all_sets_cuda_seed_when_available(self, mock_cuda_seed, mock_cuda_available):
        """Test that CUDA seed is set when CUDA is available."""
        seed_value = 42
        seed_all(seed_value)
        
        mock_cuda_seed.assert_called_once_with(seed_value)


if __name__ == "__main__":
    pytest.main([__file__])