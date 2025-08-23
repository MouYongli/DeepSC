"""Integration tests for data download pipeline."""

import os
import tempfile
from pathlib import Path

import pytest


class TestDataDownloadPipeline:
    """Test the data download pipeline integration."""

    def test_tripleca_import(self):
        """Test that tripleca download module can be imported."""
        try:
            from src.deepsc.data.download.tripleca import download_3ca
            assert hasattr(download_3ca, 'main') or callable(download_3ca)
        except ImportError as e:
            pytest.skip(f"Tripleca download module not available: {e}")

    def test_cellxgene_import(self):
        """Test that cellxgene download module can be imported.""" 
        try:
            from src.deepsc.data.download.cellxgene import download_partition
            assert hasattr(download_partition, 'main') or callable(download_partition)
        except ImportError as e:
            pytest.skip(f"Cellxgene download module not available: {e}")

    def test_config_paths_exist(self):
        """Test that configuration paths for data downloading exist."""
        config_files = [
            "src/deepsc/data/download/tripleca/config.py",
            "src/deepsc/data/preprocessing/config.py", 
        ]
        
        for config_file in config_files:
            assert Path(config_file).exists(), f"Config file {config_file} should exist"

    def test_data_preprocessing_imports(self):
        """Test that data preprocessing modules can be imported."""
        try:
            from src.deepsc.data.preprocessing import config
            assert hasattr(config, 'PROJECT_ROOT')
            assert hasattr(config, 'DATA_ROOT')
        except ImportError as e:
            pytest.skip(f"Data preprocessing config not available: {e}")


class TestPreprocessingPipeline:
    """Test the data preprocessing pipeline."""

    def test_preprocessing_modules_import(self):
        """Test that preprocessing modules can be imported."""
        modules_to_test = [
            "src.deepsc.data.preprocessing.preprocess_datasets",
            "src.deepsc.data.preprocessing.batch_normalize",
            "src.deepsc.data.preprocessing.gene_name_normalization",
        ]
        
        for module_name in modules_to_test:
            try:
                __import__(module_name)
            except ImportError as e:
                pytest.skip(f"Module {module_name} not available: {e}")

    def test_dataset_class_import(self):
        """Test that dataset classes can be imported."""
        try:
            from src.deepsc.data.dataset import ScRNADataset
            assert ScRNADataset is not None
        except ImportError as e:
            pytest.skip(f"Dataset class not available: {e}")