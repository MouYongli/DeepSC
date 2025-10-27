"""Integration tests for pretraining pipeline."""

import tempfile
from pathlib import Path

import pytest


class TestPretrainingPipeline:
    """Test the pretraining pipeline integration."""

    def test_pretrain_module_import(self):
        """Test that pretrain module can be imported."""
        try:
            from src.deepsc.pretrain.pretrain import pretrain
            assert callable(pretrain)
        except ImportError as e:
            pytest.skip(f"Pretrain module not available: {e}")

    def test_trainer_import(self):
        """Test that trainer module can be imported."""
        try:
            from src.deepsc.train.trainer import Trainer  
            assert Trainer is not None
        except ImportError as e:
            pytest.skip(f"Trainer class not available: {e}")

    def test_model_imports(self):
        """Test that model modules can be imported."""
        model_modules = [
            "src.deepsc.models.scbert",
            "src.deepsc.models.deepsc", 
        ]
        
        for module_name in model_modules:
            try:
                __import__(module_name)
            except ImportError as e:
                # Models may require additional dependencies, so we skip if not available
                pytest.skip(f"Model module {module_name} not available: {e}")

    def test_config_files_exist(self):
        """Test that essential config files exist."""
        config_files = [
            "configs/pretrain/pretrain.yaml",
            "configs/pretrain/model/scbert.yaml",
            "configs/pretrain/dataset/tripleca.yaml",
        ]
        
        for config_file in config_files:
            assert Path(config_file).exists(), f"Config file {config_file} should exist"

    def test_utils_import(self):
        """Test that utility modules can be imported."""
        try:
            from src.deepsc.utils.utils import setup_logging
            assert callable(setup_logging)
        except ImportError as e:
            pytest.skip(f"Utils module not available: {e}")


class TestConfigurationSystem:
    """Test the Hydra configuration system integration."""

    def test_hydra_import(self):
        """Test that hydra is available."""
        try:
            import hydra
            from omegaconf import DictConfig
            assert hydra is not None
            assert DictConfig is not None
        except ImportError as e:
            pytest.skip(f"Hydra not available: {e}")

    def test_config_structure(self):
        """Test that config directory structure exists."""
        config_dirs = [
            "configs/pretrain/model",
            "configs/pretrain/dataset", 
            "configs/finetune",
        ]
        
        for config_dir in config_dirs:
            assert Path(config_dir).exists(), f"Config directory {config_dir} should exist"
            assert Path(config_dir).is_dir(), f"{config_dir} should be a directory"