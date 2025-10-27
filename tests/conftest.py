"""Shared pytest fixtures and configuration for DeepSC tests."""

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    return {
        "model": {
            "dim": 200,
            "num_tokens": 7,
            "max_seq_len": 60664,
        },
        "dataset": {
            "batch_size": 32,
            "num_workers": 4,
        },
        "training": {
            "learning_rate": 1e-4,
            "epoch": 10,
        },
    }


@pytest.fixture
def sample_data_dir(temp_dir):
    """Create sample data directory structure for testing."""
    data_dir = temp_dir / "data"
    data_dir.mkdir()
    
    # Create sample files
    (data_dir / "sample.h5ad").touch()
    (data_dir / "gene_names.txt").write_text("gene1\ngene2\ngene3\n")
    (data_dir / "cell_metadata.csv").write_text("cell_id,cell_type\ncell1,type1\ncell2,type2\n")
    
    return data_dir