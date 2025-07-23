import os
from pathlib import Path

# Base paths - use environment variables with sensible defaults
PROJECT_ROOT = os.getenv("DEEPSC_PROJECT_ROOT", str(Path(__file__).parent.parent.parent.parent))
DATA_ROOT = os.getenv("DEEPSC_DATA_ROOT", os.path.join(PROJECT_ROOT, "data"))
SCRIPTS_ROOT = os.getenv("DEEPSC_SCRIPTS_ROOT", os.path.join(PROJECT_ROOT, "scripts"))

# Configuration paths
HGNC_DATABASE = os.getenv(
    "DEEPSC_HGNC_DATABASE", 
    os.path.join(SCRIPTS_ROOT, "preprocessing", "HGNC_database.txt")
)
BASE_URL = os.getenv("DEEPSC_3CA_BASE_URL", "https://www.weizmann.ac.il/sites/3CA/")
TRIPLECA_DATASET_PATH = os.getenv(
    "DEEPSC_TRIPLECA_PATH", 
    os.path.join(DATA_ROOT, "3ca", "raw")
)
CELLXGENE_DATASET_PATH = os.getenv(
    "DEEPSC_CELLXGENE_PATH", 
    os.path.join(DATA_ROOT, "cellxgene", "blood", "partition_0.h5ad")
)
GENE_MAP_PATH = os.getenv(
    "DEEPSC_GENE_MAP_PATH", 
    os.path.join(SCRIPTS_ROOT, "preprocessing", "gene_map.csv")
)
