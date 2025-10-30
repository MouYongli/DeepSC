#!/bin/bash

set -a
source .env
set +a

# ========================================
# TP10K Normalization Configuration
# ========================================
# scale_factor: 10000 for TP10K (default), 1000000 for CPM
# min_genes: Minimum number of genes per cell (default: 200)
# Uncomment --no_log1p to disable log1p transformation
# Uncomment --no_normalize to skip normalization entirely

python -m deepsc.data.preprocessing.cellxgene_data_preprocess \
    --input_dir $DEEPSC_CELLXGENE_PRIMARY_DATA_PATH \
    --output_dir $DATASET_BEFORE_SHUFFEL \
    --gene_map_path $DEEPSC_GENE_MAP_PATH \
    --num_processes 8 \
    --scale_factor 10000 \
    --min_genes 200
