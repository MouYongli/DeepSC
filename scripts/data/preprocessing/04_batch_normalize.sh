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
# Uncomment --no_tp10k to use legacy normalization (not recommended)

python -m deepsc.data.preprocessing.batch_normalize \
  --input_dir $MERGED_DATA_PATH_3CA \
  --output_dir $DATASET_BEFORE_SHUFFEL/3ca \
  --scale_factor 10000 \
  --min_genes 200
