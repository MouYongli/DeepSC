#!/bin/bash

set -a
source .env
set +a

python -m deepsc.data.preprocessing.batch_normalize \
  --input_dir $MERGED_DATA_PATH_3CA \
  --output_dir $DATASET_BEFORE_SHUFFEL/3ca \
