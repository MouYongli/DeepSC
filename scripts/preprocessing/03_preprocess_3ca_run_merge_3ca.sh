#!/bin/bash

cd "$(dirname "$0")/../.."

set -a
source .env
set +a

python -m src.deepsc.data.preprocessing.preprocess_3ca_merge \
  --input_dir "$MAPPED_DATA_PATH_3CA" \
  --output_dir "$MERGED_DATA_PATH_3CA" \
