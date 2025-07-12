#!/bin/bash

cd "$(dirname "$0")/../.."

set -a
source .env
set +a

python -m src.deepsc.data.preprocessing.preprocess_datasets_3ca \
  --output_dir "$MAPPED_DATA_PATH_3CA" \
