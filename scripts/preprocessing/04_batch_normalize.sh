#!/bin/bash

cd "$(dirname "$0")/../.."

set -a
source .env
set +a

python -m src.deepsc.data.preprocessing.batch_normalize \
  --input_dir /home/angli/baseline/DeepSC/data/3ca/merged_batch_data_csr \
  --output_dir /home/angli/baseline/DeepSC/data/3ca/merged_batch_data_csr_normalized \
