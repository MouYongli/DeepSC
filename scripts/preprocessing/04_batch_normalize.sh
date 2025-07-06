#!/bin/bash

cd "$(dirname "$0")/../.."

set -a
source .env
set +a

python -m src.deepsc.data.preprocessing.batch_normalize \
  --input_dir /home/angli/baseline/DeepSC/data/cellxgene/new_csr/heart \
  --output_dir /home/angli/baseline/DeepSC/data/cellxgene/new_csr_normalized/heart \
