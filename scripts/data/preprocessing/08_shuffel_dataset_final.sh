#!/bin/bash


set -a
source .env
set +a

python -m deepsc.data.preprocessing.shuffel_dataset_final \
  --csv-file $SHUFFEL_PLAN_PATH \
  --output-dir $DATASET_SHUFFELED \
  --num-workers 1 \
  --in-memory
