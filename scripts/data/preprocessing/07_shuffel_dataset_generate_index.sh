#!/bin/bash


set -a
source .env
set +a

python -m deepsc.data.preprocessing.shuffel_dataset_generate_index \
    --original_dir $DATASET_BEFORE_SHUFFEL \
    --shuffel_plan_path $SHUFFEL_PLAN_PATH
