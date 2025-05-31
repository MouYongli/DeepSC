#!/bin/bash

OUTPUT_DIR="/home/angli/DeepSC/mapped_batch_data/3ca"

python -m scripts.preprocessing.preprocess_datasets_3ca \
  --output_dir "$OUTPUT_DIR" \
