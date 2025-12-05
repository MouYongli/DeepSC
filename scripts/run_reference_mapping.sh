#!/bin/bash
# Reference Mapping Script for DeepSC
# Usage: bash scripts/run_reference_mapping.sh

source ~/anaconda3/etc/profile.d/conda.sh
conda activate deepsc

python examples/reference_mapping.py \
    --checkpoint /home/angli/DeepSC/results/pretraining_1120/DeepSC_11_0.ckpt \
    --ref_h5ad /home/angli/scGPT/data/annotation_pancreas/demo_train.h5ad \
    --query_h5ad /home/angli/scGPT/data/annotation_pancreas/demo_test.h5ad \
    --gene_map scripts/data/preprocessing/gene_map.csv \
    --cell_type_key Celltype \
    --visualize \
    --output_dir /home/angli/DeepSC/results/reference_mapping_pancreas
