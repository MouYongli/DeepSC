#!/usr/bin/zsh

# Experiment: Cell Type Annotation on hPancreas Dataset
# Using physical GPU 3
export CUDA_VISIBLE_DEVICES=3
export OMP_NUM_THREADS=64

NUM_GPUS=1
MASTER_PORT=12620

echo "=========================================="
echo "Cell Type Annotation: hPancreas Dataset"
echo "=========================================="
echo "Start time: $(date)"

PYTHONPATH=src torchrun \
  --nproc_per_node=$NUM_GPUS \
  --master_port=$MASTER_PORT \
  -m deepsc.finetune.finetune \
  data_path="/home/angli/DeepSC/data/processed/baseline/scgpt/hPancreas_train.h5ad" \
  data_path_eval="/home/angli/DeepSC/data/processed/baseline/scgpt/hPancreas_test.h5ad"

echo "Experiment finished at: $(date)"
