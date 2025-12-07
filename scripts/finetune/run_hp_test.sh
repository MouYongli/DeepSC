#!/usr/bin/zsh

# Experiment: Cell Type Annotation on hPancreas Dataset
# Running all 4 experiments: 2 checkpoints × 2 architectures
# Using physical GPU 1
export CUDA_VISIBLE_DEVICES=3
export OMP_NUM_THREADS=64
set -a
source .env
set +a
NUM_GPUS=1

# Experiment 1: epoch3 + stream1
echo "Start time: $(date)"

PYTHONPATH=src torchrun \
  --nproc_per_node=$NUM_GPUS \
  --master_port=12622 \
  -m deepsc.finetune.run_cta \
  data_path="/home/angli/DeepSC/data/processed/baseline/scgpt/zheng_train_preprocessed.h5ad" \
  data_path_eval="/home/angli/DeepSC/data/processed/baseline/scgpt/zheng_test_preprocessed.h5ad" \
  load_pretrained_model=true \
  model.attention_stream=2 \
  model.cross_attention_architecture="A"

echo "Experiment 1 finished at: $(date)"
echo ""
