#!/usr/bin/zsh

# Experiment: Cell Type Annotation on hPancreas Dataset
# Running all 4 experiments: 2 checkpoints × 2 architectures
# Using physical GPU 1
export CUDA_VISIBLE_DEVICES=3
export OMP_NUM_THREADS=64

NUM_GPUS=1

# Experiment 1: epoch3 + stream1
echo "=========================================="
echo "Experiment 1/4: Cell Type Annotation"
echo "Checkpoint: epoch3 (DeepSC_3_0.ckpt)"
echo "Architecture: One Stream (attention_stream=1)"
echo "=========================================="
echo "Start time: $(date)"

PYTHONPATH=src torchrun \
  --nproc_per_node=$NUM_GPUS \
  --master_port=12622 \
  -m deepsc.finetune.finetune \
  data_path="/home/angli/DeepSC/data/processed/baseline/scgpt/myeloid_train.h5ad" \
  data_path_eval="/home/angli/DeepSC/data/processed/baseline/scgpt/myeloid_test.h5ad" \
  pretrained_model_path="/home/angli/DeepSC/results/pretraining_1201/DeepSC_11_0.ckpt" \
  load_pretrained_model=true \
  model.attention_stream=2 \
  model.cross_attention_architecture="A"

echo "Experiment 1 finished at: $(date)"
echo ""
