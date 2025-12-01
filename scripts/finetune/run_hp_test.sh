#!/usr/bin/zsh

# Experiment: Cell Type Annotation on hPancreas Dataset
# Running all 4 experiments: 2 checkpoints × 2 architectures
# Using physical GPU 1
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=64

NUM_GPUS=1

echo "=========================================="
echo "Running All CTA Experiments on hPancreas"
echo "4 experiments total:"
echo "  1. epoch3 + stream1"
echo "  2. epoch3 + stream2"
echo "  3. latest + stream1"
echo "  4. latest + stream2"
echo "=========================================="
echo "Overall start time: $(date)"
echo ""

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
  data_path="/home/angli/DeepSC/data/processed/baseline/scgpt/hPancreas_train.h5ad" \
  data_path_eval="/home/angli/DeepSC/data/processed/baseline/scgpt/hPancreas_test.h5ad" \
  pretrained_model_path="/home/angli/DeepSC/results/pretraining_1201/DeepSC_1_0.ckpt" \
  load_pretrained_model=true \
  model.attention_stream=2 \
  model.cross_attention_architecture="A"

echo "Experiment 1 finished at: $(date)"
echo ""
