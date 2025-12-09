#!/usr/bin/zsh

# Experiment: Cell Type Annotation on hPancreas Dataset
# Running all 4 experiments: 2 checkpoints × 2 architectures
# Using physical GPU 1
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=64
set -a
source .env
set +a
NUM_GPUS=1

# Experiment 1: epoch3 + stream1
echo "Start time: $(date)"

PYTHONPATH=src torchrun \
  --nproc_per_node=$NUM_GPUS \
  --master_port=12620 \
  -m deepsc.finetune.run_pp \
  include_zero_gene='batch-wise' \
  pretrained_model_path="/home/angli/DeepSC/results/dimension_layers_exp/dim_512_gene_layer_5/DeepSC_5_0.ckpt" \
  model.embedding_dim=512 \
  model.gene_embedding_participate_til_layer=5

echo "Experiment 1 finished at: $(date)"
echo ""
