#!/bin/sh

set -a
source .env
set +a


# Use GPU/GPUs on one machine
# Here the example is using GPUs with physical numbers 2 and 3
export CUDA_VISIBLE_DEVICES=3
export OMP_NUM_THREADS=64

NUM_GPUS=1
MASTER_PORT=12620

torchrun \
  --nproc_per_node=$NUM_GPUS \
  --master_port=$MASTER_PORT \
  -m deepsc.pretrain.pretrain \
  model.cross_attention_architecture="B"

# Use the following command to overwrite the parameters
# torchrun \
#   --nproc_per_node=$NUM_GPUS \
#   --master_port=$MASTER_PORT \
#   -m deepsc.pretrain.pretrain
#     model=deepsc \
#     enable_mse_loss=False \
#     enable_huber_loss=True \
#     seed=44 \
#     num_device=4
