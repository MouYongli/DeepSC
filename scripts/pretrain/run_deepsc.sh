#!/usr/bin/zsh

# 使用物理编号为 2 和 3 的 GP
export CUDA_VISIBLE_DEVICES=1,2
export OMP_NUM_THREADS=64

NUM_GPUS=2
MASTER_PORT=12620

torchrun \
  --nproc_per_node=$NUM_GPUS \
  --master_port=$MASTER_PORT \
  -m deepsc.pretrain.pretrain \


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
