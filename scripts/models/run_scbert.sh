#!/usr/bin/zsh

# 启动参数配置
NUM_GPUS=4 #是否必须喝num_device相同
MASTER_PORT=12625  # 通信端口，这个必须要有，要不然会在默认端口上交换数据，容易导致冲突


PYTHONPATH=src torchrun \
  --nproc_per_node=$NUM_GPUS \
  --master_port=$MASTER_PORT \
  -m deepsc.pretrain.pretrain \
  --data_path "/home/angli/baseline/DeepSC/data/3ac/mapped_batch_data/1d84333c-0327-4ad6-be02-94fee81154ff_sparse_preprocessed.pth" \
  --num_device 4 \
  --batch_size 2 \
  --epoch 10 \
  --model_type "scbert" \
  --gene_num 60664 \
  --bin_num 5 \
  --seed 42 \