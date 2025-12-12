#!/bin/bash
#SBATCH -A p0021245
#SBATCH --job-name=deepsc_attention_stream_exp
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

echo "source /home/rn260358/miniforge3/etc/profile.d/conda.sh"
source /home/rn260358/miniforge3/etc/profile.d/conda.sh
conda activate scbertnew
echo "conda activate scbertnew"

# 使用作业ID来生成唯一端口，避免端口冲突
export MASTER_PORT=$((10000 + $SLURM_JOB_ID % 10000))
#显示出所有主机节点|取第一个
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export WORLD_SIZE=$SLURM_NTASKS
export PYTHONPATH=/hpcwork/p0021245/testdeepsc/DeepSC/src

echo "MASTER_ADDR=$MASTER_ADDR"
echo "WORLD_SIZE=$WORLD_SIZE"
echo "MASTER_PORT=$MASTER_PORT"
echo "SLURM_PROCID=$SLURM_PROCID"
echo "PYTHONPATH=$PYTHONPATH"
echo "NCCL_DEBUG=$NCCL_DEBUG"

# 创建日志目录
mkdir -p logs

# Project root directory
PROJECT_ROOT=/hpcwork/p0021245/testdeepsc/DeepSC

# ============================================================================
# Experiment 1: attention_stream=1 (single stream architecture)
# ============================================================================
echo "=========================================="
echo "Starting Experiment 1: attention_stream=1"
echo "=========================================="

# Set checkpoint directory for experiment 1
export DEEPSC_PRETRAIN_CKPT_DIR=$PROJECT_ROOT/results/pretraining_attention_stream_1

# Create checkpoint directory if it doesn't exist
mkdir -p $DEEPSC_PRETRAIN_CKPT_DIR

# Run training with attention_stream=1 for 4 epochs
srun python -m deepsc.pretrain.pretrain \
    model.attention_stream=1 \
    epoch=4

echo "Experiment 1 completed!"

# ============================================================================
# Experiment 2: attention_stream=2 (dual stream architecture)
# ============================================================================
echo "=========================================="
echo "Starting Experiment 2: attention_stream=2"
echo "=========================================="

# Update master port to avoid conflicts if running consecutively
export MASTER_PORT=$((10000 + ($SLURM_JOB_ID + 1) % 10000))

# Set checkpoint directory for experiment 2
export DEEPSC_PRETRAIN_CKPT_DIR=$PROJECT_ROOT/results/pretraining_attention_stream_2

# Create checkpoint directory if it doesn't exist
mkdir -p $DEEPSC_PRETRAIN_CKPT_DIR

# Run training with attention_stream=2 for 4 epochs
srun python -m deepsc.pretrain.pretrain \
    model.attention_stream=2 \
    epoch=4

echo "Experiment 2 completed!"

echo "=========================================="
echo "All experiments completed successfully!"
echo "=========================================="
