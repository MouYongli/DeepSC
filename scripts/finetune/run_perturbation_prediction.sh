#!/usr/bin/bash

# Perturbation Prediction Fine-tuning Script
# Uses Hydra configuration from /home/angli/DeepSC/configs/pp/

echo "=========================================="
echo "DeepSC Perturbation Prediction Fine-tuning"
echo "=========================================="
echo "Start time: $(date)"
echo ""

# Set CUDA device (modify as needed)
export CUDA_VISIBLE_DEVICES=3
export OMP_NUM_THREADS=64

# Load environment variables if .env exists
if [ -f .env ]; then
    set -a
    source .env
    set +a
    echo "Environment variables loaded from .env"
fi

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "PYTHONPATH: $PYTHONPATH"
echo ""

# Run the perturbation prediction script with Hydra
# You can override any config parameters from command line, e.g.:
# ./run_perturbation_prediction.sh learning_rate=1e-4 batch_size=64

echo "Running perturbation prediction fine-tuning..."
python src/deepsc/finetune/perturbation_prediction.py "$@"

echo ""
echo "Finished at: $(date)"
echo "=========================================="
