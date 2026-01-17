#!/bin/bash

# Bash script to run perturbation prediction fine-tuning
# This script runs the self-contained finetune_perturbation.py

# Set working directory to the script location
cd "$(dirname "$0")" || exit

# Add parent directory to PYTHONPATH so deepsc module can be imported
export PYTHONPATH="${PYTHONPATH}:$(cd ../../.. && pwd)"

echo "PYTHONPATH: $PYTHONPATH"
echo "Working directory: $(pwd)"
echo "Starting perturbation prediction fine-tuning..."
echo "=============================================="

# Run the training script
python finetune_perturbation_v2.py "$@"

echo "=============================================="
echo "Training completed!"
