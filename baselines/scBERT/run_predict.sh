#!/bin/bash
# Prediction script for scBERT model
# Run with: bash run_predict.sh

# Activate conda environment and run prediction with metrics
/home/angli/anaconda3/envs/scbert/bin/python predict_with_metrics.py \
  --data_path ./data/Zheng68K_test.h5ad \
  --model_path ./ckpts/finetune_best_zheng68k.pth \
  --label_prefix finetune_ \
  --bin_num 5 \
  --gene_num 16906 \
  --seed 2021 \
  --novel_type False \
  --unassign_thres 0.5

echo ""
echo "Prediction completed at $(date)"
