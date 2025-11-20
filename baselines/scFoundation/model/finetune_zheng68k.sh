#!/bin/bash
# Fine-tune scFoundation on Zheng68K dataset for cell type annotation

# Create output directory if not exists
mkdir -p ./checkpoints_zheng68k

# Generate timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="./checkpoints_zheng68k/training_${TIMESTAMP}.log"

echo "Training started at $(date)" | tee "${LOG_FILE}"
echo "Log file: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "==========================================" | tee -a "${LOG_FILE}"

python finetune_annotation.py \
    --data /home/angli/baseline/DeepSC/data/cell_type_annotation/Zheng68K/zheng_merged.h5ad \
    --finetune_all_encoder \
    --no_freeze_embeddings \
    --label_key cell_type_label \
    --split_key split \
    --checkpoint ./models/models.ckpt \
    --epochs 10 \
    --batch_size 32 \
    --lr 1e-4 \
    --output_dir ./checkpoints_zheng68k \
    --gradient_accumulation_steps 20 2>&1 | tee -a "${LOG_FILE}"

echo "==========================================" | tee -a "${LOG_FILE}"
echo "Training completed at $(date)" | tee -a "${LOG_FILE}"

# Find the most recent checkpoint directory and copy log there
LATEST_DIR=$(ls -td ./checkpoints_zheng68k/celltype_finetune_* 2>/dev/null | head -1)
if [ -n "${LATEST_DIR}" ]; then
    cp "${LOG_FILE}" "${LATEST_DIR}/training.log"
    echo "Log file copied to: ${LATEST_DIR}/training.log" | tee -a "${LOG_FILE}"
fi
