#!/bin/bash
# Run on GPU 2

export CUDA_VISIBLE_DEVICES=0

# Create output directory if not exists
mkdir -p ./checkpoints_hPancreas

# Generate timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="./checkpoints_hPancreas/training_${TIMESTAMP}.log"

echo "Training started at $(date)" | tee "${LOG_FILE}"
echo "Log file: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "==========================================" | tee -a "${LOG_FILE}"

python finetune_celltype.py \
    --data /home/angli/baseline/DeepSC/data/processed/baseline/scfoundation/hPancreas_merged.h5ad \
    --finetune_all_encoder \
    --no_freeze_embeddings \
    --label_key celltype \
    --gene_list /home/angli/scFoundation/model/OS_scRNA_gene_index.19264.tsv \
    --split_key split \
    --checkpoint ./models/models.ckpt \
    --epochs 10 \
    --batch_size 2 \
    --lr 1e-4 \
    --output_dir ./checkpoints_hPancreas \
    --gradient_accumulation_steps 10 2>&1 | tee -a "${LOG_FILE}"

echo "==========================================" | tee -a "${LOG_FILE}"
echo "Training completed at $(date)" | tee -a "${LOG_FILE}"

# Find the most recent checkpoint directory and copy log there
LATEST_DIR=$(ls -td ./checkpoints_hPancreas/celltype_finetune_* 2>/dev/null | head -1)
if [ -n "${LATEST_DIR}" ]; then
    cp "${LOG_FILE}" "${LATEST_DIR}/training.log"
    echo "Log file copied to: ${LATEST_DIR}/training.log" | tee -a "${LOG_FILE}"
fi
