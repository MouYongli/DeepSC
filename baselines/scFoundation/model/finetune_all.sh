#!/bin/bash
# Combined training script for all 4 datasets
# Each experiment runs sequentially using GPU 2 and GPU 3

export CUDA_VISIBLE_DEVICES=2,3

echo "=========================================="
echo "Starting combined training at $(date)"
echo "Using GPUs: 2, 3"
echo "=========================================="

# Base data directory
DATA_DIR="/home/angli/baseline/DeepSC/data/processed/baseline/scfoundation_resplit"

# ====================================
# 1. Train hPancreas
# ====================================
echo ""
echo "=========================================="
echo "Training 1/4: hPancreas"
echo "=========================================="

mkdir -p ./checkpoints_hPancreas
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="./checkpoints_hPancreas/training_${TIMESTAMP}.log"

echo "Training started at $(date)" | tee "${LOG_FILE}"
echo "Log file: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "==========================================" | tee -a "${LOG_FILE}"

python finetune_annotation.py \
    --data "${DATA_DIR}/hPancreas_merged.h5ad" \
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

LATEST_DIR=$(ls -td ./checkpoints_hPancreas/celltype_finetune_* 2>/dev/null | head -1)
if [ -n "${LATEST_DIR}" ]; then
    cp "${LOG_FILE}" "${LATEST_DIR}/training.log"
    echo "Log file copied to: ${LATEST_DIR}/training.log" | tee -a "${LOG_FILE}"
fi

# ====================================
# 2. Train myeloid
# ====================================
echo ""
echo "=========================================="
echo "Training 2/4: myeloid"
echo "=========================================="

mkdir -p ./checkpoints_myeloid
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="./checkpoints_myeloid/training_${TIMESTAMP}.log"

echo "Training started at $(date)" | tee "${LOG_FILE}"
echo "Log file: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "==========================================" | tee -a "${LOG_FILE}"

python finetune_annotation.py \
    --data "${DATA_DIR}/myeloid_merged.h5ad" \
    --finetune_all_encoder \
    --no_freeze_embeddings \
    --label_key celltype \
    --split_key split \
    --checkpoint ./models/models.ckpt \
    --epochs 10 \
    --batch_size 20 \
    --lr 1e-4 \
    --output_dir ./checkpoints_myeloid \
    --gradient_accumulation_steps 20 2>&1 | tee -a "${LOG_FILE}"

echo "==========================================" | tee -a "${LOG_FILE}"
echo "Training completed at $(date)" | tee -a "${LOG_FILE}"

LATEST_DIR=$(ls -td ./checkpoints_myeloid/celltype_finetune_* 2>/dev/null | head -1)
if [ -n "${LATEST_DIR}" ]; then
    cp "${LOG_FILE}" "${LATEST_DIR}/training.log"
    echo "Log file copied to: ${LATEST_DIR}/training.log" | tee -a "${LOG_FILE}"
fi

# ====================================
# 3. Train segerstolpe
# ====================================
echo ""
echo "=========================================="
echo "Training 3/4: segerstolpe"
echo "=========================================="

mkdir -p ./checkpoints_segerstolpe
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="./checkpoints_segerstolpe/training_${TIMESTAMP}.log"

echo "Training started at $(date)" | tee "${LOG_FILE}"
echo "Log file: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "==========================================" | tee -a "${LOG_FILE}"

python finetune_annotation.py \
    --data "${DATA_DIR}/segerstolpe_merged.h5ad" \
    --finetune_all_encoder \
    --no_freeze_embeddings \
    --label_key celltype \
    --split_key split \
    --checkpoint ./models/models.ckpt \
    --epochs 10 \
    --batch_size 2 \
    --lr 1e-4 \
    --output_dir ./checkpoints_segerstolpe \
    --gradient_accumulation_steps 10 2>&1 | tee -a "${LOG_FILE}"

echo "==========================================" | tee -a "${LOG_FILE}"
echo "Training completed at $(date)" | tee -a "${LOG_FILE}"

LATEST_DIR=$(ls -td ./checkpoints_segerstolpe/celltype_finetune_* 2>/dev/null | head -1)
if [ -n "${LATEST_DIR}" ]; then
    cp "${LOG_FILE}" "${LATEST_DIR}/training.log"
    echo "Log file copied to: ${LATEST_DIR}/training.log" | tee -a "${LOG_FILE}"
fi

# ====================================
# 4. Train zheng68k
# ====================================
echo ""
echo "=========================================="
echo "Training 4/4: zheng68k"
echo "=========================================="

mkdir -p ./checkpoints_zheng68k
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="./checkpoints_zheng68k/training_${TIMESTAMP}.log"

echo "Training started at $(date)" | tee "${LOG_FILE}"
echo "Log file: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "==========================================" | tee -a "${LOG_FILE}"

python finetune_annotation.py \
    --data "${DATA_DIR}/zheng_merged.h5ad" \
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

LATEST_DIR=$(ls -td ./checkpoints_zheng68k/celltype_finetune_* 2>/dev/null | head -1)
if [ -n "${LATEST_DIR}" ]; then
    cp "${LOG_FILE}" "${LATEST_DIR}/training.log"
    echo "Log file copied to: ${LATEST_DIR}/training.log" | tee -a "${LOG_FILE}"
fi

# ====================================
# Summary
# ====================================
echo ""
echo "=========================================="
echo "All training completed at $(date)"
echo "=========================================="
echo ""
echo "Results saved in:"
echo "  - ./checkpoints_hPancreas"
echo "  - ./checkpoints_myeloid"
echo "  - ./checkpoints_segerstolpe"
echo "  - ./checkpoints_zheng68k"
