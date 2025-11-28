#!/bin/bash
# Test script for comparing 1-stream and 2-stream attention architectures
# Tests epoch 3 checkpoint and latest checkpoint for both architectures
# Usage: bash scripts/test_attention_stream_checkpoints.sh

source ~/anaconda3/etc/profile.d/conda.sh
conda activate deepsc

# Use GPU rank 3
export CUDA_VISIBLE_DEVICES=3

# Base directories
CHECKPOINT_BASE=/home/angli/DeepSC/results/oneOrTwoStreamAttention
RESULTS_BASE=/home/angli/DeepSC/results/reference_mapping_attention_stream_comparison

# Reference mapping dataset paths (from original script)
REF_H5AD=/home/angli/scGPT/data/annotation_pancreas/demo_train.h5ad
QUERY_H5AD=/home/angli/scGPT/data/annotation_pancreas/demo_test.h5ad
GENE_MAP=/home/angli/baseline/DeepSC-117-t86/scripts/data/preprocessing/gene_map_tp10k.csv
CELL_TYPE_KEY=Celltype

# Create results directory
mkdir -p $RESULTS_BASE

echo "=========================================="
echo "Testing Attention Stream Architectures"
echo "=========================================="
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Reference H5AD: $REF_H5AD"
echo "Query H5AD: $QUERY_H5AD"
echo ""

# ============================================================================
# Test 1-stream architecture
# ============================================================================
echo "=========================================="
echo "Testing 1-Stream Architecture"
echo "=========================================="

STREAM1_DIR=$CHECKPOINT_BASE/pretraining_attention_stream_1

# Test epoch 3 checkpoint
echo ""
echo "Testing 1-stream: Epoch 3 checkpoint (DeepSC_3_0.ckpt)"
echo "------------------------------------------"
python examples/reference_mapping.py \
    --checkpoint $STREAM1_DIR/DeepSC_3_0.ckpt \
    --ref_h5ad $REF_H5AD \
    --query_h5ad $QUERY_H5AD \
    --gene_map $GENE_MAP \
    --cell_type_key $CELL_TYPE_KEY \
    --attention_stream 1 \
    --visualize \
    --output_dir $RESULTS_BASE/1stream_epoch3

echo ""
echo "1-stream epoch 3 test completed!"
echo ""

# Test latest checkpoint
echo "Testing 1-stream: Latest checkpoint"
echo "------------------------------------------"
python examples/reference_mapping.py \
    --checkpoint $STREAM1_DIR/latest_checkpoint.ckpt \
    --ref_h5ad $REF_H5AD \
    --query_h5ad $QUERY_H5AD \
    --gene_map $GENE_MAP \
    --cell_type_key $CELL_TYPE_KEY \
    --attention_stream 1 \
    --visualize \
    --output_dir $RESULTS_BASE/1stream_latest

echo ""
echo "1-stream latest checkpoint test completed!"
echo ""

# ============================================================================
# Test 2-stream architecture
# ============================================================================
echo "=========================================="
echo "Testing 2-Stream Architecture"
echo "=========================================="

STREAM2_DIR=$CHECKPOINT_BASE/pretraining_attention_stream_2

# Test epoch 3 checkpoint
echo ""
echo "Testing 2-stream: Epoch 3 checkpoint (DeepSC_3_0.ckpt)"
echo "------------------------------------------"
python examples/reference_mapping.py \
    --checkpoint $STREAM2_DIR/DeepSC_3_0.ckpt \
    --ref_h5ad $REF_H5AD \
    --query_h5ad $QUERY_H5AD \
    --gene_map $GENE_MAP \
    --cell_type_key $CELL_TYPE_KEY \
    --attention_stream 2 \
    --visualize \
    --output_dir $RESULTS_BASE/2stream_epoch3

echo ""
echo "2-stream epoch 3 test completed!"
echo ""

# Test latest checkpoint
echo "Testing 2-stream: Latest checkpoint"
echo "------------------------------------------"
python examples/reference_mapping.py \
    --checkpoint $STREAM2_DIR/latest_checkpoint.ckpt \
    --ref_h5ad $REF_H5AD \
    --query_h5ad $QUERY_H5AD \
    --gene_map $GENE_MAP \
    --cell_type_key $CELL_TYPE_KEY \
    --attention_stream 2 \
    --visualize \
    --output_dir $RESULTS_BASE/2stream_latest

echo ""
echo "2-stream latest checkpoint test completed!"
echo ""

# ============================================================================
# Summary
# ============================================================================
echo "=========================================="
echo "All Tests Completed!"
echo "=========================================="
echo ""
echo "Results saved to: $RESULTS_BASE"
echo ""
echo "Test directories:"
echo "  - 1-stream epoch 3:  $RESULTS_BASE/1stream_epoch3"
echo "  - 1-stream latest:   $RESULTS_BASE/1stream_latest"
echo "  - 2-stream epoch 3:  $RESULTS_BASE/2stream_epoch3"
echo "  - 2-stream latest:   $RESULTS_BASE/2stream_latest"
echo ""
echo "Checkpoints tested:"
echo "  1-stream:"
echo "    - $STREAM1_DIR/DeepSC_3_0.ckpt"
echo "    - $STREAM1_DIR/latest_checkpoint.ckpt"
echo "  2-stream:"
echo "    - $STREAM2_DIR/DeepSC_3_0.ckpt"
echo "    - $STREAM2_DIR/latest_checkpoint.ckpt"
echo ""
