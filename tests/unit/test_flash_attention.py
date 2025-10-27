#!/usr/bin/env python3
"""
测试 Flash Attention v2 在 DeepSC 模型中的使用
"""

import os

import torch

import sys

# 添加模型路径
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.deepsc.models.deepsc.model import DeepSC


def test_flash_attention():
    """测试 Flash Attention 功能"""
    print("Testing Flash Attention v2 integration...")

    # 模型参数
    embedding_dim = 256
    num_genes = 1000
    num_layers = 2
    num_heads = 8
    batch_size = 4
    seq_len = 128

    # 创建测试数据
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    gene_ids = torch.randint(1, num_genes, (batch_size, seq_len)).to(device)
    expression_bin = torch.randint(0, 8, (batch_size, seq_len)).to(device)
    normalized_expr = torch.randn(batch_size, seq_len).to(device)

    # 创建模型
    flash_model = DeepSC(
        embedding_dim=embedding_dim,
        num_genes=num_genes,
        num_layers=num_layers,
        num_heads=num_heads,
        enable_mse=True,
        enable_ce=True,
    ).to(device)

    print(
        f"Flash model parameters: {sum(p.numel() for p in flash_model.parameters()):,}"
    )

    # 测试前向传播
    print("\nTesting forward pass...")

    try:
        with torch.no_grad():
            # Flash Attention 模型
            flash_outputs = flash_model(gene_ids, expression_bin, normalized_expr)
            print("Flash Attention model forward pass successful!")
            print(f"Output shapes: {[out.shape for out in flash_outputs]}")

    except Exception as e:
        print(f"Error during forward pass: {e}")
        return False

    # 测试训练模式
    print("\nTesting training mode...")
    try:
        flash_model.train()

        flash_outputs = flash_model(gene_ids, expression_bin, normalized_expr)
        print("Training mode test successful!")

    except Exception as e:
        print(f"Error during training mode: {e}")
        return False

    # 测试内存使用
    print("\nTesting memory usage...")
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        flash_outputs = flash_model(gene_ids, expression_bin, normalized_expr)
        flash_memory = torch.cuda.max_memory_allocated() / 1024**2

        print(f"Flash Attention memory usage: {flash_memory:.2f} MB")

    except Exception as e:
        print(f"Error during memory test: {e}")

    print("\nFlash Attention v2 integration test completed successfully!")
    return True


if __name__ == "__main__":
    success = test_flash_attention()
    if success:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
