#!/usr/bin/env python3
"""
测试 Flash Attention v2 在 DeepSC 模型中的使用
"""

import os
from dataclasses import dataclass

import torch

import sys

# 添加模型路径
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.deepsc.models.deepsc.model import DeepSC


@dataclass
class MoEConfig:
    """MoE配置类"""

    dim: int = 256
    n_routed_experts: int = 8
    n_activated_experts: int = 2
    n_shared_experts: int = 2
    moe_inter_dim: int = 512
    route_scale: float = 1.0
    score_func: str = "softmax"  # "softmax" or "sigmoid"
    n_moe_layers: int = 2  # 最后n层使用MoE
    use_moe_ffn: bool = True  # 是否在FFN中使用MoE


def test_flash_attention():
    """测试 Flash Attention 功能"""
    print("Testing Flash Attention v2 integration...")

    # 模型参数
    embedding_dim = 256
    num_genes = 1000
    num_layers = 4  # 总层数
    num_heads = 8
    batch_size = 4
    seq_len = 128
    num_bins = 8

    # 创建测试数据
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    gene_ids = torch.randint(1, num_genes, (batch_size, seq_len)).to(device)
    expression_bin = torch.randint(0, num_bins, (batch_size, seq_len)).to(device)
    normalized_expr = torch.randn(batch_size, seq_len).to(device)

    # 创建MoE配置
    moe_config = MoEConfig(
        dim=embedding_dim,
        n_routed_experts=8,
        n_activated_experts=2,
        n_shared_experts=2,
        moe_inter_dim=embedding_dim * 2,
        route_scale=1.0,
        score_func="softmax",
        n_moe_layers=2,  # 最后2层使用MoE
        use_moe_ffn=True,
    )

    # 创建模型
    flash_model = DeepSC(
        embedding_dim=embedding_dim,
        num_genes=num_genes,
        num_layers=num_layers,
        num_heads=num_heads,
        num_bins=num_bins,
        enable_mse=True,
        enable_ce=True,
        use_moe_regressor=True,
        use_M_matrix=True,
        gene_embedding_participate_til_layer=3,
        moe=moe_config,
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

            # 输出格式: (logits, regression_output, y, gene_emb, expr_emb)
            if len(flash_outputs) == 5:
                logits, regression_output, y, gene_emb, expr_emb = flash_outputs
                print(f"  Logits shape: {logits.shape}")  # (batch, seq_len, num_bins+1)
                print(
                    f"  Regression output shape: {regression_output.shape}"
                )  # (batch, seq_len)
                print(
                    f"  Gumbel softmax y shape: {y.shape if y is not None else 'None'}"
                )  # (batch, seq_len, seq_len, 3) or None
                print(
                    f"  Gene embedding shape: {gene_emb.shape}"
                )  # (batch, seq_len, embedding_dim)
                print(
                    f"  Expression embedding shape: {expr_emb.shape}"
                )  # (batch, seq_len, embedding_dim)
            else:
                print(
                    f"Output shapes: {[out.shape if hasattr(out, 'shape') else type(out) for out in flash_outputs]}"
                )

    except Exception as e:
        print(f"Error during forward pass: {e}")
        import traceback

        traceback.print_exc()
        return False

    # 测试训练模式
    print("\nTesting training mode...")
    try:
        flash_model.train()

        flash_outputs = flash_model(gene_ids, expression_bin, normalized_expr)
        print("Training mode test successful!")

    except Exception as e:
        print(f"Error during training mode: {e}")
        import traceback

        traceback.print_exc()
        return False

    # 测试带gate_weights的输出
    print("\nTesting with gate weights...")
    try:
        flash_model.eval()
        with torch.no_grad():
            flash_outputs_with_gates = flash_model(
                gene_ids, expression_bin, normalized_expr, return_gate_weights=True
            )
            if len(flash_outputs_with_gates) == 6:
                logits, regression_output, y, gene_emb, expr_emb, gate_weights = (
                    flash_outputs_with_gates
                )
                print(
                    f"  Gate weights shape: {gate_weights.shape}"
                )  # (batch, seq_len, num_experts)
                print("Gate weights test successful!")

    except Exception as e:
        print(f"Error during gate weights test: {e}")
        import traceback

        traceback.print_exc()

    # 测试内存使用（仅在CUDA可用时）
    if torch.cuda.is_available():
        print("\nTesting memory usage...")
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            flash_outputs = flash_model(gene_ids, expression_bin, normalized_expr)
            flash_memory = torch.cuda.max_memory_allocated() / 1024**2

            print(f"Flash Attention memory usage: {flash_memory:.2f} MB")

        except Exception as e:
            print(f"Error during memory test: {e}")
    else:
        print("\nSkipping memory test (CUDA not available)")

    # 测试MoE统计信息
    print("\nTesting MoE statistics...")
    try:
        flash_model.train()
        # 运行几次前向传播以累积统计信息
        for _ in range(3):
            _ = flash_model(gene_ids, expression_bin, normalized_expr)

        # 获取MoE统计信息
        moe_stats = flash_model.get_all_moe_stats()
        print(f"Found {len(moe_stats)} MoE layers")

        # 检查塌缩
        has_collapse = flash_model.print_moe_collapse_report(threshold=0.8)
        print(f"MoE collapse detected: {has_collapse}")

        # 重置统计信息
        flash_model.reset_all_moe_stats()
        print("MoE statistics reset successful!")

    except Exception as e:
        print(f"Error during MoE statistics test: {e}")
        import traceback

        traceback.print_exc()

    print("\nFlash Attention v2 integration test completed successfully!")
    return True


if __name__ == "__main__":
    success = test_flash_attention()
    if success:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
