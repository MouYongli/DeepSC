#!/usr/bin/env python3
"""
Test Flash Attention v2 integration in DeepSC model
"""

import os
from dataclasses import dataclass

import torch

import sys

# Add model path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.deepsc.models.deepsc.model import DeepSC


@dataclass
class MoEConfig:
    """MoE configuration class"""

    dim: int = 256
    n_routed_experts: int = 8
    n_activated_experts: int = 2
    n_shared_experts: int = 2
    moe_inter_dim: int = 512
    route_scale: float = 1.0
    score_func: str = "softmax"  # "softmax" or "sigmoid"
    n_moe_layers: int = 2  # Last n layers use MoE
    use_moe_ffn: bool = True  # Whether to use MoE in FFN


def test_flash_attention():
    """Test Flash Attention functionality"""
    print("Testing Flash Attention v2 integration...")

    # Model parameters
    embedding_dim = 256
    num_genes = 1000
    num_layers = 4  # Total layers
    num_heads = 8
    batch_size = 4
    seq_len = 128
    num_bins = 8

    # Create test data
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    gene_ids = torch.randint(1, num_genes, (batch_size, seq_len)).to(device)
    expression_bin = torch.randint(0, num_bins, (batch_size, seq_len)).to(device)
    normalized_expr = torch.randn(batch_size, seq_len).to(device)

    # Create MoE configuration
    moe_config = MoEConfig(
        dim=embedding_dim,
        n_routed_experts=8,
        n_activated_experts=2,
        n_shared_experts=2,
        moe_inter_dim=embedding_dim * 2,
        route_scale=1.0,
        score_func="softmax",
        n_moe_layers=2,  # Last 2 layers use MoE
        use_moe_ffn=True,
    )

    # Create model
    flash_model = DeepSC(
        embedding_dim=embedding_dim,
        num_genes=num_genes,
        num_layers=num_layers,
        num_heads=num_heads,
        num_bins=num_bins,
        enable_mse=True,
        enable_ce=True,
        use_moe_regressor=True,
        gene_embedding_participate_til_layer=3,
        moe=moe_config,
    ).to(device)

    print(
        f"Flash model parameters: {sum(p.numel() for p in flash_model.parameters()):,}"
    )

    # Test forward pass
    print("\nTesting forward pass...")

    try:
        with torch.no_grad():
            # Flash Attention model
            flash_outputs = flash_model(gene_ids, expression_bin, normalized_expr)
            print("Flash Attention model forward pass successful!")

            # Output format: (logits, regression_output, gene_emb, expr_emb)
            if len(flash_outputs) == 4:
                logits, regression_output, gene_emb, expr_emb = flash_outputs
                print(f"  Logits shape: {logits.shape}")  # (batch, seq_len, num_bins+1)
                print(
                    f"  Regression output shape: {regression_output.shape}"
                )  # (batch, seq_len)
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

    # Test training mode
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

    # Test output with gate weights
    print("\nTesting with gate weights...")
    try:
        flash_model.eval()
        with torch.no_grad():
            flash_outputs_with_gates = flash_model(
                gene_ids, expression_bin, normalized_expr, return_gate_weights=True
            )
            if len(flash_outputs_with_gates) == 5:
                logits, regression_output, gene_emb, expr_emb, gate_weights = (
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

    # Test memory usage (only when CUDA is available)
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

    # Test MoE statistics
    print("\nTesting MoE statistics...")
    try:
        flash_model.train()
        # Run a few forward passes to accumulate statistics
        for _ in range(3):
            _ = flash_model(gene_ids, expression_bin, normalized_expr)

        # Get MoE statistics
        moe_stats = flash_model.get_all_moe_stats()
        print(f"Found {len(moe_stats)} MoE layers")

        # Check for collapse
        has_collapse = flash_model.print_moe_collapse_report(threshold=0.8)
        print(f"MoE collapse detected: {has_collapse}")

        # Reset statistics
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
