#!/usr/bin/env python3
"""
测试脚本：验证DeepSC checkpoint加载和AnnotationModel是否正常工作
"""

import os

import torch
from omegaconf import OmegaConf

import sys

# 添加模型路径到系统路径
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from cell_type_annotation_finetune import AnnotationModel, load_deepsc_from_checkpoint


def test_checkpoint_loading():
    """
    测试checkpoint加载功能
    """
    print("Testing DeepSC checkpoint loading...")

    # 创建一个测试配置，匹配checkpoint中的实际模型配置
    test_config = OmegaConf.create(
        {
            "model": {
                "embedding_dim": 256,
                "num_genes": 34682,  # checkpoint中是34684，减去2个特殊tokens
                "num_layers": 10,  # 从checkpoint错误信息中推断出的实际值
                "num_heads": 8,
                "attn_dropout": 0.1,
                "ffn_dropout": 0.1,
                "fused": True,
                "num_bins": 5,  # 从checkpoint错误信息中推断出的实际值 (8-3=5, 减去特殊tokens)
                "alpha": 0.3,
                "mask_layer_start": 9,  # 调整为最后一层
                "enable_l0": True,
                "enable_mse": True,
                "enable_ce": True,
                "num_layers_ffn": 6,  # 从checkpoint错误信息中推断出的实际值
                "use_moe_regressor": True,
            }
        }
    )

    # 使用results目录中的checkpoint路径
    checkpoint_path = "/home/angli/baseline/DeepSC/results/latest_checkpoint.ckpt"

    # 检查checkpoint文件是否存在
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint file not found at {checkpoint_path}")
        print(
            "Please update the checkpoint_path in the config file to point to your actual checkpoint."
        )

        # 查找可能的checkpoint文件
        ckpt_dir = os.path.dirname(checkpoint_path)
        if os.path.exists(ckpt_dir):
            print(f"\nLooking for checkpoint files in {ckpt_dir}:")
            for file in os.listdir(ckpt_dir):
                if file.endswith((".pth", ".pt", ".ckpt")):
                    print(f"  Found: {file}")
        return False

    try:
        # 尝试加载模型
        encoder = load_deepsc_from_checkpoint(checkpoint_path, test_config)
        print("✓ Successfully loaded DeepSC model from checkpoint")
        print(
            f"✓ Model has {sum(p.numel() for p in encoder.parameters())} total parameters"
        )
        print(
            f"✓ All parameters are frozen: {all(not p.requires_grad for p in encoder.parameters())}"
        )

        # 创建注释模型
        annotation_model = AnnotationModel(
            encoder=encoder,
            embedding_dim=test_config.model.embedding_dim,
            num_classes=10,  # 测试用的类别数
        )
        print("✓ Successfully created AnnotationModel")
        print(
            f"✓ AnnotationModel has {sum(p.numel() for p in annotation_model.parameters() if p.requires_grad)} trainable parameters"
        )

        # 创建虚拟输入来测试forward pass
        batch_size = 2
        seq_len = 1024

        # 虚拟数据（需要根据实际数据格式调整）
        gene_ids = torch.randint(
            1, test_config.model.num_genes, (batch_size, seq_len)
        )  # 避免使用padding token (0)
        expression_bin = torch.randint(
            1, test_config.model.num_bins + 3, (batch_size, seq_len)
        )  # 加上特殊tokens
        normalized_expr = torch.randn(batch_size, seq_len)

        print("\nTesting forward pass with dummy data:")
        print(
            f"  Input shapes: gene_ids={gene_ids.shape}, expression_bin={expression_bin.shape}, normalized_expr={normalized_expr.shape}"
        )

        # 测试forward pass
        annotation_model.eval()
        with torch.no_grad():
            logits = annotation_model(gene_ids, expression_bin, normalized_expr)
            print(f"✓ Forward pass successful, output shape: {logits.shape}")
            print(
                f"✓ Output logits range: [{logits.min().item():.3f}, {logits.max().item():.3f}]"
            )

        print(
            "\n🎉 All tests passed! The checkpoint loading and model setup are working correctly."
        )
        return True

    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_checkpoint_loading()
    if success:
        print("\nYou can now run the fine-tuning script with:")
        print("python cell_type_annotation_finetune.py")
    else:
        print("\nPlease fix the issues above before running the fine-tuning script.")
