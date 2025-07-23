#!/usr/bin/env python3
"""
测试细胞类型注释微调pipeline
"""

import logging
import os

import numpy as np
import torch
from omegaconf import OmegaConf

import sys

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_dummy_h5ad(output_path: str, n_cells: int = 1000, n_genes: int = 2000):
    """
    创建一个虚拟的h5ad文件用于测试
    """
    try:
        import anndata as ad
        import pandas as pd
        from scipy.sparse import csr_matrix
    except ImportError:
        print("请安装scanpy: pip install scanpy")
        return False

    logger.info(f"Creating dummy h5ad file with {n_cells} cells and {n_genes} genes")

    # 创建随机表达数据（稀疏矩阵）
    np.random.seed(42)

    # 创建一些有意义的表达模式
    expression_data = np.random.negative_binomial(
        n=5, p=0.3, size=(n_cells, n_genes)
    ).astype(np.float32)

    # 添加一些噪声和差异表达
    for i in range(0, n_cells, 200):  # 每200个细胞为一个类型
        end_idx = min(i + 200, n_cells)
        # 为每个类型添加特定的表达模式
        type_specific_genes = np.random.choice(n_genes, size=100, replace=False)
        expression_data[i:end_idx, type_specific_genes] *= np.random.uniform(2, 5)

    # 转换为稀疏矩阵
    X = csr_matrix(expression_data)

    # 创建Ensembl ID格式的基因名（使用一些真实的Ensembl ID样例）
    ensembl_ids = [
        "ENSG00000121410",
        "ENSG00000268895",
        "ENSG00000148584",
        "ENSG00000175899",
        "ENSG00000245105",
        "ENSG00000166535",
        "ENSG00000256069",
        "ENSG00000184389",
        "ENSG00000128274",
        "ENSG00000118017",
        "ENSG00000094914",
        "ENSG00000081760",
        "ENSG00000109576",
        "ENSG00000132646",
        "ENSG00000121879",
        "ENSG00000177885",
        "ENSG00000109321",
        "ENSG00000134698",
        "ENSG00000128739",
        "ENSG00000171097",
        "ENSG00000090861",
        "ENSG00000180773",
        "ENSG00000266967",
        "ENSG00000109062",
        "ENSG00000174145",
        "ENSG00000131018",
        "ENSG00000137834",
        "ENSG00000132849",
        "ENSG00000183044",
        "ENSG00000165029",
        "ENSG00000157916",
        "ENSG00000154803",
    ] * (
        n_genes // 32 + 1
    )  # 重复Ensembl ID以达到所需数量
    gene_names = ensembl_ids[:n_genes]

    # 为剩余的基因创建虚拟Ensembl ID
    for i in range(len(gene_names), n_genes):
        gene_names.append(f"ENSG{i:011d}")

    # 创建细胞类型标签
    cell_types = []
    type_names = ["T_cell", "B_cell", "NK_cell", "Monocyte", "Dendritic_cell"]

    for i in range(n_cells):
        type_idx = i // 200  # 每200个细胞一个类型
        if type_idx >= len(type_names):
            type_idx = len(type_names) - 1
        cell_types.append(type_names[type_idx])

    # 创建obs (细胞元数据)
    obs = pd.DataFrame(
        {
            "celltype": cell_types,
            "n_genes": np.array((X > 0).sum(axis=1)).flatten(),
            "total_counts": np.array(X.sum(axis=1)).flatten(),
        }
    )
    obs.index = [f"CELL_{i:06d}" for i in range(n_cells)]

    # 创建var (基因元数据)
    var = pd.DataFrame(
        {
            "ensembl_id": gene_names,  # 使用ensembl_id列存储Ensembl ID
            "gene_name": gene_names,  # 也保留gene_name列
            "n_cells": np.array((X > 0).sum(axis=0)).flatten(),
            "total_counts": np.array(X.sum(axis=0)).flatten(),
        }
    )
    var.index = gene_names

    # 创建AnnData对象
    adata = ad.AnnData(X=X, obs=obs, var=var)

    # 保存
    adata.write_h5ad(output_path)
    logger.info(f"Dummy h5ad file saved to {output_path}")

    # 打印一些统计信息
    logger.info("Cell type distribution:")
    for cell_type, count in pd.Series(cell_types).value_counts().items():
        logger.info(f"  {cell_type}: {count} cells")

    return True


def test_data_loading(h5ad_path: str):
    """测试数据加载功能"""
    logger.info("Testing data loading...")

    try:
        sys.path.append(os.path.join(os.path.dirname(__file__)))
        from finetune_data_loader import create_data_loaders

        # 创建数据加载器
        train_loader, val_loader, test_loader, num_classes = create_data_loaders(
            h5ad_path=h5ad_path,
            gene_map_path="/home/angli/DeepSC/scripts/preprocessing/gene_map.csv",
            batch_size=4,  # 小批次用于测试
            max_length=512,  # 较短长度用于测试
            num_bins=5,
            test_size=0.2,
            val_size=0.1,
            random_state=42,
            num_workers=0,  # 单进程避免多进程问题
        )

        logger.info(f"Successfully created data loaders with {num_classes} classes")
        logger.info(f"Train batches: {len(train_loader)}")
        logger.info(f"Val batches: {len(val_loader)}")
        logger.info(f"Test batches: {len(test_loader)}")

        # 测试一个批次
        logger.info("Testing first batch...")
        for batch in train_loader:
            logger.info("Batch keys and shapes:")
            for key, value in batch.items():
                logger.info(f"  {key}: {value.shape}, dtype: {value.dtype}")

            # 检查数据范围
            logger.info("Data ranges:")
            logger.info(
                f"  gene_ids: [{batch['gene_ids'].min()}, {batch['gene_ids'].max()}]"
            )
            logger.info(
                f"  expression_bin: [{batch['expression_bin'].min()}, {batch['expression_bin'].max()}]"
            )
            logger.info(
                f"  normalized_expr: [{batch['normalized_expr'].min():.3f}, {batch['normalized_expr'].max():.3f}]"
            )
            logger.info(
                f"  cell_types: [{batch['cell_types'].min()}, {batch['cell_types'].max()}]"
            )

            break

        return True, num_classes

    except Exception as e:
        logger.error(f"Data loading test failed: {e}")
        import traceback

        traceback.print_exc()
        return False, 0


def test_model_integration(num_classes: int):
    """测试模型集成"""
    logger.info("Testing model integration...")

    try:
        sys.path.append(os.path.join(os.path.dirname(__file__)))
        from cell_type_annotation_finetune import (
            AnnotationModel,
            load_deepsc_from_checkpoint,
        )

        # 创建测试配置
        test_config = OmegaConf.create(
            {
                "model": {
                    "embedding_dim": 256,
                    "num_genes": 34682,
                    "num_layers": 10,
                    "num_heads": 8,
                    "attn_dropout": 0.1,
                    "ffn_dropout": 0.1,
                    "fused": True,
                    "num_bins": 5,
                    "alpha": 0.3,
                    "mask_layer_start": 9,
                    "enable_l0": True,
                    "enable_mse": True,
                    "enable_ce": True,
                    "num_layers_ffn": 6,
                    "use_moe_regressor": True,
                }
            }
        )

        checkpoint_path = "/home/angli/baseline/DeepSC/results/latest_checkpoint.ckpt"

        # 加载encoder
        encoder = load_deepsc_from_checkpoint(checkpoint_path, test_config)
        logger.info("✓ Encoder loaded successfully")

        # 创建注释模型
        annotation_model = AnnotationModel(
            encoder=encoder,
            embedding_dim=test_config.model.embedding_dim,
            num_classes=num_classes,
        )
        logger.info("✓ Annotation model created successfully")
        logger.info(
            f"✓ Trainable parameters: {sum(p.numel() for p in annotation_model.parameters() if p.requires_grad)}"
        )

        # 测试前向传播
        batch_size = 2
        seq_len = 512

        gene_ids = torch.randint(1, test_config.model.num_genes, (batch_size, seq_len))
        expression_bin = torch.randint(
            1, test_config.model.num_bins + 3, (batch_size, seq_len)
        )
        normalized_expr = torch.randn(batch_size, seq_len)

        annotation_model.eval()
        with torch.no_grad():
            logits = annotation_model(gene_ids, expression_bin, normalized_expr)
            logger.info(f"✓ Forward pass successful, output shape: {logits.shape}")
            logger.info(
                f"✓ Output range: [{logits.min().item():.3f}, {logits.max().item():.3f}]"
            )

        return True

    except Exception as e:
        logger.error(f"Model integration test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    logger.info("Starting finetune pipeline test...")

    # 设置路径
    dummy_h5ad_path = "/home/angli/baseline/DeepSC/data/finetune/test_data.h5ad"

    # 1. 创建虚拟数据（如果不存在）
    if not os.path.exists(dummy_h5ad_path):
        logger.info("Creating dummy h5ad file for testing...")
        if not create_dummy_h5ad(dummy_h5ad_path, n_cells=1000, n_genes=500):
            logger.error("Failed to create dummy h5ad file")
            return False
    else:
        logger.info(f"Using existing h5ad file: {dummy_h5ad_path}")

    # 2. 测试数据加载
    success, num_classes = test_data_loading(dummy_h5ad_path)
    if not success:
        logger.error("Data loading test failed")
        return False

    # 3. 测试模型集成
    if not test_model_integration(num_classes):
        logger.error("Model integration test failed")
        return False

    logger.info("🎉 All tests passed! The finetune pipeline is ready.")
    logger.info("You can now run fine-tuning with:")
    logger.info(
        f"  python cell_type_annotation_finetune.py data_path={dummy_h5ad_path}"
    )

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
