#!/usr/bin/env python
"""
Test script: explore PertData data structure

Usage:
    python test_dataloader.py
"""
import torch
from gears import PertData


def explore_batch_data(batch_data, batch_idx=0):
    """
    Explore and print all attributes and data structure of batch_data

    Args:
        batch_data: a batch from DataLoader
        batch_idx: batch index (for printing)
    """
    print("=" * 80)
    print(f"Batch {batch_idx} data structure exploration")
    print("=" * 80)

    # 1. Print all attributes
    print("\n【1. All attributes of batch_data】")
    attributes = [attr for attr in dir(batch_data) if not attr.startswith("_")]
    print(f"Attribute list: {attributes}")

    # 2. Explore key attributes
    print("\n【2. Key attribute details】")

    # 2.1 batch_data.y
    if hasattr(batch_data, "y"):
        print("\nbatch_data.y:")
        print(f"  - type: {type(batch_data.y)}")
        print(
            f"  - shape: {batch_data.y.shape if hasattr(batch_data.y, 'shape') else 'N/A'}"
        )
        print(
            f"  - dtype: {batch_data.y.dtype if hasattr(batch_data.y, 'dtype') else 'N/A'}"
        )
        print(f"  - len: {len(batch_data.y)}")
        print(f"  - first 3 values: {batch_data.y[:3]}")
        batch_size = len(batch_data.y)
        print(f"  ➜ batch_size = {batch_size}")

    # 2.2 batch_data.pert
    if hasattr(batch_data, "pert"):
        print("\nbatch_data.pert:")
        print(f"  - type: {type(batch_data.pert)}")
        print(f"  - len: {len(batch_data.pert)}")
        print(f"  - first 3 values: {batch_data.pert[:3]}")
        print(f"  - Example perturbation: {batch_data.pert[0]}")

    # 2.3 batch_data.x
    if hasattr(batch_data, "x"):
        print("\nbatch_data.x:")
        print(f"  - type: {type(batch_data.x)}")
        print(f"  - shape: {batch_data.x.shape}")
        print(f"  - dtype: {batch_data.x.dtype}")
        print(f"  - device: {batch_data.x.device}")
        print(f"  - first 3 rows:\n{batch_data.x[:3]}")

    # 2.4 batch_data.edge_index (if exists)
    if hasattr(batch_data, "edge_index"):
        print("\nbatch_data.edge_index:")
        print(f"  - type: {type(batch_data.edge_index)}")
        print(f"  - shape: {batch_data.edge_index.shape}")
        print(f"  - first 5 edges:\n{batch_data.edge_index[:, :5]}")

    # 2.5 batch_data.batch (if exists)
    if hasattr(batch_data, "batch"):
        print("\nbatch_data.batch:")
        print(f"  - type: {type(batch_data.batch)}")
        print(f"  - shape: {batch_data.batch.shape}")
        print(f"  - unique values: {torch.unique(batch_data.batch).tolist()}")

    # 2.6 Other possible attributes
    other_attrs = ["pert_idx", "dose", "ctrl", "condition"]
    for attr in other_attrs:
        if hasattr(batch_data, attr):
            value = getattr(batch_data, attr)
            print(f"\nbatch_data.{attr}:")
            print(f"  - type: {type(value)}")
            if isinstance(value, torch.Tensor):
                print(f"  - shape: {value.shape}")
                print(f"  - first 3: {value[:3]}")
            else:
                print(f"  - value: {value[:3] if hasattr(value, '__len__') else value}")

    print("\n" + "=" * 80)


def main():
    print("Starting to load data...")

    # Initialize PertData
    pert_data = PertData("./data")

    # Load data集
    data_name = "adamson"  # or "norman", "dixit" etc
    print(f"\nLoading dataset: {data_name}")
    pert_data.load(data_name=data_name)

    # 获取数据集信息
    print("\n【PertData 基本信息】")
    print(
        f"Dataset: {pert_data.dataset_name if hasattr(pert_data, 'dataset_name') else data_name}"
    )

    # 准备数据集
    print("\n准备数据集...")
    pert_data.prepare_split(split="simulation", seed=1)
    pert_data.get_dataloader(batch_size=32, test_batch_size=128)

    # 获取数据加载器
    print("\n【DataLoader 信息】")
    print(f"训练集 batches: {len(pert_data.dataloader['train_loader'])}")
    print(f"验证集 batches: {len(pert_data.dataloader['val_loader'])}")
    print(f"测试集 batches: {len(pert_data.dataloader['test_loader'])}")

    # 探索第一个 batch
    print("\n\n探索训练集的第一个 batch:")
    data_iter = iter(pert_data.dataloader["train_loader"])
    batch_data = next(data_iter)
    explore_batch_data(batch_data, batch_idx=0)

    # 再探索第二个 batch（对比）
    print("\n\n探索训练集的第二个 batch:")
    batch_data = next(data_iter)
    explore_batch_data(batch_data, batch_idx=1)

    # 探索验证集 a batch
    print("\n\n探索验证集的第一个 batch:")
    val_iter = iter(pert_data.dataloader["val_loader"])
    batch_data = next(val_iter)
    explore_batch_data(batch_data, batch_idx=0)

    print("\n✅ 数据结构探索完成！")


if __name__ == "__main__":
    main()
