#!/usr/bin/env python
"""
测试脚本：探索 PertData 的数据结构

用法：
    python test_dataloader.py
"""
import torch
from gears import PertData


def explore_batch_data(batch_data, batch_idx=0):
    """
    探索并打印 batch_data 的所有属性和数据结构

    Args:
        batch_data: 来自 DataLoader 的一个 batch
        batch_idx: batch 的索引（用于打印）
    """
    print("=" * 80)
    print(f"Batch {batch_idx} 数据结构探索")
    print("=" * 80)

    # 1. 打印所有属性
    print("\n【1. batch_data 的所有属性】")
    attributes = [attr for attr in dir(batch_data) if not attr.startswith("_")]
    print(f"属性列表: {attributes}")

    # 2. 探索关键属性
    print("\n【2. 关键属性详细信息】")

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
        print(f"  - 示例扰动: {batch_data.pert[0]}")

    # 2.3 batch_data.x
    if hasattr(batch_data, "x"):
        print("\nbatch_data.x:")
        print(f"  - type: {type(batch_data.x)}")
        print(f"  - shape: {batch_data.x.shape}")
        print(f"  - dtype: {batch_data.x.dtype}")
        print(f"  - device: {batch_data.x.device}")
        print(f"  - first 3 rows:\n{batch_data.x[:3]}")

    # 2.4 batch_data.edge_index (如果有)
    if hasattr(batch_data, "edge_index"):
        print("\nbatch_data.edge_index:")
        print(f"  - type: {type(batch_data.edge_index)}")
        print(f"  - shape: {batch_data.edge_index.shape}")
        print(f"  - first 5 edges:\n{batch_data.edge_index[:, :5]}")

    # 2.5 batch_data.batch (如果有)
    if hasattr(batch_data, "batch"):
        print("\nbatch_data.batch:")
        print(f"  - type: {type(batch_data.batch)}")
        print(f"  - shape: {batch_data.batch.shape}")
        print(f"  - unique values: {torch.unique(batch_data.batch).tolist()}")

    # 2.6 其他可能的属性
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
    print("开始加载数据...")

    # 初始化 PertData
    pert_data = PertData("./data")

    # 加载数据集
    data_name = "adamson"  # 或 "norman", "dixit" 等
    print(f"\n加载数据集: {data_name}")
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

    # 探索验证集的一个 batch
    print("\n\n探索验证集的第一个 batch:")
    val_iter = iter(pert_data.dataloader["val_loader"])
    batch_data = next(val_iter)
    explore_batch_data(batch_data, batch_idx=0)

    print("\n✅ 数据结构探索完成！")


if __name__ == "__main__":
    main()
