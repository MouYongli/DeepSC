#!/usr/bin/env python
"""
测试脚本：使用 pp_new.py 相同的数据加载逻辑探索数据结构

用法：
    PYTHONPATH=src python test_pp_dataloader.py
"""
import torch
from gears import PertData


def explore_batch_data_detailed(batch_data, batch_idx=0):
    """详细探索 batch_data 的结构"""
    print("\n" + "=" * 80)
    print(f"📦 Batch {batch_idx} 详细结构")
    print("=" * 80)

    # 基本信息
    batch_size = len(batch_data.y)
    print(f"\n✓ batch_size = {batch_size}")

    # batch_data.pert
    print("\n【batch_data.pert】")
    print("  Type: {}".format(type(batch_data.pert)))
    print("  Length: {}".format(len(batch_data.pert)))
    print("  示例 (前5个):")
    for i, pert in enumerate(batch_data.pert[:5]):
        print(f"    [{i}] {pert}")

    # batch_data.y
    print("\n【batch_data.y】")
    print("  Type: {}".format(type(batch_data.y)))
    print("  Shape: {}".format(batch_data.y.shape))
    print("  Dtype: {}".format(batch_data.y.dtype))
    print("  Device: {}".format(batch_data.y.device))
    print("  Values (前3个样本, 前10个基因):")
    print(batch_data.y[:3, :10])
    print("  Min: {:.4f}, Max: {:.4f}".format(batch_data.y.min(), batch_data.y.max()))

    # batch_data.x
    print("\n【batch_data.x】")
    print("  Type: {}".format(type(batch_data.x)))
    print("  Shape: {}".format(batch_data.x.shape))
    print("  Dtype: {}".format(batch_data.x.dtype))
    print("  Device: {}".format(batch_data.x.device))
    print("  说明: x[:, 0] 是输入表达量, x[:, 1] 是扰动标志")
    print("  前3行:")
    print(batch_data.x[:3])

    # 重构为 (batch_size, num_genes)
    num_genes = batch_data.x.shape[0] // batch_size
    ori_gene_values = batch_data.x[:, 0].view(batch_size, num_genes)
    print("\n  Reshaped ori_gene_values: {}".format(ori_gene_values.shape))
    print("  第一个样本的前10个基因表达量:")
    print("    {}".format(ori_gene_values[0, :10]))

    # batch_data 的其他属性
    print("\n【batch_data 其他属性】")
    other_attrs = ["edge_index", "batch", "pert_idx", "dose", "ctrl"]
    for attr in other_attrs:
        if hasattr(batch_data, attr):
            value = getattr(batch_data, attr)
            print(f"  {attr}:")
            if isinstance(value, torch.Tensor):
                print(f"    Shape: {value.shape}, Dtype: {value.dtype}")
                if value.numel() <= 10:
                    print(f"    Value: {value}")
                else:
                    print(f"    First 5: {value.flatten()[:5]}")
            else:
                print(f"    Type: {type(value)}, Value: {value}")

    print("\n" + "=" * 80)


def main():
    print("🚀 开始测试 Perturbation Prediction 数据加载器\n")

    # 配置参数（模拟 pp.yaml）
    config = {
        "data_name": "adamson",  # 使用 adamson 数据集
        "split": "simulation",
        "seed": 1,
        "batch_size": 32,
        "test_batch_size": 128,
        "include_zero_gene": "all",  # 或 'batch-wise'
    }

    print("📋 配置信息:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # 加载数据
    print("\n📂 初始化 PertData...")
    pert_data = PertData("./data")

    print(f"\n📥 加载数据集 (data_name={config['data_name']})...")
    pert_data.load(data_name=config["data_name"])
    print(
        f"  Dataset loaded: {pert_data.dataset_name if hasattr(pert_data, 'dataset_name') else config['data_name']}"
    )

    # 准备数据集
    print(f"\n🔧 准备数据划分 (split={config['split']}, seed={config['seed']})...")
    pert_data.prepare_split(split=config["split"], seed=config["seed"])
    pert_data.get_dataloader(
        batch_size=config["batch_size"], test_batch_size=config["test_batch_size"]
    )

    # 数据加载器信息
    train_loader = pert_data.dataloader["train_loader"]
    val_loader = pert_data.dataloader["val_loader"]
    test_loader = pert_data.dataloader["test_loader"]

    print("\n📊 DataLoader 统计:")
    print("  Train batches: {}".format(len(train_loader)))
    print("  Val batches:   {}".format(len(val_loader)))
    print("  Test batches:  {}".format(len(test_loader)))

    # 探索训练集的前3个 batch
    print("\n" + "=" * 80)
    print("🔍 探索训练集 (前3个 batches)")
    print("=" * 80)

    train_iter = iter(train_loader)
    for i in range(min(3, len(train_loader))):
        batch_data = next(train_iter)
        explore_batch_data_detailed(batch_data, batch_idx=i)

    # 探索验证集的第一个 batch
    print("\n" + "=" * 80)
    print("🔍 探索验证集 (第1个 batch)")
    print("=" * 80)

    val_iter = iter(val_loader)
    batch_data = next(val_iter)
    explore_batch_data_detailed(batch_data, batch_idx=0)

    # 额外：检查数据的一致性
    print("\n" + "=" * 80)
    print("🧪 数据一致性检查")
    print("=" * 80)

    # 重新获取一个 batch
    train_iter = iter(train_loader)
    batch_data = next(train_iter)
    batch_size = len(batch_data.y)
    num_genes = batch_data.x.shape[0] // batch_size

    print("\n基本信息:")
    print("  batch_size = {}".format(batch_size))
    print("  num_genes = {}".format(num_genes))
    print("  batch_data.y.shape = {}".format(batch_data.y.shape))
    print("  batch_data.x.shape = {}".format(batch_data.x.shape))

    # 检查形状一致性
    assert batch_data.y.shape[0] == batch_size, "y 的第一维应该等于 batch_size"
    assert batch_data.y.shape[1] == num_genes, "y 的第二维应该等于 num_genes"
    assert (
        batch_data.x.shape[0] == batch_size * num_genes
    ), "x 的行数应该等于 batch_size * num_genes"
    print("\n✅ 形状一致性检查通过!")

    # 检查 include_zero_gene 的影响
    print(f"\n测试 include_zero_gene={config['include_zero_gene']}:")
    if config["include_zero_gene"] == "all":
        print("  ➜ 所有基因都会被包含在输入中")
        input_gene_ids = torch.arange(num_genes, dtype=torch.long)
        print(
            f"  ➜ input_gene_ids 范围: [0, {num_genes-1}], 长度: {len(input_gene_ids)}"
        )
    elif config["include_zero_gene"] == "batch-wise":
        print("  ➜ 只包含当前 batch 中有表达的基因")
        ori_gene_values = batch_data.x[:, 0].view(batch_size, num_genes)
        input_gene_ids = torch.where(ori_gene_values.sum(dim=0) > 0)[0]
        print(f"  ➜ 有表达的基因数: {len(input_gene_ids)} / {num_genes}")
        print(f"  ➜ 第一个10个有表达的基因ID: {input_gene_ids[:10]}")

    print("\n" + "=" * 80)
    print("✅ 数据结构探索完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
