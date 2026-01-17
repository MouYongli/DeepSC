#!/usr/bin/env python
"""
Test script: explore data structure using the same data loading logic as pp_new.py

Usage:
    PYTHONPATH=src python test_pp_dataloader.py
"""
import torch
from gears import PertData


def explore_batch_data_detailed(batch_data, batch_idx=0):
    """Explore batch_data structure in detail"""
    print("\n" + "=" * 80)
    print(f"📦 Batch {batch_idx} detailed structure")
    print("=" * 80)

    # Basic information
    batch_size = len(batch_data.y)
    print(f"\n✓ batch_size = {batch_size}")

    # batch_data.pert
    print("\n[batch_data.pert]")
    print("  Type: {}".format(type(batch_data.pert)))
    print("  Length: {}".format(len(batch_data.pert)))
    print("  Examples (first 5):")
    for i, pert in enumerate(batch_data.pert[:5]):
        print(f"    [{i}] {pert}")

    # batch_data.y
    print("\n[batch_data.y]")
    print("  Type: {}".format(type(batch_data.y)))
    print("  Shape: {}".format(batch_data.y.shape))
    print("  Dtype: {}".format(batch_data.y.dtype))
    print("  Device: {}".format(batch_data.y.device))
    print("  Values (first 3 samples, first 10 genes):")
    print(batch_data.y[:3, :10])
    print("  Min: {:.4f}, Max: {:.4f}".format(batch_data.y.min(), batch_data.y.max()))

    # batch_data.x
    print("\n[batch_data.x]")
    print("  Type: {}".format(type(batch_data.x)))
    print("  Shape: {}".format(batch_data.x.shape))
    print("  Dtype: {}".format(batch_data.x.dtype))
    print("  Device: {}".format(batch_data.x.device))
    print("  Note: x[:, 0] is input expression, x[:, 1] is perturbation flag")
    print("  First 3 rows:")
    print(batch_data.x[:3])

    # Reconstruct to (batch_size, num_genes)
    num_genes = batch_data.x.shape[0] // batch_size
    ori_gene_values = batch_data.x[:, 0].view(batch_size, num_genes)
    print("\n  Reshaped ori_gene_values: {}".format(ori_gene_values.shape))
    print("  First 10 gene expressions of first sample:")
    print("    {}".format(ori_gene_values[0, :10]))

    # Other attributes of batch_data
    print("\n[Other attributes of batch_data]")
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
    print("🚀 Start testing Perturbation Prediction data loader\n")

    # Configuration parameters (simulate pp.yaml)
    config = {
        "data_name": "adamson",  # Use adamson dataset
        "split": "simulation",
        "seed": 1,
        "batch_size": 32,
        "test_batch_size": 128,
        "include_zero_gene": "all",  # or 'batch-wise'
    }

    print("📋 Configuration info:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Load data
    print("\n📂 Initializing PertData...")
    pert_data = PertData("./data")

    print(f"\n📥 Loading dataset (data_name={config['data_name']})...")
    pert_data.load(data_name=config["data_name"])
    print(
        f"  Dataset loaded: {pert_data.dataset_name if hasattr(pert_data, 'dataset_name') else config['data_name']}"
    )

    # Prepare dataset
    print(
        f"\n🔧 Preparing data split (split={config['split']}, seed={config['seed']})..."
    )
    pert_data.prepare_split(split=config["split"], seed=config["seed"])
    pert_data.get_dataloader(
        batch_size=config["batch_size"], test_batch_size=config["test_batch_size"]
    )

    # DataLoader information
    train_loader = pert_data.dataloader["train_loader"]
    val_loader = pert_data.dataloader["val_loader"]
    test_loader = pert_data.dataloader["test_loader"]

    print("\n📊 DataLoader statistics:")
    print("  Train batches: {}".format(len(train_loader)))
    print("  Val batches:   {}".format(len(val_loader)))
    print("  Test batches:  {}".format(len(test_loader)))

    # Explore first 3 batches of training set
    print("\n" + "=" * 80)
    print("🔍 Exploring training set (first 3 batches)")
    print("=" * 80)

    train_iter = iter(train_loader)
    for i in range(min(3, len(train_loader))):
        batch_data = next(train_iter)
        explore_batch_data_detailed(batch_data, batch_idx=i)

    # Explore first batch of validation set
    print("\n" + "=" * 80)
    print("🔍 Exploring validation set (1st batch)")
    print("=" * 80)

    val_iter = iter(val_loader)
    batch_data = next(val_iter)
    explore_batch_data_detailed(batch_data, batch_idx=0)

    # Additional: Check data consistency
    print("\n" + "=" * 80)
    print("🧪 Data consistency check")
    print("=" * 80)

    # Get a batch again
    train_iter = iter(train_loader)
    batch_data = next(train_iter)
    batch_size = len(batch_data.y)
    num_genes = batch_data.x.shape[0] // batch_size

    print("\nBasic information:")
    print("  batch_size = {}".format(batch_size))
    print("  num_genes = {}".format(num_genes))
    print("  batch_data.y.shape = {}".format(batch_data.y.shape))
    print("  batch_data.x.shape = {}".format(batch_data.x.shape))

    # Check shape consistency
    assert (
        batch_data.y.shape[0] == batch_size
    ), "y's first dimension should equal batch_size"
    assert (
        batch_data.y.shape[1] == num_genes
    ), "y's second dimension should equal num_genes"
    assert (
        batch_data.x.shape[0] == batch_size * num_genes
    ), "x's row count should equal batch_size * num_genes"
    print("\n✅ Shape consistency check passed!")

    # Check the effect of include_zero_gene
    print(f"\nTesting include_zero_gene={config['include_zero_gene']}:")
    if config["include_zero_gene"] == "all":
        print("  ➜ All genes will be included in the input")
        input_gene_ids = torch.arange(num_genes, dtype=torch.long)
        print(
            f"  ➜ input_gene_ids range: [0, {num_genes-1}], length: {len(input_gene_ids)}"
        )
    elif config["include_zero_gene"] == "batch-wise":
        print("  ➜ Only include genes with expression in current batch")
        ori_gene_values = batch_data.x[:, 0].view(batch_size, num_genes)
        input_gene_ids = torch.where(ori_gene_values.sum(dim=0) > 0)[0]
        print(f"  ➜ Number of expressed genes: {len(input_gene_ids)} / {num_genes}")
        print(f"  ➜ First 10 expressed gene IDs: {input_gene_ids[:10]}")

    print("\n" + "=" * 80)
    print("✅ Data structure exploration completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
