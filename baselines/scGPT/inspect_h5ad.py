#!/usr/bin/env python3
"""
Script to inspect obs and var columns in an h5ad file
"""

import scanpy as sc
import pandas as pd

# Load the h5ad file
adata_path = "/home/angli/baseline/DeepSC/data/cell_type_annotation/myeloid_merged.train.h5ad"
print(f"Loading data from: {adata_path}")
adata = sc.read_h5ad(adata_path)

print(f"\n{'='*80}")
print(f"Dataset shape: {adata.shape} (cells x genes)")
print(f"{'='*80}\n")

# Inspect obs (cell metadata)
print("=" * 80)
print("OBS (Cell Metadata) Columns:")
print("=" * 80)
print(f"\nTotal columns in obs: {len(adata.obs.columns)}\n")

for col in adata.obs.columns:
    print(f"\n--- Column: '{col}' ---")
    print(f"Data type: {adata.obs[col].dtype}")

    # Get unique values
    unique_values = adata.obs[col].unique()
    n_unique = len(unique_values)

    print(f"Number of unique values: {n_unique}")

    # If categorical or small number of unique values, show all
    if n_unique <= 50:
        try:
            print(f"Unique values: {sorted(unique_values.tolist())}")
        except TypeError:
            print(f"Unique values: {unique_values.tolist()[:50]}")
    else:
        try:
            sorted_values = sorted(unique_values.tolist())
            print(f"First 20 unique values: {sorted_values[:20]}")
        except TypeError:
            print(f"First 20 unique values: {unique_values.tolist()[:20]}")
        print(f"(... {n_unique - 20} more values)")

    # Show value counts for categorical-like columns
    if n_unique <= 100:
        print(f"\nValue counts:")
        value_counts = adata.obs[col].value_counts()
        for val, count in value_counts.items():
            print(f"  {val}: {count}")

print("\n\n")

# Inspect var (gene metadata)
print("=" * 80)
print("VAR (Gene Metadata) Columns:")
print("=" * 80)
print(f"\nTotal columns in var: {len(adata.var.columns)}\n")

for col in adata.var.columns:
    print(f"\n--- Column: '{col}' ---")
    print(f"Data type: {adata.var[col].dtype}")

    # Get unique values
    unique_values = adata.var[col].unique()
    n_unique = len(unique_values)

    print(f"Number of unique values: {n_unique}")

    # If categorical or small number of unique values, show all
    if n_unique <= 50:
        try:
            print(f"Unique values: {sorted(unique_values.tolist())}")
        except TypeError:
            print(f"Unique values: {unique_values.tolist()[:50]}")
    else:
        try:
            sorted_values = sorted(unique_values.tolist())
            print(f"First 20 unique values: {sorted_values[:20]}")
        except TypeError:
            print(f"First 20 unique values: {unique_values.tolist()[:20]}")
        print(f"(... {n_unique - 20} more values)")

    # Show value counts for categorical-like columns
    if n_unique <= 100:
        print(f"\nValue counts:")
        value_counts = adata.var[col].value_counts()
        for val, count in value_counts.items():
            print(f"  {val}: {count}")

print("\n" + "=" * 80)
print("Inspection complete!")
print("=" * 80)
