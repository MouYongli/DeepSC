"""
Inspect h5ad file to see obs and var columns and their unique values
"""

import pandas as pd
import scanpy as sc

# Load the h5ad file
h5ad_path = (
    "/home/angli/DeepSC/data/processed/baseline/scfoundation/hPancreas_merged.h5ad"
)
print(f"Loading: {h5ad_path}")
print("=" * 80)

adata = sc.read_h5ad(h5ad_path)

print(f"\nDataset shape: {adata.shape}")
print(f"Number of cells: {adata.n_obs}")
print(f"Number of genes: {adata.n_vars}")

# Inspect obs (cell metadata)
print("\n" + "=" * 80)
print("OBS (Cell Metadata) Columns:")
print("=" * 80)
print(f"\nTotal columns in obs: {len(adata.obs.columns)}")
print(f"Columns: {list(adata.obs.columns)}\n")

for col in adata.obs.columns:
    print(f"\n{'─' * 80}")
    print(f"Column: {col}")
    print(f"  Data type: {adata.obs[col].dtype}")
    print(f"  Non-null count: {adata.obs[col].notna().sum()} / {len(adata.obs)}")

    # Show unique values
    unique_vals = adata.obs[col].unique()
    n_unique = len(unique_vals)

    if n_unique <= 50:  # Show all if <= 50 unique values
        print(f"  Number of unique values: {n_unique}")
        if pd.api.types.is_numeric_dtype(adata.obs[col]):
            print(f"  Unique values: {sorted(unique_vals)}")
        else:
            # Count occurrences for categorical
            value_counts = adata.obs[col].value_counts()
            print("  Value counts:")
            for val, count in value_counts.items():
                print(f"    {val}: {count}")
    else:  # Show summary for many unique values
        print(f"  Number of unique values: {n_unique} (too many to display)")
        if pd.api.types.is_numeric_dtype(adata.obs[col]):
            print(f"  Min: {adata.obs[col].min()}")
            print(f"  Max: {adata.obs[col].max()}")
            print(f"  Mean: {adata.obs[col].mean():.2f}")
            print(f"  Median: {adata.obs[col].median():.2f}")
        else:
            print("  Top 10 most common values:")
            value_counts = adata.obs[col].value_counts().head(10)
            for val, count in value_counts.items():
                print(f"    {val}: {count}")

# Inspect var (gene metadata)
print("\n\n" + "=" * 80)
print("VAR (Gene Metadata) Columns:")
print("=" * 80)
print(f"\nTotal columns in var: {len(adata.var.columns)}")
print(f"Columns: {list(adata.var.columns)}\n")

for col in adata.var.columns:
    print(f"\n{'─' * 80}")
    print(f"Column: {col}")
    print(f"  Data type: {adata.var[col].dtype}")
    print(f"  Non-null count: {adata.var[col].notna().sum()} / {len(adata.var)}")

    # Show unique values
    unique_vals = adata.var[col].unique()
    n_unique = len(unique_vals)

    if n_unique <= 50:  # Show all if <= 50 unique values
        print(f"  Number of unique values: {n_unique}")
        if pd.api.types.is_numeric_dtype(adata.var[col]):
            print(f"  Unique values: {sorted(unique_vals)}")
        else:
            # Count occurrences for categorical
            value_counts = adata.var[col].value_counts()
            print("  Value counts:")
            for val, count in value_counts.items():
                print(f"    {val}: {count}")
    else:  # Show summary for many unique values
        print(f"  Number of unique values: {n_unique} (too many to display)")
        if pd.api.types.is_numeric_dtype(adata.var[col]):
            print(f"  Min: {adata.var[col].min()}")
            print(f"  Max: {adata.var[col].max()}")
            print(f"  Mean: {adata.var[col].mean():.2f}")
            print(f"  Median: {adata.var[col].median():.2f}")
        else:
            print("  Top 10 most common values:")
            value_counts = adata.var[col].value_counts().head(10)
            for val, count in value_counts.items():
                print(f"    {val}: {count}")

# Check var index (gene names)
print("\n\n" + "=" * 80)
print("VAR Index (Gene Names):")
print("=" * 80)
print(f"  Index name: {adata.var.index.name}")
print(f"  Number of genes: {len(adata.var.index)}")
print(f"  First 10 gene names: {list(adata.var.index[:10])}")
print(f"  Last 10 gene names: {list(adata.var.index[-10:])}")

# Check obs index (cell barcodes)
print("\n\n" + "=" * 80)
print("OBS Index (Cell Barcodes/IDs):")
print("=" * 80)
print(f"  Index name: {adata.obs.index.name}")
print(f"  Number of cells: {len(adata.obs.index)}")
print(f"  First 10 cell IDs: {list(adata.obs.index[:10])}")
print(f"  Last 10 cell IDs: {list(adata.obs.index[-10:])}")

# Check for any additional attributes
print("\n\n" + "=" * 80)
print("Additional AnnData Attributes:")
print("=" * 80)
print(f"  uns keys: {list(adata.uns.keys()) if len(adata.uns.keys()) > 0 else 'None'}")
print(
    f"  obsm keys: {list(adata.obsm.keys()) if len(adata.obsm.keys()) > 0 else 'None'}"
)
print(
    f"  varm keys: {list(adata.varm.keys()) if len(adata.varm.keys()) > 0 else 'None'}"
)
print(
    f"  obsp keys: {list(adata.obsp.keys()) if len(adata.obsp.keys()) > 0 else 'None'}"
)
print(
    f"  varp keys: {list(adata.varp.keys()) if len(adata.varp.keys()) > 0 else 'None'}"
)
print(
    f"  layers keys: {list(adata.layers.keys()) if len(adata.layers.keys()) > 0 else 'None'}"
)

print("\n" + "=" * 80)
print("Inspection Complete!")
print("=" * 80)
