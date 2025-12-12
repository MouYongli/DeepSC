#!/usr/bin/env python3
"""
Script to analyze the columns in the obs dataframe of an h5ad file.
"""

import anndata as ad

import argparse


def analyze_obs_columns(h5ad_path):
    """
    Analyze and display the columns in the obs dataframe of an h5ad file.

    Parameters:
    -----------
    h5ad_path : str
        Path to the h5ad file
    """
    print(f"Loading h5ad file: {h5ad_path}")
    print("-" * 80)

    # Read the h5ad file
    adata = ad.read_h5ad(h5ad_path)

    # Get basic information
    print(f"\nDataset shape: {adata.shape[0]} cells × {adata.shape[1]} genes")
    print(f"\nNumber of obs columns: {len(adata.obs.columns)}")
    print("-" * 80)

    # Display all column names
    print("\nColumn names in obs:")
    print("-" * 80)
    for i, col in enumerate(adata.obs.columns, 1):
        print(f"{i:3d}. {col}")

    # Display column information with data types and non-null counts
    print("\n" + "=" * 80)
    print("Detailed column information:")
    print("=" * 80)
    print(adata.obs.info())

    # Display unique value counts for categorical columns
    print("\n" + "=" * 80)
    print("Sample values and unique counts for each column:")
    print("=" * 80)

    for col in adata.obs.columns:
        print(f"\nColumn: {col}")
        print(f"  Data type: {adata.obs[col].dtype}")

        # Count unique values
        n_unique = adata.obs[col].nunique()
        print(f"  Unique values: {n_unique}")

        # Show sample values
        if n_unique <= 20:
            print("  All unique values:")
            value_counts = adata.obs[col].value_counts()
            for val, count in value_counts.items():
                print(f"    - {val}: {count} cells")
        else:
            print("  Top 10 most common values:")
            value_counts = adata.obs[col].value_counts().head(10)
            for val, count in value_counts.items():
                print(f"    - {val}: {count} cells")

        # Show missing values if any
        n_missing = adata.obs[col].isna().sum()
        if n_missing > 0:
            print(
                f"  Missing values: {n_missing} ({n_missing/len(adata.obs)*100:.2f}%)"
            )

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Analyze the columns in the obs dataframe of an h5ad file."
    )

    parser.add_argument(
        "--input",
        type=str,
        default="/home/angli/baseline/DeepSC/data/cellxgene/raw/kidney/partition_3.h5ad",
        help="Path to the input h5ad file (default: partition_3.h5ad)",
    )

    args = parser.parse_args()

    analyze_obs_columns(args.input)
