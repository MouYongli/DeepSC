#!/usr/bin/env python3
"""
Script to create a balanced dataset from kidney partition_3.h5ad file.
Filters for is_primary_data=True and samples N cells per cell_type
(only for cell_types with >N cells).
"""

import os

import anndata as ad
import numpy as np

import argparse


def create_balanced_dataset(
    input_h5ad_path, output_h5ad_path, n_cells_per_type=500, min_cells_threshold=500
):
    """
    Create a balanced dataset by sampling cells from each cell type.

    Parameters:
    -----------
    input_h5ad_path : str
        Path to the input h5ad file
    output_h5ad_path : str
        Path to save the output h5ad file
    n_cells_per_type : int
        Number of cells to sample per cell type (default: 500)
    min_cells_threshold : int
        Minimum number of cells required for a cell type to be included (default: 500)
    """
    print(f"Loading h5ad file: {input_h5ad_path}")
    print("=" * 80)

    # Read the h5ad file
    adata = ad.read_h5ad(input_h5ad_path)
    print(f"Original dataset shape: {adata.shape[0]} cells × {adata.shape[1]} genes")

    # Filter for is_primary_data == True
    print("\nFiltering for is_primary_data == True...")
    if "is_primary_data" in adata.obs.columns:
        adata_filtered = adata[adata.obs["is_primary_data"]].copy()
        print(
            f"After filtering: {adata_filtered.shape[0]} cells × {adata_filtered.shape[1]} genes"
        )
    else:
        print("Warning: 'is_primary_data' column not found. Using all cells.")
        adata_filtered = adata.copy()

    # Check cell_type column
    if "cell_type" not in adata_filtered.obs.columns:
        raise ValueError("'cell_type' column not found in obs dataframe!")

    # Count cells per cell type
    print("\n" + "=" * 80)
    print("Cell type distribution:")
    print("=" * 80)
    cell_type_counts = (
        adata_filtered.obs["cell_type"].value_counts().sort_values(ascending=False)
    )

    for cell_type, count in cell_type_counts.items():
        print(f"  {cell_type}: {count} cells")

    # Filter cell types with more than min_cells_threshold cells
    print("\n" + "=" * 80)
    print(f"Filtering cell types with > {min_cells_threshold} cells:")
    print("=" * 80)

    eligible_cell_types = cell_type_counts[
        cell_type_counts > min_cells_threshold
    ].index.tolist()
    print(f"Number of eligible cell types: {len(eligible_cell_types)}")
    print("\nEligible cell types:")
    for cell_type in eligible_cell_types:
        count = cell_type_counts[cell_type]
        print(f"  {cell_type}: {count} cells")

    # Sample n_cells_per_type from each eligible cell type
    print("\n" + "=" * 80)
    print(f"Sampling {n_cells_per_type} cells per cell type:")
    print("=" * 80)

    sampled_indices = []
    np.random.seed(42)  # For reproducibility

    for cell_type in eligible_cell_types:
        # Get indices of cells with this cell type
        cell_type_mask = adata_filtered.obs["cell_type"] == cell_type
        cell_type_indices = np.where(cell_type_mask)[0]

        # Randomly sample n_cells_per_type cells
        sampled_idx = np.random.choice(
            cell_type_indices, size=n_cells_per_type, replace=False
        )
        sampled_indices.extend(sampled_idx)

        print(f"  {cell_type}: sampled {n_cells_per_type} cells")

    # Create the balanced dataset
    print("\n" + "=" * 80)
    print("Creating balanced dataset:")
    print("=" * 80)

    adata_balanced = adata_filtered[sampled_indices].copy()

    print(
        f"Balanced dataset shape: {adata_balanced.shape[0]} cells × {adata_balanced.shape[1]} genes"
    )
    print(f"Number of cell types: {adata_balanced.obs['cell_type'].nunique()}")

    # Verify the balanced distribution
    print("\n" + "=" * 80)
    print("Final cell type distribution in balanced dataset:")
    print("=" * 80)
    final_counts = adata_balanced.obs["cell_type"].value_counts().sort_index()
    for cell_type, count in final_counts.items():
        print(f"  {cell_type}: {count} cells")

    # Save the balanced dataset
    print("\n" + "=" * 80)
    print(f"Saving balanced dataset to: {output_h5ad_path}")
    print("=" * 80)
    adata_balanced.write_h5ad(output_h5ad_path)

    print("\n" + "=" * 80)
    print("Success! Balanced dataset created.")
    print("=" * 80)
    print("\nSummary:")
    print(f"  - Input cells: {adata.shape[0]}")
    print(f"  - After is_primary_data filter: {adata_filtered.shape[0]}")
    print(
        f"  - Eligible cell types (>{min_cells_threshold} cells): {len(eligible_cell_types)}"
    )
    print(f"  - Final balanced dataset: {adata_balanced.shape[0]} cells")
    print(f"  - Cells per type: {n_cells_per_type}")
    print(f"  - Output file: {output_h5ad_path}")


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Create a balanced dataset by sampling cells from each cell type."
    )

    parser.add_argument(
        "--n_cells",
        type=int,
        default=500,
        help="Number of cells to sample per cell type (default: 500)",
    )

    parser.add_argument(
        "--input",
        type=str,
        default="/home/angli/baseline/DeepSC/data/cellxgene/raw/kidney/partition_3.h5ad",
        help="Path to input h5ad file (default: partition_3.h5ad)",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/angli/baseline/DeepSC/data/cell_type_annotation/balanced_dataset",
        help="Directory to save output file (default: balanced_dataset directory)",
    )

    parser.add_argument(
        "--min_threshold",
        type=int,
        default=None,
        help=(
            "Minimum number of cells required for a cell type to be included. "
            "If not specified, uses the same value as --n_cells"
        ),
    )

    args = parser.parse_args()

    # Set parameters
    n_cells_per_type = args.n_cells
    min_cells_threshold = (
        args.min_threshold if args.min_threshold is not None else args.n_cells
    )

    # Generate output filename based on n_cells_per_type
    output_filename = f"balanced_dataset_kidney_{n_cells_per_type}.h5ad"
    output_file = os.path.join(args.output_dir, output_filename)

    print("Parameters:")
    print(f"  - Input file: {args.input}")
    print(f"  - Output file: {output_file}")
    print(f"  - Cells per type: {n_cells_per_type}")
    print(f"  - Min threshold: {min_cells_threshold}")
    print()

    create_balanced_dataset(
        args.input,
        output_file,
        n_cells_per_type=n_cells_per_type,
        min_cells_threshold=min_cells_threshold,
    )
