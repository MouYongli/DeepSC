#!/usr/bin/env python3
"""
Script to validate a balanced dataset.
Checks:
1. All cells have is_primary_data == True
2. All cell types have the same number of cells
"""

import anndata as ad

import argparse
import sys


def validate_balanced_dataset(h5ad_path, expected_n_cells=None):
    """
    Validate a balanced dataset.

    Parameters:
    -----------
    h5ad_path : str
        Path to the h5ad file to validate
    expected_n_cells : int, optional
        Expected number of cells per cell type

    Returns:
    --------
    bool
        True if all validations pass, False otherwise
    """
    print("=" * 80)
    print("Validating Balanced Dataset")
    print("=" * 80)
    print(f"Input file: {h5ad_path}")
    print()

    # Read the h5ad file
    try:
        adata = ad.read_h5ad(h5ad_path)
        print("✓ Successfully loaded dataset")
        print(f"  Shape: {adata.shape[0]} cells × {adata.shape[1]} genes")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        return False

    all_tests_passed = True

    # Test 1: Check if is_primary_data column exists
    print("\n" + "-" * 80)
    print("Test 1: Checking is_primary_data column")
    print("-" * 80)

    if "is_primary_data" not in adata.obs.columns:
        print("✗ FAILED: 'is_primary_data' column not found in obs")
        all_tests_passed = False
    else:
        # Test 2: Check if all cells have is_primary_data == True
        n_primary = (adata.obs["is_primary_data"]).sum()
        n_total = len(adata.obs)

        print(f"  Total cells: {n_total}")
        print(f"  Cells with is_primary_data == True: {n_primary}")

        if n_primary == n_total:
            print("✓ PASSED: All cells have is_primary_data == True")
        else:
            n_not_primary = n_total - n_primary
            print(
                f"✗ FAILED: {n_not_primary} cells do not have is_primary_data == True"
            )
            print(f"  Percentage of primary data: {n_primary/n_total*100:.2f}%")
            all_tests_passed = False

            # Show breakdown of is_primary_data values
            print("\n  Breakdown of is_primary_data values:")
            for val, count in adata.obs["is_primary_data"].value_counts().items():
                print(f"    {val}: {count} cells")

    # Test 3: Check cell_type column exists
    print("\n" + "-" * 80)
    print("Test 2: Checking cell_type distribution")
    print("-" * 80)

    if "cell_type" not in adata.obs.columns:
        print("✗ FAILED: 'cell_type' column not found in obs")
        all_tests_passed = False
    else:
        # Test 4: Check if all cell types have the same number of cells
        cell_type_counts = adata.obs["cell_type"].value_counts()
        unique_counts = cell_type_counts.unique()

        print(f"  Number of cell types: {len(cell_type_counts)}")
        print(f"  Unique cell counts: {unique_counts}")

        if len(unique_counts) == 1:
            n_cells_per_type = unique_counts[0]
            print(f"✓ PASSED: All cell types have exactly {n_cells_per_type} cells")

            # If expected_n_cells is provided, check if it matches
            if expected_n_cells is not None:
                if n_cells_per_type == expected_n_cells:
                    print(
                        f"✓ PASSED: Cell count matches expected value ({expected_n_cells})"
                    )
                else:
                    print(
                        f"✗ FAILED: Cell count ({n_cells_per_type}) does not match "
                        f"expected value ({expected_n_cells})"
                    )
                    all_tests_passed = False
        else:
            print("✗ FAILED: Cell types have different numbers of cells")
            all_tests_passed = False

            print("\n  Cell type distribution:")
            for cell_type, count in cell_type_counts.sort_index().items():
                print(f"    {cell_type}: {count} cells")

            print("\n  Statistics:")
            print(f"    Min: {cell_type_counts.min()} cells")
            print(f"    Max: {cell_type_counts.max()} cells")
            print(f"    Mean: {cell_type_counts.mean():.2f} cells")
            print(f"    Std: {cell_type_counts.std():.2f} cells")

    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    if all_tests_passed:
        print("✓ ALL TESTS PASSED")
        print("  The dataset is properly balanced and filtered.")
        return_code = 0
    else:
        print("✗ SOME TESTS FAILED")
        print("  Please review the failures above.")
        return_code = 1

    print("=" * 80)

    return return_code == 0


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Validate a balanced dataset for cell type annotation."
    )

    parser.add_argument(
        "--input", type=str, required=True, help="Path to the h5ad file to validate"
    )

    parser.add_argument(
        "--expected_n_cells",
        type=int,
        default=None,
        help="Expected number of cells per cell type (optional)",
    )

    args = parser.parse_args()

    # Run validation
    success = validate_balanced_dataset(args.input, args.expected_n_cells)

    # Exit with appropriate code
    sys.exit(0 if success else 1)
