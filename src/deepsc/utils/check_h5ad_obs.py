#!/usr/bin/env python3
from pathlib import Path

import anndata as ad

import sys


def check_h5ad_obs(h5ad_path):
    """Check obs and var columns and unique values in an h5ad file."""
    adata = ad.read_h5ad(h5ad_path)

    results = []
    results.append(f"## {Path(h5ad_path).name}\n")
    results.append(f"**File Path:** `{h5ad_path}`\n")
    results.append(f"**Total cells:** {adata.n_obs}\n")
    results.append(f"**Total genes:** {adata.n_vars}\n")

    # Check obs columns
    results.append("\n### Obs Columns and Unique Values\n")

    if len(adata.obs.columns) == 0:
        results.append("*No obs columns found*\n")
    else:
        for col in adata.obs.columns:
            unique_values = adata.obs[col].unique()
            n_unique = len(unique_values)

            results.append(f"#### {col}")
            results.append(f"- **Number of unique values:** {n_unique}")

            # Show unique values if reasonable number, otherwise show first few
            if n_unique <= 50:
                results.append(
                    f"- **Unique values:** {sorted([str(v) for v in unique_values])}"
                )
            else:
                results.append(
                    f"- **First 20 unique values:** {sorted([str(v) for v in unique_values])[:20]}"
                )
                results.append("- *(Too many unique values to display all)*")
            results.append("")

    # Check var columns
    results.append("\n### Var Columns and Unique Values\n")

    if len(adata.var.columns) == 0:
        results.append("*No var columns found*\n")
    else:
        for col in adata.var.columns:
            unique_values = adata.var[col].unique()
            n_unique = len(unique_values)

            results.append(f"#### {col}")
            results.append(f"- **Number of unique values:** {n_unique}")

            # Show unique values if reasonable number, otherwise show first few
            if n_unique <= 50:
                results.append(
                    f"- **Unique values:** {sorted([str(v) for v in unique_values])}"
                )
            else:
                results.append(
                    f"- **First 20 unique values:** {sorted([str(v) for v in unique_values])[:20]}"
                )
                results.append("- *(Too many unique values to display all)*")
            results.append("")

    return "\n".join(results)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_h5ad_obs.py <h5ad_file1> [h5ad_file2] ...")
        sys.exit(1)

    all_results = ["# H5AD Files Observation Data Analysis\n"]

    for h5ad_path in sys.argv[1:]:
        print(f"Processing {h5ad_path}...")
        try:
            result = check_h5ad_obs(h5ad_path)
            all_results.append(result)
            all_results.append("\n---\n")
        except Exception as e:
            print(f"Error processing {h5ad_path}: {e}")
            all_results.append(f"## Error processing {h5ad_path}\n")
            all_results.append(f"**Error:** {str(e)}\n\n---\n")

    # Save results
    output_path = (
        "/home/angli/baseline/DeepSC/data/cell_type_annotation/h5ad_obs_analysis.md"
    )
    with open(output_path, "w") as f:
        f.write("\n".join(all_results))

    print(f"\nResults saved to: {output_path}")
