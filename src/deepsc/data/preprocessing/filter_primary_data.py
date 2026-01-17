#!/usr/bin/env python3

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import anndata as ad
import pandas as pd

import argparse


def to_bool_series(col: pd.Series) -> pd.Series:
    """Convert is_primary_data column to a strict boolean Series.

    Missing or unknown values are treated as False.

    Args:
        col: pandas Series containing is_primary_data values

    Returns:
        pd.Series: Boolean series with consistent True/False values
    """
    if col.dtype == bool or pd.api.types.is_bool_dtype(col):
        return col.fillna(False).astype(bool)
    s = col.astype(str).str.strip().str.lower()
    mapped = s.map(
        {"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False}
    )
    return mapped.fillna(False).astype(bool)


def make_output_path(src_path: Path, root_dir: Path, out_base: Path) -> Path:
    """Generate output path based on directory structure rules.

    Args:
        src_path: Source file path
        root_dir: Root directory path
        out_base: Output base directory path

    Returns:
        Path: Generated output file path
    """
    rel = src_path.relative_to(root_dir)
    if len(rel.parts) > 1:
        # Has subdirectory: keep full relative subpath
        return out_base / rel
    else:
        # Directly under root: don't create subdirectory
        return out_base / rel.name


def process_one(
    h5ad_path_str: str, root_dir_str: str, out_base_str: str, overwrite: bool = True
) -> tuple[str, bool, str]:
    """Process a single H5AD file by filtering primary data.

    Reads the .h5ad file, filters rows where is_primary_data == True
    (including the expression matrix), and saves to target path.
    If no primary data is found, the file is skipped.

    Args:
        h5ad_path_str: Path to input H5AD file
        root_dir_str: Root directory path for relative path calculation
        out_base_str: Output base directory path
        overwrite: Whether to overwrite existing files (default: True)

    Returns:
        tuple: (file_path, success_flag, message, cell_count)
    """
    h5ad_path = Path(h5ad_path_str)
    root_dir = Path(root_dir_str)
    out_base = Path(out_base_str)

    try:
        out_path = make_output_path(h5ad_path, root_dir, out_base)
        if out_path.exists() and not overwrite:
            return (str(h5ad_path), True, f"Skip (exists): {out_path}", 0)

        # Read (modifying matrix, so don't use backed='r')
        adata = ad.read_h5ad(str(h5ad_path))

        if "is_primary_data" not in adata.obs.columns:
            # No such column: save as-is (keep structure consistent)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            adata.write(str(out_path))
            return (
                str(h5ad_path),
                True,
                "No 'is_primary_data' column; copied",
                adata.n_obs,
            )

        col_bool = to_bool_series(adata.obs["is_primary_data"])
        keep_mask = col_bool
        keep_count = int(keep_mask.sum())

        # If no primary data, skip saving
        if keep_count == 0:
            return (
                str(h5ad_path),
                True,
                "Skipped (no primary data)",
                0,
            )

        # Filter (synchronously applies to obs/X/raw etc)
        adata_f = adata[keep_mask].copy()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        adata_f.write(str(out_path))
        return (
            str(h5ad_path),
            True,
            f"Saved {keep_count} cells -> {out_path}",
            keep_count,
        )

    except Exception as e:
        return (str(h5ad_path), False, f"Error: {e}", 0)


def main():
    """Main function to process H5AD files with multiprocessing.

    Parses command line arguments and processes all H5AD files in the
    specified directory using multiple worker processes.
    """
    parser = argparse.ArgumentParser(description="Filter primary data from H5AD files")
    parser.add_argument(
        "--num_workers", type=int, default=32, help="Number of worker processes"
    )
    parser.add_argument(
        "--cellxgene_dir", required=True, help="Root directory containing H5AD files"
    )
    parser.add_argument("--output_base", required=True, help="Output base directory")
    args = parser.parse_args()

    root_dir = Path(args.cellxgene_dir)
    output_base = Path(args.output_base)
    num_workers = max(args.num_workers, 1)
    OVERWRITE = True  # Whether to overwrite if target file exists

    files = list(root_dir.rglob("*.h5ad"))
    if not files:
        print(f"[Info] No .h5ad files found under {root_dir}")
        return

    print(f"[Info] Found {len(files)} .h5ad files. Using {num_workers} processes.")
    success, fail = 0, 0
    total_rows = 0  # New total row count
    # Can reduce internal subprocess BLAS thread count，avoid over-contention（optional）
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futs = [
            ex.submit(
                process_one,
                str(p),
                str(root_dir),
                str(output_base),
                OVERWRITE,
            )
            for p in files
        ]
        for fut in as_completed(futs):
            path, ok, msg, keep_count = fut.result()
            total_rows += keep_count
            if ok:
                success += 1
                print(f"[OK] {path} | {msg}")
            else:
                fail += 1
                print(f"[FAIL] {path} | {msg}")

    print(f"\n[Done] OK: {success}, FAIL: {fail}, Total: {len(files)}")
    print(f"[Total saved rows] {total_rows}")
    print(f"[Output Base] {output_base}")


if __name__ == "__main__":
    main()
