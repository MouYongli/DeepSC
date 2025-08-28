#!/usr/bin/env python3

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import anndata as ad
import pandas as pd

import argparse


def to_bool_series(col: pd.Series) -> pd.Series:
    """把 is_primary_data 列统一转换为严格的布尔 Series。缺失/未知视为 False。"""
    if col.dtype == bool or pd.api.types.is_bool_dtype(col):
        return col.fillna(False).astype(bool)
    s = col.astype(str).str.strip().str.lower()
    mapped = s.map(
        {"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False}
    )
    return mapped.fillna(False).astype(bool)


def make_output_path(src_path: Path, root_dir: Path, out_base: Path) -> Path:
    """根据规则生成输出路径。"""
    rel = src_path.relative_to(root_dir)
    if len(rel.parts) > 1:
        # 有子目录：保持完整相对子路径
        return out_base / rel
    else:
        # 直接在根下：不新建子目录
        return out_base / rel.name


def process_one(
    h5ad_path_str: str, root_dir_str: str, out_base_str: str, overwrite: bool = True
) -> tuple[str, bool, str]:
    """
    处理单个文件：
    - 读取 .h5ad
    - 过滤 is_primary_data == True 的行（包括矩阵）
    - 写出到目标路径
    返回 (路径, 成功/失败, 信息)
    """
    h5ad_path = Path(h5ad_path_str)
    root_dir = Path(root_dir_str)
    out_base = Path(out_base_str)

    try:
        out_path = make_output_path(h5ad_path, root_dir, out_base)
        if out_path.exists() and not overwrite:
            return (str(h5ad_path), True, f"Skip (exists): {out_path}", 0)

        # 读取（修改矩阵，故不使用 backed='r'）
        adata = ad.read_h5ad(str(h5ad_path))

        if "is_primary_data" not in adata.obs.columns:
            # 没有该列：直接原样另存（保持结构一致）
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

        # 过滤（同步作用于 obs/X/raw 等）
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
    parser = argparse.ArgumentParser(description="Filter primary data from H5AD files")
    parser.add_argument(
        "--NUM_WORKERS", type=int, default=32, help="Number of worker processes"
    )
    parser.add_argument(
        "--CELLXGENE_DIR", required=True, help="Root directory containing H5AD files"
    )
    parser.add_argument("--OUTPUT_BASE", required=True, help="Output base directory")
    args = parser.parse_args()

    ROOT_DIR = Path(args.CELLXGENE_DIR)
    OUTPUT_BASE = Path(args.OUTPUT_BASE)
    NUM_WORKERS = max(args.NUM_WORKERS, 1)
    OVERWRITE = True  # 如目标文件已存在，是否覆盖

    files = list(ROOT_DIR.rglob("*.h5ad"))
    if not files:
        print(f"[Info] 在 {ROOT_DIR} 下未找到 .h5ad 文件")
        return

    print(f"[Info] Found {len(files)} .h5ad files. Using {NUM_WORKERS} processes.")
    success, fail = 0, 0
    total_rows = 0  # 新增总行数计数
    # 可降低子进程内部 BLAS 线程数，避免过度争用（可选）
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as ex:
        futs = [
            ex.submit(
                process_one,
                str(p),
                str(ROOT_DIR),
                str(OUTPUT_BASE),
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
    print(f"[Output Base] {OUTPUT_BASE}")


if __name__ == "__main__":
    main()
