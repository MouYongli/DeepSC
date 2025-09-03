import csv
import os
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

import argparse
import sys

# handle big csv file
csv.field_size_limit(sys.maxsize)


def load_plan(csv_file):
    """
    Returns:
      file_to_chunks = {file_path: {chunk_id: [row_indices,...]}}
      chunk_ids      = sorted(set of all chunk_id)
    """
    file_to_chunks = defaultdict(lambda: defaultdict(list))
    chunk_ids = set()
    with open(csv_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fp = row["source_file"]
            cid = int(row["chunk_id"])
            rows = list(map(int, row["rows"].split()))
            file_to_chunks[fp][cid].extend(rows)
            chunk_ids.add(cid)
    return file_to_chunks, sorted(chunk_ids)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Memory merge recommends =1, avoid large matrix cross-process serialization",
    )
    parser.add_argument(
        "--in-memory",
        action="store_true",
        help="Enable pure in-memory merge (no temporary files)",
    )
    args = parser.parse_args()
    ensure_dir(args.output_dir)

    file_to_chunks, chunk_ids = load_plan(args.csv_file)

    if not args.in_memory:
        raise SystemExit("Add in-memory to enable pure in-memory merge.")

    # ===== Pure in-memory merge: chunk_id -> [csr_part, csr_part, ...]
    chunk_parts = {cid: [] for cid in chunk_ids}

    # Process files sequentially (avoid huge IPC/copying)
    for fp in tqdm(
        list(file_to_chunks.keys()), desc="[Stage] load & slice (in-memory)"
    ):
        try:
            csr = sp.load_npz(fp).tocsr()
        except Exception as e:
            print(f"[WARN] load_npz failed: {fp}: {e}")
            continue

        chunk_to_rows = file_to_chunks[fp]
        for cid, rows in chunk_to_rows.items():
            if not rows:
                continue
            idx = np.unique(np.asarray(rows, dtype=np.int64))
            idx = idx[(idx >= 0) & (idx < csr.shape[0])]
            if idx.size == 0:
                continue
            sub = csr[idx]  # CSR 切片返回 CSR/CSR-like
            sub = sub.tocsr()  # 显式确保 CSR
            chunk_parts[cid].append(sub)

        # Release current file matrix
        del csr

    # Merge and save to disk
    for cid in tqdm(chunk_ids, desc="[Stage] merge & save"):
        parts = chunk_parts[cid]
        if not parts:
            print(f"[Chunk {cid:03d}] no parts, skip.")
            continue
        merged = sp.vstack(parts, format="csr")
        out_path = os.path.join(args.output_dir, f"shuffled_{cid:03d}.npz")
        sp.save_npz(out_path, merged)
        # Release memory
        chunk_parts[cid].clear()
        del merged

    print("Done (in-memory).")


if __name__ == "__main__":
    main()
