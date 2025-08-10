import os

import numpy as np
from scipy import sparse

import argparse

# Modified normalize_tensor: no 1e4 scaling


def normalize_tensor_no_scale(csr):
    valid_cells = np.diff(csr.indptr) >= 200
    csr = csr[valid_cells]
    print(f"Valid cells: {valid_cells.sum()}")
    csr.data = np.log1p(csr.data)
    print(f"Normalized data: {csr.data}")
    return csr


def process_file(filepath, out_dir):
    filename = os.path.basename(filepath)
    name, ext = os.path.splitext(filename)
    if ext == ".npz":
        csr = sparse.load_npz(filepath)
        print(f"Loaded {filepath}")
    elif ext == ".mtx":
        csr = sparse.csr_matrix(sparse.mmread(filepath))
    else:
        print(f"Skipping unsupported file: {filename}")
        return
    norm_csr = normalize_tensor_no_scale(csr)
    out_path = os.path.join(out_dir, name + "_norm.npz")
    sparse.save_npz(out_path, norm_csr)
    print(f"Saved normalized: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch normalize all sparse matrices in a folder."
    )
    parser.add_argument(
        "--input_dir", type=str, help="Input folder containing .npz or .mtx files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output folder (default: input_dir)",
    )
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        fpath = os.path.join(input_dir, fname)
        if os.path.isfile(fpath):
            process_file(fpath, output_dir)


if __name__ == "__main__":
    main()
