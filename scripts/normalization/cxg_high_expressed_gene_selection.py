import os

import anndata as ad
import numpy as np

import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process H5AD files for a specific organ."
    )
    parser.add_argument(
        "--organ", type=str, required=True, help="Organ name, e.g., blood, heart, brain"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root_dir = f"/hpcwork/rn260358/Data/cellxgene/{args.organ}"
    h5ad_files = []
    for file in os.listdir(root_dir):
        if file.endswith(".h5ad"):
            h5ad_files.append(os.path.join(root_dir, file))

    genes_50 = set()

    for file_path in h5ad_files:
        print(f"正在处理文件: {file_path}")
        try:
            adata = ad.read_h5ad(file_path)
            matrix = (
                adata.X.toarray() if not isinstance(adata.X, np.ndarray) else adata.X
            )
            gene_indices = np.where((matrix > 999).any(axis=0))[0]
            if "feature_name" in adata.var.columns:
                gene_names = adata.var["feature_name"].iloc[gene_indices]
                genes_50.update(gene_names)
        except Exception as e:
            print(f"处理失败 {file_path}: {e}")

    with open(f"/hpcwork/rn260358/Data/{args.organ}_genes_expr_gt_50.txt", "w") as f:
        for name in sorted([g for g in genes_50 if isinstance(g, str)]):
            f.write(f"{name}\n")
