import os

import anndata as ad
import numpy as np
import pandas as pd

csv_file = (
    "/home/angli/DeepSC/scripts/download/3ca/updated_target_dataset_with_datasetid.csv"
)
df = pd.read_csv(csv_file)
paths = df[["path", "dataset_id"]].dropna()

genes_50 = set()

for _, row in paths.iterrows():
    file_path = os.path.join(
        row["path"], row["dataset_id"]
    )  # concatenate path from csv
    file_path = file_path + ".h5ad"
    print(f"Processing: {file_path}")
    if not os.path.exists(file_path):
        print(f"file not exist: {file_path}")
        continue

    try:
        adata = ad.read_h5ad(file_path)
        matrix = adata.X.toarray() if not isinstance(adata.X, np.ndarray) else adata.X
        gene_indices = np.where((matrix > 50).any(axis=0))[0]
        if "feature_name" in adata.var.columns:
            gene_names = adata.var["feature_name"].iloc[gene_indices]
            genes_50.update(gene_names)
    except Exception as e:
        print(f"Failed with processing {file_path}: {e}")

with open("/home/angli/DeepSC/scripts/download/3ca/genes_expr_gt_50.txt", "w") as f:
    for name in sorted([g for g in genes_50 if isinstance(g, str)]):
        f.write(f"{name}\n")
