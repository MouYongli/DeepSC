import logging

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy.sparse import issparse

from .config import GENE_MAP_PATH


def process_h5ad_to_sparse_tensor(
    h5ad_path: str, output_path: str, gene_map_path: str = GENE_MAP_PATH
) -> dict:
    gene_map_df = pd.read_csv(gene_map_path)
    gene_map_df["id"] = gene_map_df["id"].astype(int)
    gene_map = dict(zip(gene_map_df["feature_name"], gene_map_df["id"]))

    adata = sc.read_h5ad(h5ad_path)
    feature_names = adata.var["feature_name"]
    X = adata.X.tocsr() if issparse(adata.X) else adata.X

    row_indices = []
    col_indices = []
    values = []

    for i in range(adata.n_obs):
        row = X.getrow(i).tocoo()
        nonzero_indices = row.col
        nonzero_values = row.data

        expressed_feature_names = feature_names.iloc[nonzero_indices].values
        ids = pd.Series(expressed_feature_names).map(gene_map).values

        valid_mask = pd.notna(ids)
        valid_ids = ids[valid_mask].astype(int)
        valid_expr_values = nonzero_values[valid_mask].astype(int)

        sorted_idx = np.argsort(valid_ids)
        sorted_ids = valid_ids[sorted_idx]
        sorted_expr_values = valid_expr_values[sorted_idx]

        row_indices.extend([i] * len(sorted_ids))
        col_indices.extend(sorted_ids)
        values.extend(sorted_expr_values.tolist())
        if i % 1000 == 0 or i == adata.n_obs - 1:
            progress = (i + 1) / adata.n_obs * 100
            logging.info(f"Progress: {progress:.2f}% ({i + 1}/{adata.n_obs} cells)")
            print(f"Progress: {progress:.2f}% ({i + 1}/{adata.n_obs} cells)")
    coo_tensor = torch.sparse_coo_tensor(
        indices=np.vstack((row_indices, col_indices)),
        values=values,
        size=(adata.n_obs, max(col_indices) + 1),
    ).coalesce()

    torch.save(coo_tensor, output_path)
    return {"status": "saved", "path": output_path}
