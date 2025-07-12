import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix, issparse, save_npz

from .config import GENE_MAP_PATH


def process_h5ad_to_sparse_tensor(
    h5ad_path: str, output_path: str, gene_map_path: str = GENE_MAP_PATH
) -> dict:
    gene_map_df = pd.read_csv(gene_map_path)
    gene_map_df["id"] = gene_map_df["id"].astype(int)
    gene_map = dict(zip(gene_map_df["feature_name"], gene_map_df["id"]))
    max_gene_id = gene_map_df["id"].max()

    adata = sc.read_h5ad(h5ad_path)
    feature_names = adata.var["feature_name"].values
    X = adata.X.tocsr() if issparse(adata.X) else csr_matrix(adata.X)

    # 只保留有映射的基因列
    valid_mask = np.array([f in gene_map for f in feature_names])
    valid_feature_names = feature_names[valid_mask]
    valid_gene_ids = np.array([gene_map[f] for f in valid_feature_names])

    X_valid = X[:, valid_mask]

    # 按 gene_id 排序
    sort_idx = np.argsort(valid_gene_ids)
    X_valid_sorted = X_valid[:, sort_idx]
    valid_gene_ids_sorted = valid_gene_ids[sort_idx]

    n_cells = X.shape[0]
    n_genes = max_gene_id + 1

    # 用三元组(row, col, data)构造目标稀疏矩阵
    X_valid_sorted = X_valid_sorted.tocoo()
    rows = X_valid_sorted.row
    cols = valid_gene_ids_sorted[X_valid_sorted.col]
    data = X_valid_sorted.data
    X_final = csr_matrix((data, (rows, cols)), shape=(n_cells, n_genes), dtype=X.dtype)

    save_npz(output_path, X_final)
    return {"status": "saved", "path": output_path}
