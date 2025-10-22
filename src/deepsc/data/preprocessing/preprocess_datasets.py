import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse, save_npz


def process_h5ad_to_sparse_tensor(
    h5ad_path: str,
    output_path: str,
    gene_map_path: str,
) -> dict:
    """Convert h5ad file to sparse tensor format with gene ID mapping.

    Processes an h5ad single-cell dataset file by mapping gene names to
    standardized gene IDs and converting the expression matrix to a
    sparse tensor format. Only genes present in the gene mapping are
    retained, and the output matrix is sorted by gene ID for consistency.

    Args:
        h5ad_path (str): Path to the input h5ad file containing single-cell data.
        output_path (str): Path where the processed sparse matrix (.npz)
            will be saved.
        gene_map_path (str, optional): Path to the gene mapping CSV file
            containing feature_name to gene ID mappings.
    Returns:
        dict: Status dictionary with keys:
            - "status": Always "saved" upon successful completion
            - "path": Path to the saved output file
    """
    gene_map_df = pd.read_csv(gene_map_path)
    gene_map_df["id"] = gene_map_df["id"].astype(int)
    gene_map = dict(zip(gene_map_df["feature_name"], gene_map_df["id"]))
    max_gene_id = gene_map_df["id"].max()

    adata = ad.read_h5ad(h5ad_path)
    feature_names = adata.var["feature_name"].values
    X = adata.X.tocsr() if issparse(adata.X) else csr_matrix(adata.X)

    # Keep only gene columns that have mappings
    valid_mask = np.array([f in gene_map for f in feature_names])
    valid_feature_names = feature_names[valid_mask]
    valid_gene_ids = np.array([gene_map[f] for f in valid_feature_names])

    X_valid = X[:, valid_mask]

    # Sort by gene_id
    sort_idx = np.argsort(valid_gene_ids)
    X_valid_sorted = X_valid[:, sort_idx]
    valid_gene_ids_sorted = valid_gene_ids[sort_idx]

    n_cells = X.shape[0]
    n_genes = max_gene_id + 1

    # Construct target sparse matrix using triplets (row, col, data)
    X_valid_sorted = X_valid_sorted.tocoo()
    rows = X_valid_sorted.row
    cols = valid_gene_ids_sorted[X_valid_sorted.col]
    data = X_valid_sorted.data
    X_final = csr_matrix((data, (rows, cols)), shape=(n_cells, n_genes), dtype=X.dtype)

    save_npz(output_path, X_final)
    return {"status": "saved", "path": output_path}
