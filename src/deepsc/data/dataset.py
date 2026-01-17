import torch
from anndata import AnnData
from scipy import sparse
from torch.utils.data import Dataset


class GeneExpressionDataset(Dataset):
    def __init__(self, npz_path=None, csr_matrix=None):
        if csr_matrix is not None:
            self.csr_matrix = csr_matrix
        elif npz_path is not None:
            self.csr_matrix = sparse.load_npz(npz_path)
        else:
            raise ValueError("Either npz_path or csr_matrix must be provided")
        self.num_samples = self.csr_matrix.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        row_slice = self.csr_matrix[idx]
        row_coo = row_slice.tocoo()
        gene_indices = torch.from_numpy(row_coo.col).long()
        expression_values = torch.from_numpy(row_coo.data).float()
        return {"genes": gene_indices, "expressions": expression_values}


import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp


class GeneExpressionDatasetMapped(Dataset):
    """
    Read expression matrix from h5ad; use CSV (with feature_name, id) to map gene names in var to ids,
    keep only genes that can be mapped in CSV, and reorder by id in ascending order.
    __getitem__ returns:
        {
            "genes": LongTensor[col_indices],
            "expressions": FloatTensor[values],
            "cell_type_id": LongTensor single element (if obs_celltype_col provided)
        }
    Can also return cell-type id for each cell (from adata.obs).
    """

    def __init__(
        self,
        h5ad: AnnData = None,
        h5ad_path: str = "",
        csv_path: str = "",
        var_name_col: str = "feature_name",  # Column name for gene names in h5ad var (e.g. "feature_name" / "gene_name")
        layer: str = None,  # If you want to use a specific layer, pass its name; otherwise use X
        # obs_celltype_col: str = "Factor Value[inferred cell type - authors labels]",
        obs_celltype_col: str = "cell_type",  # Cell type column name in h5ad.obs
    ):
        # 1) Read CSV, build feature_name -> id mapping
        df_map = pd.read_csv(csv_path)
        if not {"feature_name", "id"}.issubset(df_map.columns):
            raise ValueError("CSV must contain columns: feature_name, id")
        # Deduplicate: if same feature_name has multiple rows, keep first row
        df_map = df_map.dropna(subset=["feature_name", "id"]).drop_duplicates(
            subset=["feature_name"]
        )
        # Ensure id is integer (convert string numbers to int)
        df_map["id"] = df_map["id"].astype(int)
        name2id = dict(zip(df_map["feature_name"].astype(str), df_map["id"].tolist()))

        # 2) Read h5ad
        if h5ad_path != "":
            adata = sc.read_h5ad(
                h5ad_path
            )  # For large files can use backed="r", but column trimming is limited; here read directly for simplicity
        else:
            adata = h5ad
        # If var_name_col doesn't exist, use index
        if var_name_col is None or var_name_col not in adata.var.columns:
            print(
                f"⚠️ var_name_col='{var_name_col}' does not exist, using adata.var.index as gene names"
            )
            var_names = adata.var.index.astype(str).values
        else:
            var_names = adata.var[var_name_col].astype(str).values

        # 2.1) Read cell type column from obs (for __getitem__ return)
        if obs_celltype_col not in adata.obs.columns:
            raise ValueError(f"h5ad.obs does not contain column: {obs_celltype_col}")
        celltype_cat = adata.obs[obs_celltype_col].astype("category")
        # Stable id mapping: 0..(n-1)
        self.celltype_ids = celltype_cat.cat.codes.to_numpy(dtype=np.int64)
        self.celltype_categories = list(celltype_cat.cat.categories)
        # Provide id->name dict for external queries
        self.id2type = {i: name for i, name in enumerate(self.celltype_categories)}

        # 3) Find genes that exist in both h5ad.var and CSV, collect their (id, original column index)
        matched = []
        for j, nm in enumerate(var_names):
            _id = name2id.get(nm, None)
            if _id is not None:
                matched.append((_id, j))
        # If no genes matched, raise exception
        if len(matched) == 0:
            raise ValueError(
                "No genes found matching feature_name -> id mapping in CSV."
            )

        # 4) Sort by id in ascending order and record new column order
        matched.sort(key=lambda x: x[0])  # [(id, col_j), ...] sort by id
        self.sorted_ids = np.array([t[0] for t in matched], dtype=np.int64)
        cols_sorted = np.array([t[1] for t in matched], dtype=np.int64)

        # 5) Extract expression matrix (X or layer), convert to CSR, and reorder + subset columns
        X = adata.layers[layer] if layer is not None else adata.X
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        else:
            X = X.tocsr()

        # Keep only matched columns, sorted by id in ascending order
        self.csr_matrix = X[:, cols_sorted]
        self.num_samples = self.csr_matrix.shape[0]

        # optional: save mapping from column position -> gene id for debugging/tracing
        # The kth column of the new matrix corresponds to gene id self.sorted_ids[k]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        row_slice = self.csr_matrix[idx]
        row_coo = row_slice.tocoo()
        col_pos = row_coo.col.astype(np.int64)  # submatrix column position j
        gene_ids_np = self.sorted_ids[col_pos]  # map back to real id
        gene_indices = torch.from_numpy(gene_ids_np).long()  # ← return id
        expression_values = torch.from_numpy(row_coo.data.astype(np.float32))
        cell_type_id = int(self.celltype_ids[idx])
        return {
            "genes": gene_indices,  # now using id from CSV
            "expressions": expression_values.float(),
            "cell_type_id": torch.tensor(cell_type_id, dtype=torch.long),
        }


def create_global_celltype_mapping(*h5ad_paths, obs_celltype_col="cell_type"):
    """
    Collect all celltypes from multiple h5ad files and create a unified celltype to ID mapping

    Args:
        *h5ad_paths: Multiple h5ad file paths or AnnData objects
        obs_celltype_col: Name of celltype column in obs

    Returns:
        dict: Mapping from celltype name to ID {celltype_name: id}
        dict: Mapping from ID to celltype name {id: celltype_name}
    """
    import scanpy as sc

    all_celltypes = set()

    # Collect celltypes from all h5ad files
    for h5ad_path in h5ad_paths:
        if isinstance(h5ad_path, str):
            adata = sc.read_h5ad(h5ad_path)
        else:
            adata = h5ad_path  # If AnnData object is passed directly

        if obs_celltype_col not in adata.obs.columns:
            raise ValueError(f"h5ad.obs does not contain column: {obs_celltype_col}")

        celltypes = adata.obs[obs_celltype_col].astype(str).unique()
        all_celltypes.update(celltypes)

    # Sort alphabetically to ensure stable mapping
    sorted_celltypes = sorted(all_celltypes)

    # Create bidirectional mapping
    type2id = {celltype: idx for idx, celltype in enumerate(sorted_celltypes)}
    id2type = {idx: celltype for idx, celltype in enumerate(sorted_celltypes)}

    print(f"Found {len(sorted_celltypes)} unique cell types:")
    for celltype, idx in type2id.items():
        print(f"  {idx}: {celltype}")

    return type2id, id2type


class GeneExpressionDatasetMappedWithGlobalCelltype(Dataset):
    """
    Improved GeneExpressionDatasetMapped that supports using predefined celltype mappings
    Ensures training set and test set use the same celltype ID mapping
    """

    def __init__(
        self,
        h5ad: AnnData = None,
        h5ad_path: str = "",
        csv_path: str = "",
        var_name_col: str = "feature_name",
        layer: str = None,
        obs_celltype_col: str = "cell_type",
        global_type2id: dict = None,  # Predefined celltype mapping
        global_id2type: dict = None,  # Predefined ID to celltype mapping
    ):
        import numpy as np
        import pandas as pd
        import scanpy as sc
        import scipy.sparse as sp

        # 1) Read CSV, establish feature_name -> id mapping
        df_map = pd.read_csv(csv_path)
        if not {"feature_name", "id"}.issubset(df_map.columns):
            raise ValueError("CSV must contain columns: feature_name, id")
        # Deduplicate: if same feature_name has multiple rows, keep first row
        df_map = df_map.dropna(subset=["feature_name", "id"]).drop_duplicates(
            subset=["feature_name"]
        )
        # Ensure id is integer (convert string numbers to int)
        df_map["id"] = df_map["id"].astype(int)
        name2id = dict(zip(df_map["feature_name"].astype(str), df_map["id"].tolist()))

        # 2) Read h5ad
        if h5ad_path != "":
            adata = sc.read_h5ad(h5ad_path)
        else:
            adata = h5ad
        # If var_name_col doesn't exist, use index
        if var_name_col is None or var_name_col not in adata.var.columns:
            print(
                f"⚠️ var_name_col='{var_name_col}' does not exist, using adata.var.index as gene names"
            )
            var_names = adata.var.index.astype(str).values
        else:
            var_names = adata.var[var_name_col].astype(str).values

        # 2.1) Read cell type column from obs, use global mapping
        if obs_celltype_col not in adata.obs.columns:
            raise ValueError(f"h5ad.obs does not contain column: {obs_celltype_col}")

        celltype_series = adata.obs[obs_celltype_col].astype(str)

        if global_type2id is not None and global_id2type is not None:
            # Use predefined mapping
            self.type2id = global_type2id
            self.id2type = global_id2type
            print(f"Using global celltype mapping: {self.type2id}")
            print(f"Using global celltype mapping: {self.id2type}")

            # Map current dataset's celltypes to unified IDs
            self.celltype_ids = np.array(
                [self.type2id.get(ct, -1) for ct in celltype_series.values],
                dtype=np.int64,
            )

            # Check if there are any celltypes not found
            missing_types = set(celltype_series.unique()) - set(self.type2id.keys())
            if missing_types:
                raise ValueError(
                    f"Found celltypes in dataset not in global mapping: {missing_types}"
                )

            # Validate label value range
            invalid_labels = self.celltype_ids == -1
            if invalid_labels.any():
                invalid_count = invalid_labels.sum()
                raise ValueError(
                    f"Found {invalid_count} invalid celltype labels (-1), indicating unmapped celltypes exist"
                )

            max_label = self.celltype_ids.max()
            min_label = self.celltype_ids.min()
            num_classes = len(self.type2id)

            print(
                f"Label range validation: min={min_label}, max={max_label}, num_classes={num_classes}"
            )

            if max_label >= num_classes or min_label < 0:
                raise ValueError(
                    f"Label value range [{min_label}, {max_label}] exceeds valid class range [0, {num_classes-1}]"
                )

            self.celltype_categories = [
                self.id2type[i] for i in sorted(self.id2type.keys())
            ]

        else:
            # Original local mapping method (for backward compatibility)
            celltype_cat = celltype_series.astype("category")
            self.celltype_ids = celltype_cat.cat.codes.to_numpy(dtype=np.int64)
            self.celltype_categories = list(celltype_cat.cat.categories)
            self.type2id = {name: i for i, name in enumerate(self.celltype_categories)}
            self.id2type = {i: name for i, name in enumerate(self.celltype_categories)}

        # 3) Find genes that exist in both h5ad.var and CSV, collect their (id, original column index)
        matched = []
        for j, nm in enumerate(var_names):
            _id = name2id.get(nm, None)
            if _id is not None:
                matched.append((_id, j))
        # If no genes matched, raise exception
        if len(matched) == 0:
            raise ValueError(
                "No genes found matching feature_name -> id mapping in CSV."
            )

        # 4) Sort by id in ascending order and record new column order
        matched.sort(key=lambda x: x[0])  # [(id, col_j), ...] sort by id
        self.sorted_ids = np.array([t[0] for t in matched], dtype=np.int64)
        cols_sorted = np.array([t[1] for t in matched], dtype=np.int64)

        # 5) Extract expression matrix (X or layer), convert to CSR, and reorder + subset columns
        X = adata.layers[layer] if layer is not None else adata.X
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        else:
            X = X.tocsr()

        # Keep only matched columns, sorted by id in ascending order
        self.csr_matrix = X[:, cols_sorted]
        self.num_samples = self.csr_matrix.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        row_slice = self.csr_matrix[idx]
        row_coo = row_slice.tocoo()
        col_pos = row_coo.col.astype(np.int64)  # submatrix column position j
        gene_ids_np = self.sorted_ids[col_pos]  # map back to real id
        gene_indices = torch.from_numpy(gene_ids_np).long()  # ← return id
        expression_values = torch.from_numpy(row_coo.data.astype(np.float32))
        cell_type_id = int(self.celltype_ids[idx])
        return {
            "genes": gene_indices,  # now using id from CSV
            "expressions": expression_values.float(),
            "cell_type_id": torch.tensor(cell_type_id, dtype=torch.long),
        }
