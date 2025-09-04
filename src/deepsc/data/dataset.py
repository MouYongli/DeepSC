import torch
from anndata import AnnData
from scipy import sparse
from torch.utils.data import Dataset


class SCDataset(Dataset):
    def __init__(self, coo_tensor, num_bin=5):
        self.coo_tensor = coo_tensor.coalesce()
        self.num_samples = coo_tensor.shape[0]
        self.num_bin = num_bin

    def __getitem__(self, idx):
        # TODO:  check with yongli and jin er 这里需要确认一下，是否需要随机采样
        # rand_idx = random.randint(0, self.num_samples - 1)
        row = self.coo_tensor[idx].to_dense()
        row[row > self.num_bin] = self.num_bin
        row = row.long()
        # TODO: 这里去除了to device, 似乎用fabric的dataloader setup可以自动转移到device上面，还需要确认一下
        # TODO: 和yongli确认，如果在这里激normalization的话比较低效，会造成重复计算
        # TODO: 研究是否可以把datamask放到这里面？
        row = torch.cat((row, torch.tensor([0], dtype=torch.long, device=row.device)))
        return row

    def __len__(self):
        return self.num_samples


class GeneExpressionDataset(Dataset):
    def __init__(self, coo_tensor, num_bin=50):
        assert (
            coo_tensor.layout == torch.sparse_coo
        ), "Input must be a sparse COO tensor"
        coo_tensor = coo_tensor.coalesce()

        values = coo_tensor.values()
        indices = coo_tensor.indices()
        self.csr_matrix = sparse.csr_matrix(
            (
                values.cpu().numpy(),
                (indices[0].cpu().numpy(), indices[1].cpu().numpy()),
            ),
            shape=coo_tensor.shape,
        )

        self.num_samples = coo_tensor.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        row_slice = self.csr_matrix[idx]
        row_coo = row_slice.tocoo()

        gene_indices = torch.from_numpy(row_coo.col).long()
        expression_values = torch.from_numpy(row_coo.data).float()

        return {"genes": gene_indices, "expressions": expression_values}


class GeneExpressionDatasetNew(Dataset):
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
    从 h5ad 中读取表达矩阵；用 CSV(含 feature_name, id) 将 var 里的基因名映射为 id，
    只保留能在 CSV 里找到映射的基因，并按 id 升序重排列。
    __getitem__ 返回:
        {
            "genes": LongTensor[col_indices],
            "expressions": FloatTensor[values],
            "cell_type_id": LongTensor 单个元素 (若 obs_celltype_col 提供)
        }
    也可以返回每个细胞的 cell-type id（取自 adata.obs）。
    """

    def __init__(
        self,
        h5ad: AnnData = None,
        h5ad_path: str = "",
        csv_path: str = "",
        var_name_col: str = "feature_name",  # h5ad的var里对应基因名的列名（如 "feature_name" / "gene_name"）
        layer: str = None,  # 如果想用某个layer，传入其名称；否则用X
        # obs_celltype_col: str = "Factor Value[inferred cell type - authors labels]",
        obs_celltype_col: str = "cell_type",  # h5ad.obs中细胞类型列名
    ):
        # 1) 读CSV，建立 feature_name -> id 映射
        df_map = pd.read_csv(csv_path)
        if not {"feature_name", "id"}.issubset(df_map.columns):
            raise ValueError("CSV 必须包含列: feature_name, id")
        # 去重：若同一feature_name多行，保留第一行
        df_map = df_map.dropna(subset=["feature_name", "id"]).drop_duplicates(
            subset=["feature_name"]
        )
        # 确保 id 为整数（若是字符串数字也转为int）
        df_map["id"] = df_map["id"].astype(int)
        name2id = dict(zip(df_map["feature_name"].astype(str), df_map["id"].tolist()))

        # 2) 读 h5ad
        if h5ad_path != "":
            adata = sc.read_h5ad(
                h5ad_path
            )  # 大文件可改为 backed="r"，但列裁剪会受限，这里为简单起见直接读入
        else:
            adata = h5ad
        # 如果没有 var_name_col，就用 index
        if var_name_col is None or var_name_col not in adata.var.columns:
            print(
                f"⚠️ var_name_col='{var_name_col}' 不存在，使用 adata.var.index 作为基因名"
            )
            var_names = adata.var.index.astype(str).values
        else:
            var_names = adata.var[var_name_col].astype(str).values

        # 2.1) 读取 obs 中的细胞类型列（用于 __getitem__ 返回）
        if obs_celltype_col not in adata.obs.columns:
            raise ValueError(f"h5ad.obs 不包含列: {obs_celltype_col}")
        celltype_cat = adata.obs[obs_celltype_col].astype("category")
        # 稳定的 id 映射：0..(n-1)
        self.celltype_ids = celltype_cat.cat.codes.to_numpy(dtype=np.int64)
        self.celltype_categories = list(celltype_cat.cat.categories)
        # 提供一个 id->name 的字典，便于外部查询
        self.id2type = {i: name for i, name in enumerate(self.celltype_categories)}

        # 3) 找到既在 h5ad.var 又在 CSV 的基因列，收集其(id, 原列索引)
        matched = []
        for j, nm in enumerate(var_names):
            _id = name2id.get(nm, None)
            if _id is not None:
                matched.append((_id, j))
        # 如果没有匹配到任何基因，抛出异常
        if len(matched) == 0:
            raise ValueError(
                "没有任何基因在 CSV 中找到匹配的 feature_name -> id 映射。"
            )

        # 4) 按 id 升序排序并记录新列顺序
        matched.sort(key=lambda x: x[0])  # [(id, col_j), ...] 按 id 排序
        self.sorted_ids = np.array([t[0] for t in matched], dtype=np.int64)
        cols_sorted = np.array([t[1] for t in matched], dtype=np.int64)

        # 5) 取表达矩阵（X或layer），转CSR，并按列重排 + 子集
        X = adata.layers[layer] if layer is not None else adata.X
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        else:
            X = X.tocsr()

        # 只保留匹配到的列，并按 id 升序
        self.csr_matrix = X[:, cols_sorted]
        self.num_samples = self.csr_matrix.shape[0]

        # 可选：保存从列位置 -> 基因id 的映射，便于调试/回溯
        # 新矩阵的第k列对应的基因id为 self.sorted_ids[k]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        row_slice = self.csr_matrix[idx]
        row_coo = row_slice.tocoo()
        col_pos = row_coo.col.astype(np.int64)  # 子矩阵列位置 j
        gene_ids_np = self.sorted_ids[col_pos]  # 映射回真实 id
        gene_indices = torch.from_numpy(gene_ids_np).long()  # ← 返回 id
        expression_values = torch.from_numpy(row_coo.data.astype(np.float32))
        cell_type_id = int(self.celltype_ids[idx])
        return {
            "genes": gene_indices,  # 现在是 CSV 中的 id
            "expressions": expression_values.float(),
            "cell_type_id": torch.tensor(cell_type_id, dtype=torch.long),
        }


def create_global_celltype_mapping(*h5ad_paths, obs_celltype_col="cell_type"):
    """
    从多个h5ad文件中收集所有的celltype，创建统一的celltype到ID的映射表

    Args:
        *h5ad_paths: 多个h5ad文件路径或AnnData对象
        obs_celltype_col: obs中celltype列的名称

    Returns:
        dict: celltype名称到ID的映射 {celltype_name: id}
        dict: ID到celltype名称的映射 {id: celltype_name}
    """
    import scanpy as sc

    all_celltypes = set()

    # 从所有h5ad文件中收集celltype
    for h5ad_path in h5ad_paths:
        if isinstance(h5ad_path, str):
            adata = sc.read_h5ad(h5ad_path)
        else:
            adata = h5ad_path  # 如果直接传入AnnData对象

        if obs_celltype_col not in adata.obs.columns:
            raise ValueError(f"h5ad.obs 不包含列: {obs_celltype_col}")

        celltypes = adata.obs[obs_celltype_col].astype(str).unique()
        all_celltypes.update(celltypes)

    # 按字母顺序排序以确保稳定的映射
    sorted_celltypes = sorted(all_celltypes)

    # 创建双向映射
    type2id = {celltype: idx for idx, celltype in enumerate(sorted_celltypes)}
    id2type = {idx: celltype for idx, celltype in enumerate(sorted_celltypes)}

    print(f"Found {len(sorted_celltypes)} unique cell types:")
    for celltype, idx in type2id.items():
        print(f"  {idx}: {celltype}")

    return type2id, id2type


class GeneExpressionDatasetMappedWithGlobalCelltype(Dataset):
    """
    改进的GeneExpressionDatasetMapped，支持使用预定义的celltype映射表
    确保训练集和测试集使用相同的celltype ID映射
    """

    def __init__(
        self,
        h5ad: AnnData = None,
        h5ad_path: str = "",
        csv_path: str = "",
        var_name_col: str = "feature_name",
        layer: str = None,
        obs_celltype_col: str = "cell_type",
        global_type2id: dict = None,  # 预定义的celltype映射
        global_id2type: dict = None,  # 预定义的ID到celltype映射
    ):
        import numpy as np
        import pandas as pd
        import scanpy as sc
        import scipy.sparse as sp

        # 1) 读CSV，建立 feature_name -> id 映射
        df_map = pd.read_csv(csv_path)
        if not {"feature_name", "id"}.issubset(df_map.columns):
            raise ValueError("CSV 必须包含列: feature_name, id")
        # 去重：若同一feature_name多行，保留第一行
        df_map = df_map.dropna(subset=["feature_name", "id"]).drop_duplicates(
            subset=["feature_name"]
        )
        # 确保 id 为整数（若是字符串数字也转为int）
        df_map["id"] = df_map["id"].astype(int)
        name2id = dict(zip(df_map["feature_name"].astype(str), df_map["id"].tolist()))

        # 2) 读 h5ad
        if h5ad_path != "":
            adata = sc.read_h5ad(h5ad_path)
        else:
            adata = h5ad
        # 如果没有 var_name_col，就用 index
        if var_name_col is None or var_name_col not in adata.var.columns:
            print(
                f"⚠️ var_name_col='{var_name_col}' 不存在，使用 adata.var.index 作为基因名"
            )
            var_names = adata.var.index.astype(str).values
        else:
            var_names = adata.var[var_name_col].astype(str).values

        # 2.1) 读取 obs 中的细胞类型列，使用全局映射表
        if obs_celltype_col not in adata.obs.columns:
            raise ValueError(f"h5ad.obs 不包含列: {obs_celltype_col}")

        celltype_series = adata.obs[obs_celltype_col].astype(str)

        if global_type2id is not None and global_id2type is not None:
            # 使用预定义的映射表
            self.type2id = global_type2id
            self.id2type = global_id2type
            print(f"Using global celltype mapping: {self.type2id}")
            print(f"Using global celltype mapping: {self.id2type}")

            # 将当前数据集的celltype映射为统一的ID
            self.celltype_ids = np.array(
                [self.type2id.get(ct, -1) for ct in celltype_series.values],
                dtype=np.int64,
            )

            # 检查是否有未找到的celltype
            missing_types = set(celltype_series.unique()) - set(self.type2id.keys())
            if missing_types:
                raise ValueError(
                    f"数据集中发现未在全局映射中的celltype: {missing_types}"
                )

            # 验证标签值的范围
            invalid_labels = self.celltype_ids == -1
            if invalid_labels.any():
                invalid_count = invalid_labels.sum()
                raise ValueError(
                    f"发现{invalid_count}个无效的celltype标签（-1），这表明存在未映射的celltype"
                )

            max_label = self.celltype_ids.max()
            min_label = self.celltype_ids.min()
            num_classes = len(self.type2id)

            print(
                f"标签范围验证: min={min_label}, max={max_label}, num_classes={num_classes}"
            )

            if max_label >= num_classes or min_label < 0:
                raise ValueError(
                    f"标签值范围[{min_label}, {max_label}]超出了有效类别范围[0, {num_classes-1}]"
                )

            self.celltype_categories = [
                self.id2type[i] for i in sorted(self.id2type.keys())
            ]

        else:
            # 原始的局部映射方式（为了向后兼容）
            celltype_cat = celltype_series.astype("category")
            self.celltype_ids = celltype_cat.cat.codes.to_numpy(dtype=np.int64)
            self.celltype_categories = list(celltype_cat.cat.categories)
            self.type2id = {name: i for i, name in enumerate(self.celltype_categories)}
            self.id2type = {i: name for i, name in enumerate(self.celltype_categories)}

        # 3) 找到既在 h5ad.var 又在 CSV 的基因列，收集其(id, 原列索引)
        matched = []
        for j, nm in enumerate(var_names):
            _id = name2id.get(nm, None)
            if _id is not None:
                matched.append((_id, j))
        # 如果没有匹配到任何基因，抛出异常
        if len(matched) == 0:
            raise ValueError(
                "没有任何基因在 CSV 中找到匹配的 feature_name -> id 映射。"
            )

        # 4) 按 id 升序排序并记录新列顺序
        matched.sort(key=lambda x: x[0])  # [(id, col_j), ...] 按 id 排序
        self.sorted_ids = np.array([t[0] for t in matched], dtype=np.int64)
        cols_sorted = np.array([t[1] for t in matched], dtype=np.int64)

        # 5) 取表达矩阵（X或layer），转CSR，并按列重排 + 子集
        X = adata.layers[layer] if layer is not None else adata.X
        if not sp.issparse(X):
            X = sp.csr_matrix(X)
        else:
            X = X.tocsr()

        # 只保留匹配到的列，并按 id 升序
        self.csr_matrix = X[:, cols_sorted]
        self.num_samples = self.csr_matrix.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        row_slice = self.csr_matrix[idx]
        row_coo = row_slice.tocoo()
        col_pos = row_coo.col.astype(np.int64)  # 子矩阵列位置 j
        gene_ids_np = self.sorted_ids[col_pos]  # 映射回真实 id
        gene_indices = torch.from_numpy(gene_ids_np).long()  # ← 返回 id
        expression_values = torch.from_numpy(row_coo.data.astype(np.float32))
        cell_type_id = int(self.celltype_ids[idx])
        return {
            "genes": gene_indices,  # 现在是 CSV 中的 id
            "expressions": expression_values.float(),
            "cell_type_id": torch.tensor(cell_type_id, dtype=torch.long),
        }
