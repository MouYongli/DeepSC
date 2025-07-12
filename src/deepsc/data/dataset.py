import numpy as np
import torch
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


# 直接将gene count 进行normalize, 避免转成anndata然后再转回去
def normalize_tensor(csr):
    """
    Normalize gene count tensor by filtering cells and applying log2 normalization.
    
    Args:
        csr: scipy.sparse CSR matrix containing gene expression data
        
    Returns:
        scipy.sparse CSR matrix: Normalized expression data
        
    Raises:
        TypeError: If input is not a sparse CSR matrix
        ValueError: If matrix has invalid dimensions or no valid cells
    """
    if not sparse.issparse(csr):
        raise TypeError("Input must be a sparse matrix")
    
    if not sparse.isspmatrix_csr(csr):
        raise TypeError("Input must be a CSR matrix")
    
    if csr.shape[0] == 0 or csr.shape[1] == 0:
        raise ValueError("Input matrix cannot have zero dimensions")
    
    # Filter cells with at least 200 genes
    valid_cells = np.diff(csr.indptr) >= 200
    
    if not np.any(valid_cells):
        raise ValueError("No cells with >= 200 genes found. Cannot normalize.")
    
    csr = csr[valid_cells]
    
    # Calculate row sums for normalization
    row_sums = np.array(csr.sum(axis=1)).flatten()
    
    if np.any(row_sums < 0):
        raise ValueError("Negative values found in expression data")
    
    # Normalize to 10,000 counts per cell
    row_scales = np.divide(
        1e4, row_sums, out=np.zeros_like(row_sums, dtype=np.float32), where=row_sums > 0
    )
    csr = csr.multiply(row_scales[:, None])
    
    # Apply log2(1 + x) transformation
    csr.data = np.log2(1 + csr.data)
    
    return csr


def preprocess_sparse_tensor(pth_data_path, save_path=None):
    coo_tensor = torch.load(pth_data_path)
    if not coo_tensor.is_coalesced():
        raise ValueError(
            "The input sparse tensor is not coalesced. Please call coalesce() first."
        )

    values = coo_tensor.values().cpu().numpy()
    indices = coo_tensor.indices().cpu().numpy()
    shape = coo_tensor.shape
    csr = sparse.coo_matrix((values, (indices[0], indices[1])), shape=shape).tocsr()

    processed_csr = normalize_tensor(csr)
    processed_coo = processed_csr.tocoo()

    torch_sparse = torch.sparse_coo_tensor(
        indices=np.vstack((processed_coo.row, processed_coo.col)),
        values=processed_coo.data,
        size=processed_coo.shape,
    ).coalesce()

    if save_path is None:
        save_path = pth_data_path.replace(".pth", "_preprocessed.pth")

    torch.save(torch_sparse, save_path)


# 用于区分训练集和测试集，
# TODO：这里耗时部较长，搞清楚如何优化
def extract_rows_from_sparse_tensor_slow(tensor, row_ids):
    tensor = tensor.coalesce()
    idx = tensor.indices()
    val = tensor.values()
    # 找出属于目标 row 的位置
    mask = torch.isin(idx[0], torch.tensor(row_ids, device=idx.device))
    new_indices = idx[:, mask]
    new_values = val[mask]
    row_id_map = {orig: i for i, orig in enumerate(row_ids)}
    remapped_rows = torch.tensor(
        [row_id_map[int(r)] for r in new_indices[0].tolist()], device=idx.device
    )
    new_indices[0] = remapped_rows
    return torch.sparse_coo_tensor(
        new_indices, new_values, size=(len(row_ids), tensor.shape[1])
    )


def extract_rows_from_sparse_tensor(tensor, row_ids):
    """
    更快的提取稀疏张量指定行的方法，利用scipy csr切片。
    tensor: torch.sparse_coo_tensor
    row_ids: list or np.ndarray of row indices
    返回: torch.sparse_coo_tensor，shape=(len(row_ids), n_cols)
    """
    tensor = tensor.coalesce()
    values = tensor.values().cpu().numpy()
    indices = tensor.indices().cpu().numpy()
    shape = tensor.shape
    # 构建scipy csr
    csr = sparse.coo_matrix((values, (indices[0], indices[1])), shape=shape).tocsr()
    # 切片
    sub_csr = csr[row_ids]
    sub_coo = sub_csr.tocoo()
    torch_sparse = torch.sparse_coo_tensor(
        indices=np.vstack((sub_coo.row, sub_coo.col)),
        values=sub_coo.data,
        size=(len(row_ids), shape[1]),
    ).coalesce()
    return torch_sparse
