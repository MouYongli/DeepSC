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
