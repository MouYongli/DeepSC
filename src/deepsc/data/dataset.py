import torch
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
