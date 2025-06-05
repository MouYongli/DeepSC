from torch.utils.data import Dataset
import torch
import random

class SCDataset(Dataset):
    def __init__(self, data, class_cap=6, pad_token=0, device="cuda"):
        super().__init__()
        self.data = data
        self.class_cap = class_cap
        self.pad_token = pad_token
        self.device = device

    def __getitem__(self, index):
        rand_start = random.randint(0, self.data.shape[0] - 1)
        full_seq = self.data[rand_start].toarray()[0]
        full_seq[full_seq > self.class_cap] = self.class_cap
        full_seq = torch.from_numpy(full_seq).long()
        full_seq = torch.cat((full_seq, torch.tensor([self.pad_token]))).to(self.device)
        return full_seq

    def __len__(self):
        return self.data.shape[0]
    

def extract_rows_from_sparse_tensor(tensor, row_ids):
    tensor = tensor.coalesce()
    idx = tensor.indices()
    val = tensor.values()
    
    # 找出属于目标 row 的位置
    mask = torch.isin(idx[0], torch.tensor(row_ids, device=idx.device))
    new_indices = idx[:, mask]
    new_values = val[mask]
    
    row_id_map = {orig: i for i, orig in enumerate(row_ids)}
    remapped_rows = torch.tensor([row_id_map[int(r)] for r in new_indices[0].tolist()], device=idx.device)
    new_indices[0] = remapped_rows

    return torch.sparse_coo_tensor(new_indices, new_values, size=(len(row_ids), tensor.shape[1]))
