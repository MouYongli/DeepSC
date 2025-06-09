import math
from functools import reduce

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


# 直接将gene count 进行normalize, 避免转成anndata然后再转回去
def normalize_tensor(csr):
    valid_cells = np.diff(csr.indptr) >= 200
    csr = csr[valid_cells]
    row_sums = np.array(csr.sum(axis=1)).flatten()
    row_scales = np.divide(
        1e4, row_sums, out=np.zeros_like(row_sums, dtype=np.float64), where=row_sums > 0
    )
    csr = csr.multiply(row_scales[:, None])
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


def data_mask(
    data,
    mask_prob=0.15,
    replace_prob=0.9,
    num_tokens=None,
    random_token_prob=0,
    mask_token_id=6,  # 应该是bin_num +1
    pad_token_id=6,  # 应该是bin_num +1 但两个一样会不会有问题
    mask_ignore_token_ids=[0],
):
    mask_ignore_token_ids = set([*mask_ignore_token_ids, pad_token_id])
    # do not mask [pad] tokens, or any other tokens in the tokens designated to be excluded ([cls], [sep])
    # also do not include these special tokens in the tokens chosen at random
    no_mask = mask_with_tokens(
        data, mask_ignore_token_ids
    )  # ignore_token as True, will not be masked later
    mask = get_mask_subset_with_prob(
        ~no_mask, mask_prob
    )  # get the True/False mask matrix
    # get mask indices
    # mask_indices = torch.nonzero(mask, as_tuple=True)   # get the index of mask(nonzero value of mask matrix)
    # mask input with mask tokens with`replace_prob` (keep tokens the same with probability 1 - replace_prob)
    masked_input = data.clone().detach()
    # if random token probability > 0 for mlm
    if random_token_prob > 0:
        assert (
            num_tokens is not None
        ), "num_tokens keyword must be supplied when instantiating MLM if using random token replacement"
        random_token_prob = prob_mask_like(
            data, random_token_prob
        )  # get the mask matrix of random token replace
        random_tokens = torch.randint(
            0, num_tokens, data.shape, device=data.device
        )  # generate random token matrix with the same shape as input
        random_no_mask = mask_with_tokens(
            random_tokens, mask_ignore_token_ids
        )  # not masked matrix for the random token matrix
        random_token_prob &= (
            ~random_no_mask
        )  # get the pure mask matrix of random token replace
        random_indices = torch.nonzero(
            random_token_prob, as_tuple=True
        )  # index of random token replace
        masked_input[random_indices] = random_tokens[
            random_indices
        ]  # replace some tokens by random token
    # [mask] input
    replace_prob = prob_mask_like(
        data, replace_prob
    )  # get the mask matrix of token being masked
    masked_input = masked_input.masked_fill(
        mask * replace_prob, mask_token_id
    )  # get the data has been masked by mask_token
    # mask out any tokens to padding tokens that were not originally going to be masked
    labels = data.masked_fill(~mask, pad_token_id)  # the label of masked tokens
    return masked_input, labels


# get the mask matrix which cannot be masked
def mask_with_tokens(t, token_ids):
    init_no_mask = torch.full_like(t, False, dtype=torch.bool)
    mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask)
    return mask


def get_mask_subset_with_prob(mask, prob):
    batch, seq_len, device = *mask.shape, mask.device
    max_masked = math.ceil(
        prob * seq_len
    )  # num of mask of a single sequence in average
    num_tokens = mask.sum(
        dim=-1, keepdim=True
    )  # num of pure tokens of each sequence except special tokens
    mask_excess = (
        torch.cat((torch.zeros(0), torch.arange(mask.size(-1)).repeat(mask.size(0))))
        .reshape(mask.size(0), mask.size(-1))
        .to(device)
    )
    mask_excess = (
        mask_excess >= (num_tokens * prob).ceil()
    )  # only 15% of pure tokens can be masked
    mask_excess = mask_excess[
        :, :max_masked
    ]  # get difference between 15% of pure tokens and 15% of all tokens
    rand = torch.rand((batch, seq_len), device=device).masked_fill(
        ~mask, -1e9
    )  # rand (0-1) as prob, special token use -1e9
    _, sampled_indices = rand.topk(max_masked, dim=-1)  # get index of topk prob to mask
    sampled_indices = (sampled_indices + 1).masked_fill_(
        mask_excess, 0
    )  # delete difference of mask not pure
    new_mask = torch.zeros(
        (batch, seq_len + 1), device=device
    )  # get (batch, seq_len) shape zero matrix
    new_mask.scatter_(-1, sampled_indices, 1)  # set masks in zero matrix as 1
    return new_mask[:, 1:].bool()  # the final mask, True is mask


# get the random prob matrix and True means smaller than prob threshold
def prob_mask_like(t, prob):
    return torch.zeros_like(t).float().uniform_(0, 1) < prob
