from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple

import torch

from deepsc.preprocess import binning


@dataclass
class DataCollator:
    """
    Data collator for the mask value learning task. It pads the sequences to
    the maximum length in the batch and masks the gene expression values.

    Args:
        do_padding (:obj:`bool`): whether to pad the sequences to the max length.
        pad_token_id (:obj:`int`, optional): the token id to use for padding.
            This is required if do_padding is True.
        pad_value (:obj:`int`): the value to use for padding the expression
            values to the max length.
        do_mlm (:obj:`bool`): whether to do masking with MLM.
        do_binning (:obj:`bool`): whether to bin the expression values.
        mlm_probability (:obj:`float`): the probability of masking with MLM.
        mask_value (:obj:`int`): the value to fill at the expression postions
            that are masked.
        max_length (:obj:`int`, optional): the maximum length of the sequences.
            This is required if do_padding is True.
        sampling (:obj:`bool`): whether to do sampling instead of truncation if
            length > max_length.
        keep_first_n_tokens (:obj:`int`): the number of tokens in the beginning
            of the sequence to keep unchanged from sampling. This is useful when
            special tokens have been added to the beginning of the sequence.
            Default to 1.
        gene_from_zero (:obj:`bool`): whether to add 1 to gene tokens and set pad_token_id to 0.
    """

    num_bins: int
    num_genes: int = 34682
    do_padding: bool = True
    pad_token_id: int = 0
    pad_value: int = 0
    do_mlm: bool = True
    do_binning: bool = True
    mlm_probability: float = 0.15
    max_length: Optional[int] = None
    sampling: bool = True
    keep_first_n_tokens: int = 1
    gene_from_zero: bool = True

    def __post_init__(self):
        self.mask_value = self.num_bins + 2
        self.cls_token_id = self.num_genes + 1
        self.cls_value = self.num_bins + 1
        if self.do_padding:
            if self.pad_token_id is None:
                raise ValueError("`pad_token_id` is required if `do_padding`.")
            if self.max_length is None:
                raise ValueError("`max_length` is required if `do_padding`.")
        if self.mlm_probability <= 0 or self.mlm_probability >= 1:
            raise ValueError("`mlm_probability` must be between 0 and 1.")

        if self.keep_first_n_tokens < 0 or self.keep_first_n_tokens > self.max_length:
            raise ValueError(
                "`keep_first_n_tokens` must be between 0 and `max_length` "
                f"({self.max_length})."
            )

    def __call__(
        self, examples: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        Each example is like:
            {'id': tensor(184117),
            'genes': tensor([36572, 17868, ..., 17072]),
            'expressions': tensor([ 0.,  2., ..., 18.])}
        """
        if not isinstance(examples[0], Mapping):
            return NotImplementedError
        max_ori_len = max(len(example["genes"]) for example in examples)
        _max_length = self.max_length if max_ori_len >= self.max_length else max_ori_len
        padded_genes = []
        padded_expressions = []

        for i in range(len(examples)):
            genes = examples[i]["genes"]
            expressions = examples[i]["expressions"]
            if self.gene_from_zero:
                genes = genes + 1
            if self.do_binning:
                expressions = binning(
                    row=expressions,
                    n_bins=self.num_bins,
                )
                expressions = expressions.long()
                expressions = expressions + 1
            genes = torch.cat(
                [
                    torch.tensor(
                        [self.cls_token_id], dtype=genes.dtype, device=genes.device
                    ),
                    genes,
                ]
            )
            expressions = torch.cat(
                [
                    torch.tensor(
                        [self.cls_value],
                        dtype=expressions.dtype,
                        device=expressions.device,
                    ),
                    expressions,
                ]
            )
            genes, expressions = self._sample_or_truncate_plus_pad(
                genes, expressions, _max_length
            )  # torch tensors of length _max_length
            padded_genes.append(genes)
            padded_expressions.append(expressions)
        padded_genes = torch.stack(padded_genes, dim=0)
        padded_expressions = torch.stack(padded_expressions, dim=0)
        data_dict = {
            "gene": padded_genes,
            "expr": padded_expressions,
        }

        # mask
        if self.do_mlm:
            masked_expressions, mask = self._mask(padded_expressions, return_mask=True)
        else:
            masked_expressions = padded_expressions
            mask = torch.zeros_like(padded_expressions, dtype=torch.bool)
        data_dict["masked_expr"] = masked_expressions

        # 检查 masked_expr 的第二维的第一个数是否为 51
        if (
            data_dict["masked_expr"].shape[1] == 0
            or data_dict["masked_expr"][0, 0].item() != self.num_bins + 1
        ):
            raise ValueError(
                f"masked_expr 的第二维的第一个数不是 {self.num_bins+1}，而是 {data_dict['masked_expr'][0, 0].item()}"
            )

        # 新增 label: mask 位置为原始 expression，其他为 -100
        label = torch.full_like(padded_expressions, -100)
        label[mask] = padded_expressions[mask]
        data_dict["label"] = label

        return data_dict

    def _mask(
        self, expressions: torch.Tensor, return_mask: bool = False
    ) -> torch.Tensor:
        """
        Mask the expression values with MLM.
        """
        device = expressions.device
        shape = expressions.shape

        probability_matrix = torch.full(shape, self.mlm_probability)
        # set padded postion probability to 0
        probability_matrix[expressions.eq(self.pad_value)] = 0
        if self.keep_first_n_tokens > 0:
            probability_matrix[:, : self.keep_first_n_tokens] = 0

        mask = torch.bernoulli(probability_matrix).bool()
        mask = mask.to(device)

        masked_expressions = expressions.masked_fill(mask, self.mask_value)
        if return_mask:
            return masked_expressions, mask
        return masked_expressions

    def _sample_or_truncate_plus_pad(
        self,
        genes: torch.LongTensor,
        expressions: torch.Tensor,
        max_length: int,
    ) -> Tuple[torch.LongTensor, torch.Tensor]:
        assert len(genes) == len(expressions)
        if len(genes) == max_length:
            return genes, expressions
        if len(genes) > max_length:  # sample or truncate
            if self.sampling:
                return self._sample(genes, expressions, max_length)
            else:
                return genes[:max_length], expressions[:max_length]
        else:  # pad
            return self._pad(genes, expressions, max_length)

    def _sample(
        self,
        genes: torch.LongTensor,
        expressions: torch.Tensor,
        max_length: int,
    ) -> Tuple[torch.LongTensor, torch.Tensor]:
        # NOTE: the fastest way to sample in torch has been benchmarked here
        # https://discuss.pytorch.org/t/torch-equivalent-of-numpy-random-choice/16146/19
        # it shows the randperm on gpu is the fastest.
        # NOTE: also, the current implementation permute the orders of the genes
        # and expressions, although it is probably a nice argmentation.
        device = genes.device
        if self.keep_first_n_tokens == 0:
            indices = torch.randperm(len(genes), device=device)[:max_length]
            return genes[indices], expressions[indices]

        # keep the first n tokens unchanged
        _n = self.keep_first_n_tokens
        indices = torch.randperm(len(genes) - _n, device=device)[: max_length - _n]
        indices = torch.cat([torch.arange(_n), indices + _n], dim=0)
        return genes[indices], expressions[indices]

    def _pad(
        self,
        genes: torch.LongTensor,
        expressions: torch.Tensor,
        max_length: int,
    ):
        device = genes.device
        genes = torch.cat(
            [
                genes,
                torch.full(
                    (max_length - len(genes),),
                    self.pad_token_id,
                    dtype=genes.dtype,
                    device=device,
                ),
            ]
        )
        expressions = torch.cat(
            [
                expressions,
                torch.full(
                    (max_length - len(expressions),),
                    self.pad_value,
                    dtype=expressions.dtype,
                    device=device,
                ),
            ]
        )
        return genes, expressions

    def _truncate_by_expression(
        self,
        genes: torch.LongTensor,
        expressions: torch.Tensor,
        max_length: int,
    ) -> Tuple[torch.LongTensor, torch.Tensor]:
        # 保留最前面的 keep_first_n_tokens 不变
        n = self.keep_first_n_tokens
        # 对剩余部分按表达值降序排列，取 top-k
        expr_tail = expressions[n:]
        genes_tail = genes[n:]
        # 获取降序索引
        sorted_idx = torch.argsort(expr_tail, descending=True)[: (max_length - n)]
        # 最终索引：前 n 个 + 按表达值选出的
        selected_idx = torch.cat(
            [torch.arange(n, device=genes.device), sorted_idx + n], dim=0
        )
        # 根据索引截取
        return genes[selected_idx], expressions[selected_idx]
