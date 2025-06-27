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
    pad_token_id: int = 0
    pad_value: int = 0
    num_genes: int = 34682
    do_padding: bool = True
    do_mlm: bool = True
    do_binning: bool = True
    mlm_probability: float = 0.15
    max_length: Optional[int] = None
    sampling: bool = True
    keep_first_n_tokens: int = 1
    gene_from_zero: bool = True

    def __post_init__(self):
        self.cls_token_id = self.num_genes + 1
        self.cls_value = self.num_bins + 1
        self.mask_value = self.num_bins + 2
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
        该函数的输出为 data_dict 包括以下key：
        gene: 基因id
        masked_discrete_expr: 离散表达值掩码 (输入模型)
        masked_continuous_expr: 连续表达值掩码 (输入模型)
        discrete_expr_label: 离散表达值label （和模型的输出比较）(添加了-100,表明这些位置不参加cross entropy loss)
        continuous_expr_label: 连续表达值label （和模型的输出比较）（未添加-100）
        mask: 掩码的位置 （用于计算MSE的label，continuous_expr_label使用，在masked_mse函数中使用）
        """
        if not isinstance(examples[0], Mapping):
            return NotImplementedError
        max_ori_len = max(len(example["genes"]) for example in examples)
        _max_length = self.max_length if max_ori_len >= self.max_length else max_ori_len
        padded_genes = []
        padded_discrete_expr = []
        padded_continuous_expr = []
        for i in range(len(examples)):
            genes = examples[i]["genes"]
            expressions = examples[i]["expressions"]
            expression_label = expressions.clone().float()
            # 如果gene_from_zero为True，则将gene加1 为pad_token_id留出位置
            if self.gene_from_zero:
                genes = genes + 1
            # 做binning
            if self.do_binning:
                expressions = binning(
                    row=expressions,
                    n_bins=self.num_bins,
                )
                # TODO: 这里需要检查一下，是否需要long (我猜应该不需要了，之前需要他是因为没找出bug)
                expressions = expressions.long()
            # 添加cls token
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
            expression_label = torch.cat(
                [
                    torch.tensor(
                        [self.cls_value],
                        dtype=expression_label.dtype,
                        device=expression_label.device,
                    ),
                    expression_label,
                ]
            )
            # 对gene, expressions, expression_label进行采样或截断并填充
            genes, expressions, expression_label = self._sample_or_truncate_plus_pad(
                genes, expressions, expression_label, _max_length
            )  # torch tensors of length _max_length
            padded_genes.append(genes)
            padded_discrete_expr.append(expressions)
            padded_continuous_expr.append(expression_label)
        # 对padded_genes, padded_discrete_expr, padded_continuous_expr进行stack
        padded_genes = torch.stack(padded_genes, dim=0)
        padded_discrete_expr = torch.stack(padded_discrete_expr, dim=0)
        padded_continuous_expr = torch.stack(padded_continuous_expr, dim=0)
        continuous_expr_label = padded_continuous_expr.clone().float()

        # 这两个添加完cls之后即可加入data_dict
        data_dict = {
            "gene": padded_genes,
            "continuous_expr_label": continuous_expr_label,
        }

        # 为padded_continuous_expr添加mask 以及为padded_discrete_expr添加mask
        if self.do_mlm:
            masked_discrete_expressions, mask = self._mask(
                padded_discrete_expr, return_mask=True
            )
            masked_continuous_expressions, _ = self._mask(
                padded_continuous_expr, return_mask=True
            )
        else:
            masked_discrete_expressions = padded_discrete_expr
            mask = torch.zeros_like(padded_discrete_expr, dtype=torch.bool)
        data_dict["masked_discrete_expr"] = masked_discrete_expressions
        data_dict["masked_continuous_expr"] = masked_continuous_expressions
        # 检查masked_discrete_expr的第二维的第一个数是否为cls
        if (
            data_dict["masked_discrete_expr"].shape[1] == 0
            or data_dict["masked_discrete_expr"][0, 0].item() != self.num_bins + 1
        ):
            raise ValueError(
                f"离散mask 的第二维的第一个数不是 {self.num_bins}，而是 {data_dict['masked_discrete_expr'][0, 0].item()}"
            )

        # 新增 label: mask 位置为原始 expression，其他为 -100
        # discrete_expr_label可以在这里做掩码，然而continuous_expr_label,可以在trainer里传入自定义的masked mse 里面，在那里处理应该做掩码的indices
        discrete_expr_label = torch.full_like(padded_discrete_expr, -100)
        discrete_expr_label[mask] = padded_discrete_expr[mask]
        data_dict["discrete_expr_label"] = discrete_expr_label
        data_dict["mask"] = mask

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
        expression_label: torch.Tensor,
        max_length: int,
    ) -> Tuple[torch.LongTensor, torch.Tensor]:
        assert len(genes) == len(expressions)
        if len(genes) == max_length:
            return genes, expressions, expression_label
        if len(genes) > max_length:  # sample or truncate
            if self.sampling:
                return self._sample(genes, expressions, expression_label, max_length)
            else:
                return (
                    genes[:max_length],
                    expressions[:max_length],
                    expression_label[:max_length],
                )
        else:  # pad
            return self._pad(genes, expressions, expression_label, max_length)

    def _sample(
        self,
        genes: torch.LongTensor,
        expressions: torch.Tensor,
        expression_label: torch.Tensor,
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
            return genes[indices], expressions[indices], expression_label[indices]

        # keep the first n tokens unchanged
        _n = self.keep_first_n_tokens
        indices = torch.randperm(len(genes) - _n, device=device)[: max_length - _n]
        indices = torch.cat([torch.arange(_n), indices + _n], dim=0)
        return genes[indices], expressions[indices], expression_label[indices]

    def _pad(
        self,
        genes: torch.LongTensor,
        expressions: torch.Tensor,
        expression_label: torch.Tensor,
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
        expression_label = torch.cat(
            [
                expression_label,
                torch.full(
                    (max_length - len(expression_label),),
                    self.pad_value,
                    dtype=expression_label.dtype,
                    device=device,
                ),
            ]
        )
        return genes, expressions, expression_label

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
