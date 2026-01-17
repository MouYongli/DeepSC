from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple

import torch


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
        dynamic_mask_probabilities (:obj:`dict`, optional): dictionary containing mask probabilities for each bin.
            If provided, will override the default hardcoded probabilities.
    """

    num_bins: int
    pad_token_id: int = 0
    pad_value: int = 0
    num_genes: int = 34683
    do_padding: bool = True
    do_mlm: bool = True
    do_binning: bool = True
    mlm_probability: float = 0.15
    max_length: Optional[int] = None
    sampling: bool = True
    keep_first_n_tokens: int = 1
    gene_from_zero: bool = True
    dynamic_mask_probabilities: Optional[dict] = None
    use_max_cell_length: bool = True
    cell_type: bool = False

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

        # Validate dynamic_mask_probabilities
        if self.dynamic_mask_probabilities is not None:
            for bin_idx in range(1, self.num_bins + 1):
                if bin_idx not in self.dynamic_mask_probabilities:
                    raise ValueError(f"Missing mask probability for bin {bin_idx}")
                prob = self.dynamic_mask_probabilities[bin_idx]
                if not (0 <= prob <= 1):
                    raise ValueError(
                        f"Invalid mask probability {prob} for bin {bin_idx}"
                    )

    def __call__(
        self, examples: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        This function outputs data_dict including the following keys:
        gene: gene id
        masked_discrete_expr: discrete expression mask (input to model)
        masked_continuous_expr: continuous expression mask (input to model)
        discrete_expr_label: discrete expression label (compared with model output) (added -100, indicating these positions don't participate in cross entropy loss)
        continuous_expr_label: continuous expression label (compared with model output) (not added -100)
        mask: mask positions (used for MSE label calculation, used by continuous_expr_label, used in masked_mse function)
        """
        if not isinstance(examples[0], Mapping):
            return NotImplementedError
        if self.use_max_cell_length:
            max_ori_len = max(len(example["genes"]) for example in examples)
            _max_length = (
                self.max_length if max_ori_len >= self.max_length else max_ori_len
            )
        else:
            _max_length = self.max_length
        if self.cell_type:
            if "cell_type_id" not in examples[0]:
                raise ValueError("`cell_type_id` is required when `cell_type` is True.")
            cell_type_ids = torch.tensor(
                [example["cell_type_id"] for example in examples], dtype=torch.long
            )
        padded_genes = []
        padded_discrete_expr = []
        padded_continuous_expr = []
        for i in range(len(examples)):
            genes = examples[i]["genes"]
            expressions = examples[i]["expressions"]
            expression_label = expressions.clone().float()
            # If gene_from_zero is True, add 1 to gene to reserve space for pad_token_id
            if self.gene_from_zero:
                genes = genes + 1
            # Do binning
            if self.do_binning:
                # Discretize by cell
                expressions = self.discretize_expression(expressions)
                expressions = expressions.long()
            # Add cls token
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
            # Sample or truncate and pad gene, expressions, expression_label
            genes, expressions, expression_label = self._sample_or_truncate_plus_pad(
                genes, expressions, expression_label, _max_length
            )  # torch tensors of length _max_length
            padded_genes.append(genes)
            padded_discrete_expr.append(expressions)
            padded_continuous_expr.append(expression_label)
        # Stack padded_genes, padded_discrete_expr, padded_continuous_expr
        padded_genes = torch.stack(padded_genes, dim=0)
        padded_discrete_expr = torch.stack(padded_discrete_expr, dim=0)
        padded_continuous_expr = torch.stack(padded_continuous_expr, dim=0)
        continuous_expr_label = padded_continuous_expr.clone().float()

        # These two can be added to data_dict after adding cls
        data_dict = {
            "gene": padded_genes,
            "continuous_expr_label": continuous_expr_label,
        }

        # Add mask for padded_continuous_expr and padded_discrete_expr
        if self.do_mlm:
            # Sample mask only once
            masked_discrete_expressions, mask = self._mask(
                padded_discrete_expr, return_mask=True
            )
            # Apply the same mask to continuous
            masked_continuous_expressions = padded_continuous_expr.masked_fill(
                mask, self.mask_value
            )
        else:
            masked_discrete_expressions = padded_discrete_expr
            masked_continuous_expressions = padded_continuous_expr
            mask = torch.zeros_like(padded_discrete_expr, dtype=torch.bool)
        data_dict["masked_discrete_expr"] = masked_discrete_expressions
        data_dict["masked_continuous_expr"] = masked_continuous_expressions
        # Check if the first number in the second dimension of masked_discrete_expr is cls
        if (
            data_dict["masked_discrete_expr"].shape[1] == 0
            or data_dict["masked_discrete_expr"][0, 0].item() != self.num_bins + 1
        ):
            raise ValueError(
                f"The first number in the second dimension of discrete mask is not {self.num_bins}, but {data_dict['masked_discrete_expr'][0, 0].item()}"
            )

        # Add label: mask position is original expression, others are -100
        # discrete_expr_label can be masked here, however continuous_expr_label can be passed to custom masked mse in trainer, where indices to be masked are handled
        discrete_expr_label = torch.full_like(padded_discrete_expr, -100)
        discrete_expr_label[mask] = padded_discrete_expr[mask]
        data_dict["discrete_expr_label"] = discrete_expr_label
        data_dict["mask"] = mask
        if self.cell_type:
            data_dict["cell_type_id"] = cell_type_ids
        return data_dict

    def _mask(
        self, expressions: torch.Tensor, return_mask: bool = False
    ) -> torch.Tensor:
        """
        Mask the expression values with MLM.
        """
        device = expressions.device
        shape = expressions.shape
        if self.dynamic_mask_probabilities is None:
            probability_matrix = torch.full(
                shape, self.mlm_probability, dtype=torch.float
            )
        else:
            probability_matrix = torch.zeros(shape, dtype=torch.float, device=device)
            for bin_idx in range(1, self.num_bins + 1):
                if bin_idx in self.dynamic_mask_probabilities:
                    probability_matrix[expressions == bin_idx] = (
                        self.dynamic_mask_probabilities[bin_idx]
                    )

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
                return self._top_expr_or_pad(
                    genes, expressions, expression_label, max_length
                )
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

    def discretize_expression(self, normalized_expr: torch.Tensor) -> torch.Tensor:
        """
        Expression discretization: b_j = Discretize_N(x̃_j)
        Args:
            normalized_expr: normalized expression x̃, shape: (g,)
        Returns:
            bin_indices: discretized bin indices b, shape: (g,)
        """
        min_val = normalized_expr.min()
        max_val = normalized_expr.max()
        normalized_range = (normalized_expr - min_val) / (max_val - min_val + 1e-8)
        bin_indices = torch.floor(normalized_range * (self.num_bins - 1)).long()
        bin_indices = torch.clamp(bin_indices, 0, self.num_bins - 1)
        return bin_indices + 1

    def discretize_expression_log_bins(
        self, normalized_expr: torch.Tensor
    ) -> torch.Tensor:
        """
        New expression discretization method:
        - Elements less than 2 are assigned to bin1
        - Elements >= 2 are divided into remaining bins by logarithmic intervals (larger elements have wider bin widths)
        Args:
            normalized_expr: original expression x, shape: (g,)
        Returns:
            bin_indices: discretized bin indices b, shape: (g,)
        """
        num_bins = self.num_bins
        expr = normalized_expr.clone()
        bin_indices = torch.zeros_like(expr, dtype=torch.long)

        # 1. Values less than 2 are assigned to bin1
        mask_lt2 = expr < 2
        bin_indices[mask_lt2] = 1

        # 2. Values >= 2 are divided into remaining bins (logarithmic binning)
        mask_ge2 = ~mask_lt2
        if mask_ge2.any():
            expr_ge2 = expr[mask_ge2]
            min_val = 2.0
            max_val = expr_ge2.max().item()
            n_log_bins = num_bins - 1  # bin1 already used
            if max_val > min_val:
                # Logarithmic binning
                log_min = torch.log(torch.tensor(min_val))
                log_max = torch.log(torch.tensor(max_val) + 1e-8)
                log_expr = torch.log(expr_ge2)
                # Normalize to [0,1]
                norm_log = (log_expr - log_min) / (log_max - log_min + 1e-8)
                # Assign to bins [2, num_bins]
                bins = torch.floor(norm_log * (n_log_bins - 1)).long() + 2
                bins = torch.clamp(bins, 2, num_bins)
                bin_indices[mask_ge2] = bins
            else:
                # If all values >= 2 are equal, assign to bin2
                bin_indices[mask_ge2] = 2
        return bin_indices

    def discretize_expression_three_bins(
        self, normalized_expr: torch.Tensor
    ) -> torch.Tensor:
        """
        Divide according to the following rules into 3 bins:
        - Less than 3 is bin1
        - Greater than or equal to 3 and less than 6.7 is bin2
        - Greater than or equal to 6.7 is bin3
        Args:
            normalized_expr: original expression x, shape: (g,)
        Returns:
            bin_indices: discretized bin indices b, shape: (g,)
        """
        expr = normalized_expr.clone()
        bin_indices = torch.zeros_like(expr, dtype=torch.long)

        # bin1: less than 3
        mask_bin1 = expr <= 2
        bin_indices[mask_bin1] = 1

        # bin2: greater than or equal to 3 and less than 6.7
        mask_bin2 = (expr > 2) & (expr < 5.5)
        bin_indices[mask_bin2] = 2

        # bin3: greater than or equal to 6.7
        mask_bin3 = expr >= 5.5
        bin_indices[mask_bin3] = 3

        return bin_indices

    def _top_expr_or_pad(
        self,
        genes: torch.LongTensor,
        expressions: torch.Tensor,
        expression_label: torch.Tensor,
        max_length: int,
    ) -> Tuple[torch.LongTensor, torch.Tensor, torch.Tensor]:
        """
        Prioritize truncating genes with highest expression values, pad if still not long enough.
        Keep the first keep_first_n_tokens unchanged.
        """
        device = genes.device
        _n = self.keep_first_n_tokens
        total_len = len(genes)
        # Keep first n unchanged
        if _n > 0:
            # Keep first n directly
            fixed_genes = genes[:_n]
            fixed_expr = expressions[:_n]
            fixed_label = expression_label[:_n]
            # Sort remaining part by expression value
            rest_genes = genes[_n:]
            rest_expr = expressions[_n:]
            rest_label = expression_label[_n:]
            if len(rest_genes) > 0:
                sorted_indices = torch.argsort(rest_expr, descending=True)
                rest_genes = rest_genes[sorted_indices]
                rest_expr = rest_expr[sorted_indices]
                rest_label = rest_label[sorted_indices]
            # Concatenate
            needed = max_length - _n
            selected_genes = rest_genes[:needed]
            selected_expr = rest_expr[:needed]
            selected_label = rest_label[:needed]
            out_genes = torch.cat([fixed_genes, selected_genes], dim=0)
            out_expr = torch.cat([fixed_expr, selected_expr], dim=0)
            out_label = torch.cat([fixed_label, selected_label], dim=0)
        else:
            # Sort all by expression value
            sorted_indices = torch.argsort(expressions, descending=True)
            out_genes = genes[sorted_indices][:max_length]
            out_expr = expressions[sorted_indices][:max_length]
            out_label = expression_label[sorted_indices][:max_length]
        # Pad if still not long enough
        if len(out_genes) < max_length:
            out_genes, out_expr, out_label = self._pad(
                out_genes, out_expr, out_label, max_length
            )
        return out_genes, out_expr, out_label
