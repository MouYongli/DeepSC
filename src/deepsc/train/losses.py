import logging
from typing import Optional

import torch
from torch import nn

from src.deepsc.utils import LDAMLoss


class LossManager:
    """
    Centralized loss strategies for CE and regression.

    - ce_loss_mode: one of {standard, mean, weighted, ldam, alternating, mogaide, warm_alternating}
    - regression_loss: one of {mse, huber}
    """

    def __init__(
        self,
        args,
        num_bins: int,
        ignore_index: int = -100,
        class_counts: Optional[torch.Tensor] = None,
    ) -> None:
        self.args = args
        self.num_bins = num_bins
        self.ignore_index = ignore_index
        self.class_counts = class_counts

        # Determine CE mode: prefer explicit key; else infer from legacy flags
        self.ce_mode = (
            getattr(args, "ce_loss_mode", None) or self._infer_ce_mode_from_flags()
        )
        self.regression_mode = getattr(args, "regression_loss", None) or (
            "huber" if getattr(args, "enable_huber_loss", False) else "mse"
        )

        # Prepare CE criteria
        self._ce = nn.CrossEntropyLoss(reduction="mean", ignore_index=self.ignore_index)
        self._weighted_ce = None
        weight_tensor = None
        # Prefer explicit weights from args
        if hasattr(args, "ce_weight") and args.ce_weight is not None:
            try:
                wt = torch.tensor(list(args.ce_weight), dtype=torch.float)
                if wt.numel() == (self.num_bins + 1):
                    weight_tensor = wt
                else:
                    logging.warning(
                        f"[LossManager] ce_weight length {wt.numel()} != num_bins+1 ({self.num_bins+1}); ignoring."
                    )
            except Exception as e:
                logging.warning(f"[LossManager] invalid ce_weight: {e}")
        # Backward-compat: if weighted flag set and bins match old hardcoded example, use it
        if (
            weight_tensor is None
            and getattr(args, "weighted_ce_loss", False)
            and (self.num_bins + 1) == 4
        ):
            weight_tensor = torch.tensor([0.0, 1.0, 6.9, 118.7], dtype=torch.float)
        if weight_tensor is not None:
            self._weighted_ce = nn.CrossEntropyLoss(
                weight=weight_tensor, reduction="mean", ignore_index=self.ignore_index
            )

        # Prepare LDAM if needed and counts available
        self._ldam = None
        if self.ce_mode in {"ldam", "alternating", "warm_alternating", "mogaide"}:
            if class_counts is not None:
                cls_num_list = class_counts.detach().cpu().numpy()
                if cls_num_list[0] == 0:
                    cls_num_list[0] = 1
                self._ldam = LDAMLoss(
                    cls_num_list=cls_num_list,
                    max_m=0.5,
                    s=30,
                    ignore_index=self.ignore_index,
                )
            else:
                logging.warning(
                    "[LossManager] class_counts missing; LDAM-based modes will fall back to standard/mean."
                )

        # Prepare regression criterion
        if self.regression_mode == "huber":
            self._reg_crit = nn.HuberLoss(reduction="none")
        else:
            self._reg_crit = nn.MSELoss(reduction="none")

        # Progress context
        self.epoch: int = 1
        self.iter_idx: Optional[int] = None
        self.epoch_length: Optional[int] = None

    def _infer_ce_mode_from_flags(self) -> str:
        a = self.args
        if getattr(a, "enable_warm_alternating_ldam_mean_ce_loss", False):
            return "warm_alternating"
        if getattr(a, "enable_alternating_ldam_mean_ce_loss", False):
            return "alternating"
        if getattr(a, "enable_adaptive_ce_loss", False):
            return "mogaide"
        if getattr(a, "use_ldam_loss", False):
            return "ldam"
        if getattr(a, "weighted_ce_loss", False):
            return "weighted"
        if getattr(a, "mean_ce_loss", False):
            return "mean"
        return "standard"

    def set_progress(
        self,
        epoch: int,
        iter_idx: Optional[int] = None,
        epoch_length: Optional[int] = None,
    ) -> None:
        self.epoch = epoch
        self.iter_idx = iter_idx
        self.epoch_length = epoch_length

    def _flatten_logits_labels(self, logits: torch.Tensor, labels: torch.Tensor):
        logits_flat = logits.view(-1, self.num_bins + 1)
        labels_flat = labels.view(-1)
        return logits_flat, labels_flat

    def _per_bin_mean_ce(
        self, logits: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute per-bin CE (exclude bin 0) and average."""
        logits_flat, labels_flat = self._flatten_logits_labels(logits, labels)
        ce = nn.CrossEntropyLoss(reduction="mean")
        losses = []
        valid_mask = labels_flat != self.ignore_index
        for i in range(1, self.num_bins + 1):
            mask = (labels_flat == i) & valid_mask
            if mask.any():
                losses.append(ce(logits_flat[mask], labels_flat[mask]))
        if not losses:
            return torch.tensor(0.0, device=logits.device)
        return torch.stack(losses).mean()

    def ce_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.to(logits.device)
        logits_flat, labels_flat = self._flatten_logits_labels(logits, labels)

        mode = self.ce_mode
        if mode == "mean":
            return self._per_bin_mean_ce(logits, labels)
        if mode == "weighted" and self._weighted_ce is not None:
            self._weighted_ce.to(logits.device)
            return self._weighted_ce(logits_flat, labels_flat)
        if mode == "ldam" and self._ldam is not None:
            self._ldam.to(logits.device)
            return self._ldam(logits_flat, labels_flat)
        if mode == "alternating":
            if (self.epoch % 2 == 1) and (self._ldam is not None):
                self._ldam.to(logits.device)
                return self._ldam(logits_flat, labels_flat)
            return self._per_bin_mean_ce(logits, labels)
        if mode == "mogaide":
            if self.epoch in (1, 2):
                return self._ce(logits_flat, labels_flat)
            if self.epoch in (3, 4) and self._weighted_ce is not None:
                self._weighted_ce.to(logits.device)
                return self._weighted_ce(logits_flat, labels_flat)
            return self._per_bin_mean_ce(logits, labels)
        if (
            mode == "warm_alternating"
            and self.epoch_length
            and self.iter_idx is not None
            and self._ldam is not None
        ):
            progress = max(
                0.0, min(1.0, float(self.iter_idx) / float(self.epoch_length))
            )
            if self.epoch % 2 == 1:
                ldam_w, ce_w = 1.0 - progress, progress
            else:
                ldam_w, ce_w = progress, 1.0 - progress
            self._ldam.to(logits.device)
            ldam_val = self._ldam(logits_flat, labels_flat)
            mean_val = self._per_bin_mean_ce(logits, labels)
            return ldam_w * ldam_val + ce_w * mean_val

        # Fallback: standard CE
        return self._ce(logits_flat, labels_flat)

    def get_regression_criterion(self):
        return self._reg_crit


class LegacyLossCalculator:
    """
    Legacy loss calculation functions moved from trainer.py
    This class provides backward compatibility for existing loss calculation methods.
    """

    def __init__(
        self,
        args,
        num_bins: int,
        ignore_index: int = -100,
        class_counts: Optional[torch.Tensor] = None,
    ):
        self.args = args
        self.num_bins = num_bins
        self.ignore_index = ignore_index
        self.class_counts = class_counts
        self.epoch = 1
        self.iteration = 0
        self.epoch_length = 0

        # Initialize loss functions
        self.cross_entropy_loss_fn = nn.CrossEntropyLoss(
            reduction="mean", ignore_index=self.ignore_index
        )
        self.ldam_loss_fn = None
        self.regression_loss_fn = None

    def calculate_per_bin_ce_loss(self, logits, discrete_expr_label, ignore_index=-100):
        """
        logits: (batch, seq_len, num_bins)
        discrete_expr_label: (batch, seq_len)
        返回: (num_bins,) 每个bin的平均交叉熵损失
        Average calculation does not include bin0
        """
        num_bins = self.num_bins
        ce_losses = []
        logits_flat = logits.reshape(-1, num_bins + 1)
        labels_flat = discrete_expr_label.reshape(-1)
        for i in range(1, num_bins + 1):  # Skip bin0
            # Only count samples where label is i and not ignore_index
            mask = (labels_flat == i) & (labels_flat != ignore_index)
            if mask.sum() == 0:
                ce_losses.append(torch.tensor(0.0, device=logits.device))
                continue
            logits_i = logits_flat[mask]
            labels_i = labels_flat[mask]
            ce = nn.CrossEntropyLoss(reduction="mean")
            ce_loss = ce(logits_i, labels_i)
            ce_losses.append(ce_loss)
        if len(ce_losses) == 0:
            return torch.tensor(0.0, device=logits.device)
        return torch.stack(ce_losses)  # (num_bins-1,)

    def calculate_ce_loss(self, logits, discrete_expr_label):
        self.cross_entropy_loss_fn.to(logits.device)
        logits_reshaped = logits.view(-1, self.num_bins + 1)
        labels_reshaped = discrete_expr_label.view(-1)
        return self.cross_entropy_loss_fn(logits_reshaped, labels_reshaped)

    def calculate_ldam_ce_loss(self, logits, discrete_expr_label):
        self.ldam_loss_fn.to(logits.device)
        logits_reshaped = logits.view(-1, self.num_bins + 1)
        labels_reshaped = discrete_expr_label.view(-1)
        return self.ldam_loss_fn(logits_reshaped, labels_reshaped)

    def calculate_mean_ce_loss(self, logits, discrete_expr_label):
        per_bin_ce_loss = self.calculate_per_bin_ce_loss(logits, discrete_expr_label)
        mean_ce_loss = (
            per_bin_ce_loss.mean()
            if per_bin_ce_loss is not None
            else torch.tensor(0.0, device=logits.device)
        )
        return mean_ce_loss

    def calculate_alternating_ldam_mean_ce_loss(
        self, epoch, logits, discrete_expr_label
    ):
        if epoch % 2 == 1:
            ce_loss = self.calculate_ldam_ce_loss(logits, discrete_expr_label)
        else:
            ce_loss = self.calculate_mean_ce_loss(logits, discrete_expr_label)
        return ce_loss

    def calculate_mogaide_ce_loss(self, logits, discrete_expr_label):
        if self.epoch == 1 or self.epoch == 2:
            ce_loss = self.cross_entropy_loss_fn(logits, discrete_expr_label)
        else:
            ce_loss = self.calculate_mean_ce_loss(logits, discrete_expr_label)
        return ce_loss

    def calculate_warm_alternating_ldam_mean_ce_loss(
        self, epoch, logits, discrete_expr_label, epoch_length, index
    ):
        ldam_loss = self.calculate_ldam_ce_loss(logits, discrete_expr_label)
        per_bin_ce_loss_mean = self.calculate_mean_ce_loss(logits, discrete_expr_label)
        # Odd rounds: first half biased towards ldam, even rounds: first half biased towards ce
        progress = index / epoch_length if epoch_length > 0 else 0.0
        if epoch % 2 == 1:
            # Odd rounds: first half more ldam, second half more ce
            ldam_weight = 1.0 - progress
            ce_weight = progress
        else:
            # Even rounds: first half more ce, second half more ldam
            ldam_weight = progress
            ce_weight = 1.0 - progress
        ce_loss = ldam_weight * ldam_loss + ce_weight * per_bin_ce_loss_mean
        return ce_loss

    def init_ldam_loss(self):
        logging.info("Using LDAM loss...")
        # 使用缓存的class_counts
        class_counts = self.class_counts
        cls_num_list = class_counts.cpu().numpy()

        # Ensure padding class (class 0) has reasonable weight, avoid division by zero error
        if cls_num_list[0] == 0:
            logging.warning(
                "Warning: Padding class (bin 0) has 0 samples, setting to 1 to avoid division by zero"
            )
            cls_num_list[0] = (
                1  # Set to 1 to avoid division by zero, but doesn't affect actual training
            )

        return LDAMLoss(
            cls_num_list=cls_num_list, max_m=0.5, s=30, ignore_index=self.ignore_index
        )

    def init_loss_fn(self):
        if (
            self.args.enable_alternating_ldam_mean_ce_loss
            or self.args.enable_warm_alternating_ldam_mean_ce_loss
            or self.args.use_ldam_loss
        ):
            self.ldam_loss_fn = self.init_ldam_loss()
        elif self.args.weighted_ce_loss or self.args.enable_adaptive_ce_loss:
            self.cross_entropy_loss_fn = nn.CrossEntropyLoss(
                reduction="mean", ignore_index=self.ignore_index
            )
        else:
            self.cross_entropy_loss_fn = nn.CrossEntropyLoss(
                reduction="mean", ignore_index=self.ignore_index
            )

        if self.args.enable_mse_loss:
            self.regression_loss_fn = nn.MSELoss(reduction="none")
        elif self.args.enable_huber_loss:
            self.regression_loss_fn = nn.HuberLoss(reduction="none")

    def calculate_loss(
        self,
        enable_ce,
        enable_mse,
        logits=None,
        discrete_expr_label=None,
        regression_output=None,
        continuous_expr_label=None,
        mask=None,
        y=None,
        ce_loss_weight=1.0,
        mse_loss_weight=1.0,
        l0_lambda=0.0,
        is_val=False,
    ):
        """
        Main loss calculation function moved from trainer.py
        """
        from src.deepsc.utils import (
            masked_mse_loss,
            weighted_masked_mse_loss,
            weighted_masked_mse_loss_v2,
        )

        total_loss = 0.0
        ce_loss = torch.tensor(
            0.0, device=logits.device if logits is not None else "cpu"
        )
        regression_loss = torch.tensor(
            0.0, device=logits.device if logits is not None else "cpu"
        )
        l0_loss = 0.0

        if enable_ce and logits is not None and discrete_expr_label is not None:
            # Ensure label and logits are on the same device
            discrete_expr_label = discrete_expr_label.to(logits.device)
            if self.args.enable_alternating_ldam_mean_ce_loss:
                ce_loss = self.calculate_alternating_ldam_mean_ce_loss(
                    self.epoch, logits, discrete_expr_label
                )
            elif self.args.enable_adaptive_ce_loss:
                ce_loss = self.calculate_mogaide_ce_loss(logits, discrete_expr_label)
            elif self.args.enable_warm_alternating_ldam_mean_ce_loss:
                ce_loss = self.calculate_warm_alternating_ldam_mean_ce_loss(
                    self.epoch,
                    logits,
                    discrete_expr_label,
                    self.epoch_length,
                    self.iteration,
                )
            elif self.args.use_ldam_loss:
                ce_loss = self.calculate_ldam_ce_loss(logits, discrete_expr_label)
            elif self.args.mean_ce_loss:
                ce_loss = self.calculate_mean_ce_loss(logits, discrete_expr_label)
            else:
                ce_loss = self.calculate_ce_loss(logits, discrete_expr_label)
            total_loss += ce_loss_weight * ce_loss

        if (
            enable_mse
            and regression_output is not None
            and continuous_expr_label is not None
            and mask is not None
        ):
            if self.args.use_normal_regression_loss:
                regression_loss = masked_mse_loss(
                    regression_output,
                    continuous_expr_label,
                    mask,
                    loss_fn=self.regression_loss_fn,
                    reduction="mean",
                )
            elif self.args.use_hard_regression_loss:
                regression_loss = weighted_masked_mse_loss(
                    regression_output,
                    continuous_expr_label,
                    mask,
                    loss_fn=self.regression_loss_fn,
                    reduction="mean",
                    log_each=is_val and self.args.show_mse_loss_details,
                )
            elif self.args.use_exp_regression_loss:
                regression_loss = weighted_masked_mse_loss_v2(
                    regression_output,
                    continuous_expr_label,
                    mask,
                    loss_fn=self.regression_loss_fn,
                    reduction="mean",
                    log_each=is_val and self.args.show_mse_loss_details,
                )
            total_loss += mse_loss_weight * regression_loss

        if y is not None:
            l0_loss = (y[..., 0].abs().sum() + y[..., 2].abs().sum()) / (
                y.shape[0] * y.shape[1] * y.shape[2]
            )
            total_loss += 0.1 * l0_loss
        else:
            l0_loss = torch.tensor(0.0)  # Ensure it's a Tensor

        return total_loss, ce_loss, regression_loss, l0_loss
