import logging
from typing import Optional

import torch
from torch import nn

from src.deepsc.utils import LDAMLoss


class LossCalculator:
    """
    Legacy loss calculation functions moved from trainer.py
    This class provides backward compatibility for existing loss calculation methods.
    """

    def __init__(
        self,
        enable_l0: bool,
        enable_mse: bool,
        enable_ce: bool,
        ce_loss_weight: float,
        target_mse_loss_weight: float,
        weighted_ce_loss: bool,
        mean_ce_loss: bool,
        use_ldam_loss: bool,
        enable_adaptive_ce_loss: bool,
        enable_alternating_ldam_mean_ce_loss: bool,
        enable_mse_loss: bool,
        enable_huber_loss: bool,
        use_normal_regression_loss: bool,
        use_hard_regression_loss: bool,
        use_exp_regression_loss: bool,
        l0_lambda: float,
        num_bins: int,
        ignore_index: int = -100,
        class_counts: Optional[torch.Tensor] = None,
        show_mse_loss_details: bool = False,
    ):
        self.enable_ce = enable_ce
        self.enable_mse = enable_mse
        self.enable_l0 = enable_l0
        self.ce_loss_weight = ce_loss_weight
        self.mse_loss_weight = target_mse_loss_weight
        self.weighted_ce_loss = weighted_ce_loss
        self.mean_ce_loss = mean_ce_loss
        self.use_ldam_loss = use_ldam_loss
        self.enable_adaptive_ce_loss = enable_adaptive_ce_loss
        self.enable_alternating_ldam_mean_ce_loss = enable_alternating_ldam_mean_ce_loss
        self.enable_mse_loss = enable_mse_loss
        self.enable_huber_loss = enable_huber_loss
        self.use_normal_regression_loss = use_normal_regression_loss
        self.use_hard_regression_loss = use_hard_regression_loss
        self.use_exp_regression_loss = use_exp_regression_loss
        self.l0_lambda = l0_lambda
        self.num_bins = num_bins
        self.ignore_index = ignore_index
        self.class_counts = class_counts
        self.epoch = 1
        self.iteration = 0
        self.show_mse_loss_details = show_mse_loss_details

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
        if self.enable_alternating_ldam_mean_ce_loss or self.use_ldam_loss:
            self.ldam_loss_fn = self.init_ldam_loss()
        elif self.weighted_ce_loss or self.enable_adaptive_ce_loss:
            self.cross_entropy_loss_fn = nn.CrossEntropyLoss(
                reduction="mean", ignore_index=self.ignore_index
            )
        else:
            self.cross_entropy_loss_fn = nn.CrossEntropyLoss(
                reduction="mean", ignore_index=self.ignore_index
            )

        if self.enable_mse_loss:
            self.regression_loss_fn = nn.MSELoss(reduction="none")
        elif self.enable_huber_loss:
            self.regression_loss_fn = nn.HuberLoss(reduction="none")

    def calculate_loss(
        self,
        logits=None,
        discrete_expr_label=None,
        regression_output=None,
        continuous_expr_label=None,
        mask=None,
        y=None,
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

        if self.enable_ce and logits is not None and discrete_expr_label is not None:
            # Ensure label and logits are on the same device
            discrete_expr_label = discrete_expr_label.to(logits.device)
            if self.enable_alternating_ldam_mean_ce_loss:
                ce_loss = self.calculate_alternating_ldam_mean_ce_loss(
                    self.epoch, logits, discrete_expr_label
                )
            elif self.enable_adaptive_ce_loss:
                ce_loss = self.calculate_mogaide_ce_loss(logits, discrete_expr_label)
            elif self.use_ldam_loss:
                ce_loss = self.calculate_ldam_ce_loss(logits, discrete_expr_label)
            elif self.mean_ce_loss:
                ce_loss = self.calculate_mean_ce_loss(logits, discrete_expr_label)
            else:
                ce_loss = self.calculate_ce_loss(logits, discrete_expr_label)
            total_loss += self.ce_loss_weight * ce_loss

        if (
            self.enable_mse
            and regression_output is not None
            and continuous_expr_label is not None
            and mask is not None
        ):
            if self.use_normal_regression_loss:
                regression_loss = masked_mse_loss(
                    regression_output,
                    continuous_expr_label,
                    mask,
                    loss_fn=self.regression_loss_fn,
                    reduction="mean",
                )
            elif self.use_hard_regression_loss:
                regression_loss = weighted_masked_mse_loss(
                    regression_output,
                    continuous_expr_label,
                    mask,
                    loss_fn=self.regression_loss_fn,
                    reduction="mean",
                    log_each=is_val and self.logging.show_mse_loss_details,
                )
            elif self.use_exp_regression_loss:
                regression_loss = weighted_masked_mse_loss_v2(
                    regression_output,
                    continuous_expr_label,
                    mask,
                    loss_fn=self.regression_loss_fn,
                    reduction="mean",
                    log_each=is_val and self.logging.show_mse_loss_details,
                )
            total_loss += self.mse_loss_weight * regression_loss

        if y is not None:
            l0_loss = (y[..., 0].abs().sum() + y[..., 2].abs().sum()) / (
                y.shape[0] * y.shape[1] * y.shape[2]
            )
            total_loss += 0.1 * l0_loss
        else:
            l0_loss = torch.tensor(0.0)  # Ensure it's a Tensor

        return total_loss, ce_loss, regression_loss, l0_loss
