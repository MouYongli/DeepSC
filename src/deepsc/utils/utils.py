from __future__ import print_function
import logging
import math
import os
import os.path as osp
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.loss import _WeightedLoss
from torch.optim.lr_scheduler import _LRScheduler

from datetime import datetime


def testPackage():
    print("#############")


def path_of_file(file_path, file_name):
    if file_name == "cell":
        searchKey1 = "cell"
        searchKey2 = ".csv"

    if file_name == "gene":
        searchKey1 = "gene"
        searchKey2 = ".txt"

    files_in_directory = {
        f.name.lower(): f.name for f in file_path.iterdir() if f.is_file()
    }
    lower_files = list(files_in_directory.keys())
    search_file_path = Path("")

    search_files = [
        f for f in lower_files if f.startswith(searchKey1) and f.endswith(searchKey2)
    ]
    if search_files:
        if not len(search_files) > 1:
            # print(f"find {file_name} file: {search_files[0]} in path {file_path}")
            original_file_name = files_in_directory[search_files[0]]
            search_file_path = file_path / original_file_name
            return search_file_path
        else:
            print(f"Multiple files found in path {file_path}")
    else:
        parent_folder = file_path.parent
        files_in_parent_directory = {
            f.name.lower(): f.name for f in parent_folder.iterdir() if f.is_file()
        }
        lower_files_in_parent_directory = list(files_in_parent_directory.keys())
        search_files = [
            f
            for f in lower_files_in_parent_directory
            if f.startswith(searchKey1) and f.endswith(searchKey2)
        ]
        if search_files:
            if not len(search_files) > 1:
                original_file_name = files_in_parent_directory[search_files[0]]
                search_file_path = parent_folder / original_file_name
                # print(f"find gene file: {search_files[0]} in path {parent_folder}")
                return search_file_path
            else:
                print(f"Multiple files found in path {file_path}")
        else:
            print(f"Corresponding file not found in path {file_path}")


def seed_all(seed_value, cuda_deterministic=False):
    """
    set all random seeds
    """
    random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


# TODO: Duplicated utils functions? refactor!


def setup_logging(type: str, log_path: str = "./logs") -> str:
    os.makedirs(log_path, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = osp.join(log_path, f"pretrain_{type}_{timestamp}.log")

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="w",
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)

    logging.info(f"日志文件: {log_file}")
    return log_file


def set_log(logfileName, rank=-1):
    """
    save log
    """
    log_file_folder = os.path.dirname(logfileName)
    time_now = datetime.datetime.now()
    logfileName = f"{logfileName}_{time_now.year}_{time_now.month}_{time_now.day}_{time_now.hour}_{time_now.minute}.log"
    if not os.path.exists(log_file_folder):
        os.makedirs(log_file_folder)
    else:
        pass

    logging.basicConfig(
        level=logging.INFO if rank in [-1, 0] else logging.WARN,
        format="[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s",
        datefmt="[%X]",
        handlers=[logging.FileHandler(logfileName), logging.StreamHandler()],
    )
    logger = logging.getLogger()
    return logger


def save_ckpt(
    epoch, model, optimizer, scheduler, losses, model_name, ckpt_folder, iteration=None
):
    """
    save checkpoint
    """
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "losses": losses,
    }
    # Always save as latest_checkpoint.pth (overwrite)
    torch.save(ckpt, os.path.join(ckpt_folder, "latest_checkpoint.pth"))
    # Save with epoch and iteration if provided, else just epoch
    if iteration is not None:
        filename = f"{model_name}_{epoch}_{iteration}.pth"
    else:
        filename = f"{model_name}_{epoch}.pth"
    torch.save(ckpt, os.path.join(ckpt_folder, filename))


def save_ckpt_fabric(
    epoch,
    model,
    optimizer,
    scheduler,
    losses,
    model_name,
    ckpt_folder,
    fabric,
    iteration=None,
):
    if not os.path.exists(ckpt_folder):
        os.makedirs(ckpt_folder)
    state = {
        "model": model,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "iteration": iteration,
        "epoch": epoch,
    }
    print(f"Saving checkpoint to {os.path.join(ckpt_folder, 'latest_checkpoint.ckpt')}")
    fabric.save(os.path.join(ckpt_folder, "latest_checkpoint.ckpt"), state)
    if iteration is not None:
        filename = f"{model_name}_{epoch}_{iteration}.ckpt"
    else:
        filename = f"{model_name}_{epoch}.ckpt"
    fabric.save(os.path.join(ckpt_folder, filename), state)


def get_reduced(tensor, current_device, dest_device, world_size):
    """
    将不同GPU上的变量或tensor集中在主GPU上，并得到均值
    """
    tensor = (
        tensor.clone().detach() if torch.is_tensor(tensor) else torch.tensor(tensor)
    )
    tensor = tensor.to(current_device)
    torch.distributed.reduce(tensor, dst=dest_device)
    tensor_mean = tensor.item() / world_size
    return tensor_mean


def get_reduced_with_fabric(tensor, fabric):
    reduced_tensor = fabric.all_reduce(tensor, reduce_op="mean")
    return reduced_tensor.item()


def get_ndtensor_reduced(tensor, current_device, dest_device, world_size):
    """
    将不同GPU上的变量或tensor集中在主GPU上，并得到均值, 需要是2维张量
    """
    tensor = (
        tensor.clone().detach() if torch.is_tensor(tensor) else torch.tensor(tensor)
    )
    tensor = tensor.to(current_device)
    torch.distributed.reduce(tensor, dst=dest_device)
    tensor_mean = torch.zeros(tensor.shape)
    if len(tensor.shape) == 2:
        for i in range(tensor.shape[0]):
            for j in range(tensor.shape[1]):
                tensor_mean[i, j] = tensor[i, j].item() / world_size
    elif len(tensor.shape) == 1:
        for i in range(tensor.shape[0]):
            tensor_mean[i] = tensor[i].item() / world_size
    return tensor_mean


def numel(m: torch.nn.Module, only_trainable: bool = False):
    """
    returns the total number of parameters used by `m` (only counting
    shared parameters once); if `only_trainable` is True, then only
    includes parameters with `requires_grad = True`
    """
    parameters = m.parameters()
    if only_trainable:
        parameters = list(p for p in parameters if p.requires_grad)
    unique = dict((p.data_ptr(), p) for p in parameters).values()
    return sum(p.numel() for p in unique)


def label_smooth(y, K, epsilon=0.1):
    """
    Label smoothing for multiclass labels
    One hot encode labels `y` over `K` classes. `y` should be of the form [1, 6, 3, etc.]
    """
    m = len(y)
    out = np.ones((m, K)) * epsilon / K
    for index in range(m):
        out[index][y[index] - 1] += 1 - epsilon
    return torch.tensor(out)


class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    """
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.
    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """

    def __init__(self, dataset, batch_size, world_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = world_size
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = (
            int(
                math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)
            )
            * self.batch_size
        )
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[
            self.rank * self.num_samples : (self.rank + 1) * self.num_samples
        ]
        return iter(indices)

    def __len__(self):
        return self.num_samples


def distributed_concat(tensor, num_total_examples, world_size):
    """
    合并不同进程的inference结果
    """
    output_tensors = [tensor.clone() for _ in range(world_size)]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    # truncate the dummy elements added by SequentialDistributedSampler
    return concat[:num_total_examples]


class CosineAnnealingWarmupRestarts(_LRScheduler):
    """
    optimizer (Optimizer): Wrapped optimizer.
    first_cycle_steps (int): First cycle step size.
    cycle_mult(float): Cycle steps magnification. Default: -1.
    max_lr(float): First cycle's max learning rate. Default: 0.1.
    min_lr(float): Min learning rate. Default: 0.001.
    warmup_steps(int): Linear warmup step size. Default: 0.
    gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
    last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        first_cycle_steps: int,
        cycle_mult: float = 1.0,
        max_lr: float = 0.1,
        min_lr: float = 0.001,
        warmup_steps: int = 0,
        gamma: float = 1.0,
        last_epoch: int = -1,
    ):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle_mult = cycle_mult  # cycle steps magnification
        self.base_max_lr = max_lr  # first max learning rate
        self.max_lr = max_lr  # max learning rate in the current cycle
        self.min_lr = min_lr  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size
        self.gamma = gamma  # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [
                (self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps
                + base_lr
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr
                + (self.max_lr - base_lr)
                * (
                    1
                    + math.cos(
                        math.pi
                        * (self.step_in_cycle - self.warmup_steps)
                        / (self.cur_cycle_steps - self.warmup_steps)
                    )
                )
                / 2
                for base_lr in self.base_lrs
            ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = (
                    int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult)
                    + self.warmup_steps
                )
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.0:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(
                        math.log(
                            (
                                epoch / self.first_cycle_steps * (self.cycle_mult - 1)
                                + 1
                            ),
                            self.cycle_mult,
                        )
                    )
                    self.cycle = n
                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps
                        * (self.cycle_mult**n - 1)
                        / (self.cycle_mult - 1)
                    )
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (
                        n
                    )
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


class DistanceLoss(_WeightedLoss):
    """
    CrossEntropyLoss with Distance Weighted
    """

    def __init__(self, weight=None, reduction="mean", ignore_index=None):
        super().__init__(weight=weight, reduction=reduction)
        self.weight = weight
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        if len(inputs.shape) > 2:
            inputs = inputs.reshape(-1, inputs.size(-1))
        if len(targets.shape) > 1:
            targets = targets.reshape(-1)
        if self.ignore_index is not None:
            keep_index = (targets != self.ignore_index).nonzero(as_tuple=True)[0]
            targets = torch.index_select(
                targets, 0, keep_index
            )  # targets[targets != self.ignore_index]
            inputs = torch.index_select(inputs, 0, keep_index)
        lsm = F.log_softmax(inputs, -1)
        targets = (
            torch.empty(size=(targets.size(0), inputs.size(-1)), device=targets.device)
            .fill_(0)
            .scatter_(1, targets.data.unsqueeze(1), 1)
        )
        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)
        loss = -(targets * lsm).sum(-1)
        inputs = nn.Softmax(dim=-1)(inputs)[..., 1:-1].argmax(dim=-1) + 1
        # print('inputs', inputs.device, inputs.shape)
        targets = nn.Softmax(dim=-1)(targets)[..., 1:-1].argmax(dim=-1) + 1
        # print('targets', targets.device, targets.shape)
        distance = abs(inputs - targets) + 1e-2
        # print('loss.shape', loss.shape)
        # print('distance.shape', distance.shape)
        loss = loss * distance
        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()
        return loss


class LabelSmoothCrossEntropyLoss(_WeightedLoss):
    """
    CrossEntropyLoss with Label Somoothing
    """

    def __init__(self, weight=None, reduction="mean", smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets: torch.Tensor, n_classes: int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = (
                torch.empty(size=(targets.size(0), n_classes), device=targets.device)
                .fill_(smoothing / (n_classes - 1))
                .scatter_(1, targets.data.unsqueeze(1), 1.0 - smoothing)
            )
        return targets

    def forward(self, inputs, targets):
        targets = LabelSmoothCrossEntropyLoss._smooth_one_hot(
            targets, inputs.size(-1), self.smoothing
        )
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()

        return loss


def masked_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    reduction: str = "mean",
):
    """
    计算只在 mask=True 的位置上的 MSE Loss。

    Args:
        pred (Tensor): 模型输出，shape = (B, T)
        target (Tensor): 目标值，shape = (B, T)
        mask (BoolTensor): 掩码，True 表示需要计算的位置
        reduction (str): "mean"、"sum" 或 "none"

    Returns:
        Tensor: loss 值
    """
    loss_fn = nn.MSELoss(reduction="none")
    elementwise_loss = loss_fn(pred, target)  # shape: (B, T)

    masked_loss = elementwise_loss[mask]  # 只取被掩码的部分

    if reduction == "mean":
        if masked_loss.numel() == 0:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        return masked_loss.mean()
    elif reduction == "sum":
        if masked_loss.numel() == 0:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        return masked_loss.sum()
    elif reduction == "none":
        return masked_loss
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}")


def log_stats(
    is_master,
    num_bins,
    final,
    labels,
    epoch,
    index,
    print_detailed_stats=False,
    print_pred_dist=False,
):
    if not is_master:
        return

    non_padded_mask = labels != -100
    valid_labels = labels[non_padded_mask]
    valid_preds = final[non_padded_mask]
    non_padded_count = valid_preds.numel()
    num_classes = num_bins + 1

    if non_padded_count == 0:
        return

    if print_pred_dist:
        pred_counts = torch.bincount(valid_preds.to(torch.int), minlength=num_classes)
        pred_dist = pred_counts.float() / non_padded_count
        print(f"--- Epoch {epoch} Iteration {index} Prediction Distribution ---")
        for i, p in enumerate(pred_dist):
            if p > 0:
                print(f"  Bin {i}: {p.item():.2%}")
        print("---------------------------------------------")

    if print_detailed_stats:
        # 1. Label distribution
        label_counts = torch.bincount(valid_labels.to(torch.int), minlength=num_classes)
        label_dist = label_counts.float() / non_padded_count

        # 2. Prediction distribution
        pred_counts = torch.bincount(valid_preds.to(torch.int), minlength=num_classes)
        pred_dist = pred_counts.float() / non_padded_count

        # 5. Correct prediction distribution
        correct_mask = valid_preds == valid_labels
        correct_preds = valid_preds[correct_mask]
        num_correct = correct_preds.numel()
        if num_correct > 0:
            correct_pred_counts = torch.bincount(
                correct_preds.to(torch.int), minlength=num_classes
            )
            correct_pred_dist = correct_pred_counts.float() / num_correct
        else:
            correct_pred_dist = torch.zeros(num_classes, device=labels.device)

        # 6. Incorrect prediction distribution
        incorrect_mask = ~correct_mask
        incorrect_preds = valid_preds[incorrect_mask]
        num_incorrect = incorrect_preds.numel()
        if num_incorrect > 0:
            incorrect_pred_counts = torch.bincount(
                incorrect_preds.to(torch.int), minlength=num_classes
            )
            incorrect_pred_dist = incorrect_pred_counts.float() / num_incorrect
        else:
            incorrect_pred_dist = torch.zeros(num_classes, device=labels.device)

        # 7. Labels for incorrect predictions
        incorrect_labels = valid_labels[incorrect_mask]
        if num_incorrect > 0:
            incorrect_label_counts = torch.bincount(
                incorrect_labels.to(torch.int), minlength=num_classes
            )
            incorrect_label_dist = incorrect_label_counts.float() / num_incorrect
        else:
            incorrect_label_dist = torch.zeros(num_classes, device=labels.device)

        def print_dist(dist_tensor, name):
            print(f"--- {name} Distribution ---")
            for i, p in enumerate(dist_tensor):
                if p > 0:
                    print(f"  Bin {i}: {p:.2%}")

        print(f"\n===== Epoch {epoch} Validation Stats =====")
        # 3 & 4. Total non-padded count
        print(f"Total non-padded labels (predictions): {non_padded_count}")

        # 1. Label distribution
        print_dist(label_dist, "Label")
        # 2. Prediction distribution
        print_dist(pred_dist, "Prediction")
        # 5. Correct prediction distribution
        print(f"Total correct predictions: {num_correct}")
        print_dist(correct_pred_dist, "Correct Predictions")
        # 6. Incorrect prediction distribution
        print(f"Total incorrect predictions: {num_incorrect}")
        print_dist(incorrect_pred_dist, "Incorrect Predictions")
        # 7. Labels for incorrect predictions
        print_dist(incorrect_label_dist, "Labels of Incorrect Predictions")
        print("================================\n")


def compute_bin_distribution(final, valid_mask, num_bins, topk=None):
    """
    计算预测分布。
    Args:
        final: 预测的类别 (tensor)
        valid_mask: 有效位置的mask (tensor, bool)
        num_bins: bin的数量 (int)
        topk: 若为None，返回前num_bins个bin的分布，否则返回预测比例最多的前topk个bin及其编号和比例
    Returns:
        如果topk为None，返回[(bin编号, 比例), ...]，长度为num_bins
        否则，返回[(bin编号, 比例), ...]，长度为topk
    """
    if final is not None and valid_mask.any():
        pred_counts = torch.bincount(final[valid_mask].cpu(), minlength=num_bins + 1)
        pred_dist = (pred_counts.float() / pred_counts.sum()).tolist()
        if topk is not None:
            top_bins = sorted(enumerate(pred_dist), key=lambda x: x[1], reverse=True)[
                :topk
            ]
            return top_bins
        else:
            return list(enumerate(pred_dist[:num_bins]))
    else:
        return None
