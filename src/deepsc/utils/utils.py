from __future__ import print_function
import logging
import math
import os
import os.path as osp
import random
import warnings
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


class FocalLoss(_WeightedLoss):
    """
    Focal Loss for multi-class classification.
    Args:
        weight (Tensor, optional): a manual rescaling weight given to each class.
        gamma (float, optional): focusing parameter gamma >= 0.
        ignore_index (int, optional): Specifies a
        target value that is ignored and does not contribute to the input gradient.
        reduction (string, optional): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
    """

    def __init__(self, weight=None, gamma=2.0, ignore_index=-100, reduction="mean"):
        super().__init__(weight=weight, reduction=reduction)
        self.weight = weight
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input, target):
        if len(input.shape) > 2:
            input = input.reshape(-1, input.size(-1))
        if len(target.shape) > 1:
            target = target.reshape(-1)
        valid_mask = target != self.ignore_index
        input = input[valid_mask]
        target = target[valid_mask]
        logpt = F.log_softmax(input, dim=-1)
        pt = torch.exp(logpt)
        logpt = logpt.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = pt.gather(1, target.unsqueeze(1)).squeeze(1)
        if self.weight is not None:
            at = self.weight.gather(0, target)
            logpt = logpt * at
        loss = -((1 - pt) ** self.gamma) * logpt
        if self.reduction == "mean":
            return (
                loss.mean()
                if loss.numel() > 0
                else torch.tensor(0.0, device=input.device)
            )
        elif self.reduction == "sum":
            return loss.sum()
        else:
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


def weighted_masked_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    reduction: str = "mean",
    log_each: bool = False,
    loss_type: str = "mse",
):
    """
    分区加权 MSE Loss，只在 mask=True 的位置计算。
    权重区间：
        target == 1       → 0.2
        1 < target < 3    → 0.5
        3 <= target < 5   → 1.0
        target >= 5       → 2.0
    加权后进行归一化处理，防止梯度爆炸或 collapse。
    loss_type: 'mse'（默认）或 'huber'，决定损失函数类型。
    """

    if loss_type == "mse":
        loss_fn = nn.MSELoss(reduction="none")
    elif loss_type == "huber":
        loss_fn = nn.HuberLoss(reduction="none")
    else:
        raise ValueError(f"Invalid loss_type: {loss_type}. Supported: 'mse', 'huber'.")

    elementwise_loss = loss_fn(pred, target)  # shape: (B, T)

    with torch.no_grad():
        weights = torch.ones_like(target)
        weights[(target == 1)] = 0.2
        weights[(target > 1) & (target < 3)] = 0.5
        weights[(target >= 3) & (target < 5)] = 1.0
        weights[target >= 5] = 2.0

    elementwise_loss = elementwise_loss[mask]
    weights = weights[mask]

    if elementwise_loss.numel() == 0:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    weights = weights / (weights.sum() + 1e-8)

    final_loss = elementwise_loss * weights

    if log_each:
        print("[VAL] pred.mean():", pred[mask].mean().item())
        print("[VAL] pred.std():", pred[mask].std().item())
        print("[VAL] target.mean():", target[mask].mean().item())
        print("[VAL] target.std():", target[mask].std().item())
        print("[VAL] weighted loss sample:", final_loss[:10].tolist())

    if reduction == "mean":
        return final_loss.sum()
    elif reduction == "sum":
        return final_loss.sum()
    elif reduction == "none":
        return final_loss
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}")


def interval_average_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    loss_type: str = "mse",
    log_stats: bool = False,
    return_tensor: bool = False,
):
    """
    计算四个区间的平均MSE Loss，只在 mask=True 的位置计算。

    区间划分：
        target == 1       → interval_1
        1 < target < 3    → interval_2
        3 <= target < 5   → interval_3
        target >= 5       → interval_4

    Args:
        pred (torch.Tensor): 预测值
        target (torch.Tensor): 目标值
        mask (torch.Tensor): 掩码，True表示需要计算的位置
        loss_type (str): 'mse'（默认）或 'huber'
        log_stats (bool): 是否打印统计信息
        return_tensor (bool): 是否返回tensor格式的overall loss

    Returns:
        dict: 包含四个区间平均loss的字典
            {
                'interval_1': float,  # target == 1
                'interval_2': float,  # 1 < target < 3
                'interval_3': float,  # 3 <= target < 5
                'interval_4': float,  # target >= 5
                'overall': float/tensor,  # 所有区间平均值的平均
                'stats': dict,  # 统计信息（如果log_stats=True）
            }
    """

    if loss_type == "mse":
        loss_fn = nn.MSELoss(reduction="none")
    elif loss_type == "huber":
        loss_fn = nn.HuberLoss(reduction="none")
    else:
        raise ValueError(f"Invalid loss_type: {loss_type}. Supported: 'mse', 'huber'.")

    elementwise_loss = loss_fn(pred, target)  # shape: (B, T)

    # 定义四个区间
    intervals = {
        "interval_1": target == 1,
        "interval_2": (target > 1) & (target < 3),
        "interval_3": (target >= 3) & (target < 5),
        "interval_4": target >= 5,
    }

    results = {}
    interval_means = []
    stats = {}

    for interval_name, interval_mask in intervals.items():
        # 结合mask和区间mask
        combined_mask = mask & interval_mask
        interval_loss = elementwise_loss[combined_mask]

        if interval_loss.numel() == 0:
            results[interval_name] = 0.0
            stats[f"{interval_name}_count"] = 0
            stats[f"{interval_name}_mean"] = 0.0
            stats[f"{interval_name}_std"] = 0.0
        else:
            avg_loss = interval_loss.mean().item()
            results[interval_name] = avg_loss
            interval_means.append(avg_loss)

            # 统计信息
            stats[f"{interval_name}_count"] = interval_loss.numel()
            stats[f"{interval_name}_mean"] = avg_loss
            stats[f"{interval_name}_std"] = interval_loss.std().item()

    # 计算总体平均：对各个区间的平均值再求平均
    if interval_means:
        overall_loss = sum(interval_means) / len(interval_means)
        results["overall"] = (
            overall_loss
            if not return_tensor
            else torch.tensor(overall_loss, device=pred.device)
        )
        stats["total_count"] = sum(
            stats[f"{interval_name}_count"] for interval_name in intervals.keys()
        )
        stats["total_mean"] = overall_loss
        stats["total_std"] = 0.0  # 简化处理
    else:
        results["overall"] = (
            0.0 if not return_tensor else torch.tensor(0.0, device=pred.device)
        )
        stats["total_count"] = 0
        stats["total_mean"] = 0.0
        stats["total_std"] = 0.0

    # 添加统计信息到结果中
    if log_stats:
        results["stats"] = stats
        print(f"[Interval MSE Stats] Total samples: {stats['total_count']}")
        for interval_name in intervals.keys():
            count = stats[f"{interval_name}_count"]
            mean_loss = stats[f"{interval_name}_mean"]
            std_loss = stats[f"{interval_name}_std"]
            print(
                f"[Interval MSE Stats] {interval_name}: count={count}, mean={mean_loss:.4f}, std={std_loss:.4f}"
            )

    return results


def weighted_masked_mse_loss_v2(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    reduction: str = "mean",
    log_each: bool = False,
):
    """
    以2为底的指数加权 MSE Loss，只在 mask=True 的位置上计算。
    权重 = 2^(target - shift)
    shift 可调节权重起点，默认为0。
    """
    loss_fn = nn.MSELoss(reduction="none")
    elementwise_loss = loss_fn(pred, target)  # shape: (B, T)

    # 生成以2为底的指数权重
    weights = torch.pow(1.8, target - 1)

    # 只在 mask=True 的地方计算
    weighted_loss = elementwise_loss * weights
    masked_weighted_loss = weighted_loss[mask]

    if reduction == "mean":
        if masked_weighted_loss.numel() == 0:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        return masked_weighted_loss.mean()
    elif reduction == "sum":
        if masked_weighted_loss.numel() == 0:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        return masked_weighted_loss.sum()
    elif reduction == "none":
        return masked_weighted_loss
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


"""
@file source.py
@author Ryan Missel

Class definition for the CosineAnnealingWarmRestarts with both Max-LR Decay and global LinearWarmup.
"""


class CosineAnnealingWarmRestartsWithDecayAndLinearWarmup(_LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr, :math:`T_{cur}`
    is the number of epochs since the last restart and :math:`T_{i}` is the number
    of epochs between two warm restarts in SGDR:
    """

    def __init__(
        self,
        optimizer,
        T_0,
        T_mult=1,
        eta_min=0,
        last_epoch=-1,
        verbose=False,
        warmup_steps=350,
        decay=1,
    ):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = last_epoch
        super(CosineAnnealingWarmRestartsWithDecayAndLinearWarmup, self).__init__(
            optimizer, last_epoch
        )

        # Decay attributes
        self.decay = decay
        self.initial_lrs = self.base_lrs

        # Warmup attributes
        self.warmup_steps = warmup_steps
        self.current_steps = 0
        self.verbose = verbose

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.warmup_steps == 0:
            # 没有 warmup，直接用 cosine annealing
            return [
                self.eta_min
                + (base_lr - self.eta_min)
                * (1 + math.cos(math.pi * self.T_cur / self.T_i))
                / 2
                for base_lr in self.base_lrs
            ]
        else:
            # 有 warmup，按原有逻辑
            return [
                (self.current_steps / self.warmup_steps)
                * (
                    self.eta_min
                    + (base_lr - self.eta_min)
                    * (1 + math.cos(math.pi * self.T_cur / self.T_i))
                    / 2
                )
                for base_lr in self.base_lrs
            ]

    def step(self, epoch=None):
        """Step could be called after every batch update"""
        if epoch is None and self.last_epoch < 0:
            epoch = 0

        if self.T_cur + 1 == self.T_i:
            if self.verbose:
                print("multiplying base_lrs by {:.4f}".format(self.decay))
            self.base_lrs = [base_lr * self.decay for base_lr in self.base_lrs]

        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1

            if self.current_steps < self.warmup_steps:
                self.current_steps += 1

            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult

        self.last_epoch = math.floor(epoch)

        class _enable_get_lr_call:

            def __init__(self, o):
                self.o = o

            def __enter__(self):
                self.o._get_lr_called_within_step = True
                return self

            def __exit__(self, type, value, traceback):
                self.o._get_lr_called_within_step = False
                return self

        with _enable_get_lr_call(self):
            for i, data in enumerate(zip(self.optimizer.param_groups, self.get_lr())):
                param_group, lr = data
                param_group["lr"] = lr
                self.print_lr(self.verbose, i, lr, epoch)

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]


def interval_masked_mse_loss(pred, target, mask):
    """
    统计四个区间的未加权 MSE
    返回: dict, key为区间名，value为MSE
    区间：lt3, 3to5, 5to7, ge7
    """
    loss_fn = nn.MSELoss(reduction="none")
    elementwise_loss = loss_fn(pred, target)
    results = {}
    intervals = [
        ("lt3", target < 3),
        ("3to5", (target >= 3) & (target < 5)),
        ("5to7", (target >= 5) & (target < 7)),
        ("ge7", target >= 7),
    ]
    for name, cond in intervals:
        interval_mask = cond & mask
        if interval_mask.sum() == 0:
            results[name] = torch.tensor(0.0, device=pred.device)
        else:
            results[name] = elementwise_loss[interval_mask].mean()
    return results


class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30, ignore_index=-100):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        self.m_list = torch.tensor(m_list, dtype=torch.float32)
        self.s = s
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, x, target):
        valid_mask = target != self.ignore_index
        if not valid_mask.any():
            return torch.tensor(0.0, device=x.device, requires_grad=True)

        x_valid = x[valid_mask]
        target_valid = target[valid_mask]

        index = torch.zeros_like(x_valid, dtype=torch.bool)
        index.scatter_(1, target_valid.view(-1, 1), True)

        m_list = self.m_list.to(x.device)
        batch_m = m_list[target_valid]  # shape: (N,)

        x_m = x_valid.clone()
        x_m[index] -= batch_m

        return F.cross_entropy(self.s * x_m, target_valid, weight=self.weight)
