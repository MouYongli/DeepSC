import logging
import math
import os
import os.path as osp
import random
import warnings
from pathlib import Path

import numpy as np
import scanpy as sc
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.loss import _WeightedLoss
from torch.optim.lr_scheduler import _LRScheduler

import wandb
from datetime import datetime


def path_of_file(file_path, file_name):
    """
    Find a file in a given directory based on file type.

    Args:
        file_path: Path to search in
        file_name: Type of file to search for ('cell' or 'gene')

    Returns:
        Path: The found file path

    Raises:
        ValueError: If file_name is not 'cell' or 'gene', or if multiple files found
        FileNotFoundError: If no matching file is found
        NotADirectoryError: If file_path is not a valid directory
    """
    if not isinstance(file_path, Path):
        file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Directory does not exist: {file_path}")

    if not file_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {file_path}")

    if file_name == "cell":
        search_key1 = "cell"
        search_key2 = ".csv"
    elif file_name == "gene":
        search_key1 = "gene"
        search_key2 = ".txt"
    else:
        raise ValueError(f"Invalid file_name: {file_name}. Must be 'cell' or 'gene'")

    files_in_directory = {
        f.name.lower(): f.name for f in file_path.iterdir() if f.is_file()
    }
    lower_files = list(files_in_directory.keys())

    search_files = [
        f for f in lower_files if f.startswith(search_key1) and f.endswith(search_key2)
    ]

    if search_files:
        if len(search_files) == 1:
            original_file_name = files_in_directory[search_files[0]]
            search_file_path = file_path / original_file_name
            return search_file_path
        else:
            raise ValueError(
                f"Multiple {file_name} files found in {file_path}: {search_files}"
            )
    else:
        # Search in parent directory
        parent_folder = file_path.parent
        if not parent_folder.exists() or not parent_folder.is_dir():
            raise FileNotFoundError(
                f"No {file_name} file found in {file_path} and parent directory is invalid"
            )

        files_in_parent_directory = {
            f.name.lower(): f.name for f in parent_folder.iterdir() if f.is_file()
        }
        lower_files_in_parent_directory = list(files_in_parent_directory.keys())
        search_files = [
            f
            for f in lower_files_in_parent_directory
            if f.startswith(search_key1) and f.endswith(search_key2)
        ]

        if search_files:
            if len(search_files) == 1:
                original_file_name = files_in_parent_directory[search_files[0]]
                search_file_path = parent_folder / original_file_name
                return search_file_path
            else:
                raise ValueError(
                    f"Multiple {file_name} files found in parent directory {parent_folder}: {search_files}"
                )
        else:
            raise FileNotFoundError(
                f"No {file_name} file found in {file_path} or its parent directory"
            )


def seed_all(seed_value, cuda_deterministic=False):
    """
    set all random seeds
    """
    random.seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        # torch.manual_seed already calls torch.cuda.manual_seed_all internally
        # So we only need to call torch.cuda.manual_seed for the current GPU
        torch.cuda.manual_seed(seed_value)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def setup_logging(
    log_path: str = "./logs",
    log_name: str = "deepsc",
    rank: int = -1,
    add_timestamp: bool = True,
    log_level: str = "INFO",
) -> str:
    """
    Setup unified logging configuration.

    Args:
        log_path: Directory to store log files
        log_name: Base name for the log file
        rank: Process rank for distributed training (-1 for single process)
        add_timestamp: Whether to add timestamp to log filename
        log_level: Logging level

    Returns:
        str: Path to the created log file
    """
    os.makedirs(log_path, exist_ok=True)

    # Build log filename
    if add_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{log_name}_{timestamp}.log"
    else:
        time_now = datetime.now()
        log_filename = f"{log_name}_{time_now.month}_{time_now.day}_{time_now.hour}_{time_now.minute}.log"

    log_file = osp.join(log_path, log_filename)

    # Set logging level based on rank
    if rank in [-1, 0]:
        level = getattr(logging, log_level.upper())
    else:
        level = logging.WARN

    # Configure logging
    logging.basicConfig(
        level=level,
        format="[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s",
        datefmt="[%X]",
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()],
        force=True,  # Reset any existing logging configuration
    )

    logger = logging.getLogger()
    logger.info(f"Log file initialized: {log_file}")

    return log_file


# Backward compatibility functions
def set_log(log_file_name, rank=-1):
    """Deprecated: Use setup_logging instead."""
    import warnings

    warnings.warn(
        "set_log is deprecated. Use setup_logging instead.", DeprecationWarning
    )
    return setup_logging(
        log_path=os.path.dirname(log_file_name),
        log_name=os.path.basename(log_file_name).replace(".log", ""),
        rank=rank,
        add_timestamp=False,
    )


def save_checkpoint(
    epoch,
    model,
    optimizer,
    scheduler,
    model_name: str,
    ckpt_folder: str,
    iteration=None,
    fabric=None,
    losses=None,
    chunk_idx=0,
):
    """
    Unified checkpoint saving function that works with or without Fabric.

    Args:
        epoch: Current epoch number
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        model_name: Name for the checkpoint file
        ckpt_folder: Directory to save checkpoints
        iteration: Optional iteration number
        fabric: Optional Fabric instance for distributed training
        losses: Optional losses dict (for non-fabric mode)
    """
    import wandb

    os.makedirs(ckpt_folder, exist_ok=True)

    # Get current wandb run_id and config if wandb is active
    wandb_run_id = None
    wandb_config = None
    if wandb.run is not None:
        wandb_run_id = wandb.run.id
        wandb_config = {
            "project": wandb.run.project,
            "entity": wandb.run.entity,
            "name": wandb.run.name,
            "tags": list(wandb.run.tags) if wandb.run.tags else [],
            "config": dict(wandb.run.config),
        }

    if fabric is not None:
        # Fabric mode - use fabric.save()
        state = {
            "model": (
                model.module.state_dict()
                if hasattr(model, "module")
                else model.state_dict()
            ),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "iteration": iteration,
            "wandb_run_id": wandb_run_id,
            "wandb_config": wandb_config,
            "chunk_idx": chunk_idx,  # 保存chunk索引
        }

        # Save latest checkpoint
        latest_path = os.path.join(ckpt_folder, "latest_checkpoint.ckpt")
        logging.info(f"Saving checkpoint to {latest_path}")
        fabric.save(latest_path, state)

        # Save numbered checkpoint
        if iteration is not None:
            filename = f"{model_name}_{epoch}_{iteration}.ckpt"
        else:
            filename = f"{model_name}_{epoch}.ckpt"
        fabric.save(os.path.join(ckpt_folder, filename), state)


# Backward compatibility functions
def save_ckpt(
    epoch, model, optimizer, scheduler, losses, model_name, ckpt_folder, iteration=None
):
    """Deprecated: Use save_checkpoint instead."""
    import warnings

    warnings.warn(
        "save_ckpt is deprecated. Use save_checkpoint instead.", DeprecationWarning
    )
    return save_checkpoint(
        epoch,
        model,
        optimizer,
        scheduler,
        model_name,
        ckpt_folder,
        iteration=iteration,
        losses=losses,
    )


def save_ckpt_fabric(
    epoch,
    model,
    optimizer,
    scheduler,
    model_name,
    ckpt_folder,
    fabric,
    iteration=None,
    chunk_idx=0,
):
    """Deprecated: Use save_checkpoint instead."""
    import warnings

    warnings.warn(
        "save_ckpt_fabric is deprecated. Use save_checkpoint instead.",
        DeprecationWarning,
    )
    return save_checkpoint(
        epoch,
        model,
        optimizer,
        scheduler,
        model_name,
        ckpt_folder,
        iteration=iteration,
        fabric=fabric,
        chunk_idx=chunk_idx,  # 保存chunk索引
    )


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
    loss_fn,
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
    loss_fn = loss_fn
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
    loss_fn,
    reduction: str = "mean",
    log_each: bool = False,
):
    """
    分区加权 MSE Loss，只在 mask=True 的位置计算。
    权重区间：
        target == 1       → 0.2
        1 < target < 3    → 0.5
        3 <= target < 5   → 1.0
        target >= 5       → 2.0
    加权后进行归一化处理，防止梯度爆炸或 collapse。
    """
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


def weighted_masked_mse_loss_v2(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    loss_fn,
    reduction: str = "mean",
    log_each: bool = False,
):
    """
    以2为底的指数加权 MSE Loss，只在 mask=True 的位置上计算。
    权重 = 2^(target - shift)
    shift 可调节权重起点，默认为0。
    """
    loss_fn = loss_fn
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
        recall, precision, f1, macro_f1, average_recall, average_precision = (
            compute_classification_metrics(
                valid_preds, valid_labels, num_classes, labels.device
            )
        )
        # INSERT_YOUR_CODE
        # 将每个类别的recall, precision, f1上传到wandb
        if "wandb" in globals() and wandb.run is not None:
            log_dict = {}
            for i in range(num_classes):
                log_dict[f"recall/bin{i}"] = recall[i].item()
                log_dict[f"precision/bin{i}"] = precision[i].item()
                log_dict[f"f1/bin{i}"] = f1[i].item()
            log_dict["val/macro_f1"] = macro_f1
            # 只对非第0类(bin0)求平均
            log_dict["val/average_recall"] = average_recall
            log_dict["val/average_precision"] = average_precision
            wandb.log(log_dict)

        print(f"\n===== Epoch {epoch} Validation Stats =====")
        print(f"Total non-padded labels (predictions): {valid_preds.numel()}")
        print(f"{'Class':>8} | {'Recall':>8} | {'Precision':>10} | {'F1':>8}")
        print("-" * 44)
        for i in range(num_classes):
            print(
                f"{i:>8} | {recall[i].item():>8.4f} | {precision[i].item():>10.4f} | {f1[i].item():>8.4f}"
            )
        print("-" * 44)
        print(f"{'Macro F1':>8} | {macro_f1:>8.4f}")
        print("================================\n")
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
        # 排除padding类（第0类），只对有效类别计算margin
        valid_cls_num_list = cls_num_list[1:]  # 排除第0类
        m_list = 1.0 / np.sqrt(np.sqrt(valid_cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        # 为padding类（第0类）设置margin为0
        self.m_list = torch.zeros(len(cls_num_list), dtype=torch.float32)
        self.m_list[1:] = torch.tensor(m_list, dtype=torch.float32)  # 第0类margin为0
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


def check_grad_flow(
    model: nn.Module,
    loss_tensor: torch.Tensor,
    verbose: bool = True,
    retain_graph: bool = False,
    backward_fn=None,
):
    """
    检查模型的梯度传导是否正常。
    Args:
        model: nn.Module
        loss_tensor: loss
        verbose: 是否详细打印
        retain_graph: 是否保留计算图
        backward_fn: 自定义反向传播函数（如 fabric.backward）
    """
    print("=" * 60)
    print("➡️ [检查开始] 反向传播中梯度传导情况...")

    # 保存原始梯度状态
    original_grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            original_grads[name] = param.grad.clone()

    model.zero_grad()

    try:
        if backward_fn is not None:
            backward_fn(loss_tensor, retain_graph=retain_graph)
        else:
            loss_tensor.backward(retain_graph=retain_graph)
    except Exception as e:
        print(f"[ERROR] 反向传播失败: {e}")
        # 恢复原始梯度
        for name, param in model.named_parameters():
            if name in original_grads:
                param.grad = original_grads[name]
        return {"ok": [], "zero": [], "none": []}

    no_grad_names = []
    zero_grad_names = []
    ok_grad_names = []

    for name, param in model.named_parameters():
        try:
            if param.grad is None:
                no_grad_names.append(name)
                if verbose:
                    print(f"[❌ NONE ] {name}: grad is None")
            elif torch.all(param.grad == 0):
                zero_grad_names.append(name)
                if verbose:
                    print(f"[⚠️ ZERO] {name}: grad == 0")
            else:
                ok_grad_names.append(name)
                if verbose:
                    print(
                        f"[✅ OK  ] {name}: grad max={param.grad.abs().max():.4e}, min={param.grad.abs().min():.4e}"
                    )
        except Exception as e:
            print(f"[ERROR] 检查参数 {name} 梯度时出错: {e}")
            no_grad_names.append(name)

    print("-" * 60)
    print(f"✅ 有效梯度参数数：{len(ok_grad_names)}")
    print(f"⚠️ 梯度为0的参数数：{len(zero_grad_names)}")
    print(f"❌ grad is None 的参数数：{len(no_grad_names)}")
    print("=" * 60)

    # 恢复原始梯度
    for name, param in model.named_parameters():
        if name in original_grads:
            param.grad = original_grads[name]

    return {
        "ok": ok_grad_names,
        "zero": zero_grad_names,
        "none": no_grad_names,
    }


def compute_M_from_y(y):
    """
    从 Gumbel Softmax 输出 y 计算门控矩阵 M
    Args:
        y: Gumbel Softmax 输出, shape: (batch, g, g, 3)
    Returns:
        M: 门控矩阵, shape: (batch, g, g)
    """
    return y[..., 0] * (-1) + y[..., 1] * 0 + y[..., 2] * (+1)


def print_m_matrix(epoch, index, M):
    print(f"\n=== Epoch {epoch}, Iteration {index}: M Matrix ===")
    # 只打印第一个batch的第一个样本的M矩阵
    M_sample = M[0].detach().cpu().numpy()
    print(f"M matrix shape: {M_sample.shape}")

    total_elements = M_sample.size
    inhibition_count = np.sum(M_sample == -1)
    no_relation_count = np.sum(M_sample == 0)
    activation_count = np.sum(M_sample == 1)

    print("M matrix distribution:")
    print(
        f"  Inhibition (-1): {inhibition_count} ({inhibition_count/total_elements*100:.1f}%)"
    )
    print(
        f"  No relation (0): {no_relation_count} ({no_relation_count/total_elements*100:.1f}%)"
    )
    print(
        f"  Activation (1): {activation_count} ({activation_count/total_elements*100:.1f}%)"
    )
    print("=" * 50)


# 计算每个类别的TP, FP, FN
def compute_classification_metrics(valid_preds, valid_labels, num_classes, device):
    """
    计算每个类别的TP, FP, FN, recall, precision, f1, macro_f1
    返回: recall, precision, f1, macro_f1
    注意：不计算bin0（即类别0）的指标
    """
    # True Positives (TP): 预测为i且真实为i
    TP = torch.zeros(num_classes, dtype=torch.long, device=device)
    for i in range(num_classes):
        TP[i] = ((valid_preds == i) & (valid_labels == i)).sum()

    # False Positives (FP): 预测为i但真实不是i
    FP = torch.zeros(num_classes, dtype=torch.long, device=device)
    for i in range(num_classes):
        FP[i] = ((valid_preds == i) & (valid_labels != i)).sum()

    # False Negatives (FN): 真实为i但预测不是i
    FN = torch.zeros(num_classes, dtype=torch.long, device=device)
    for i in range(num_classes):
        FN[i] = ((valid_preds != i) & (valid_labels == i)).sum()

    # Recall, Precision, F1
    recall = torch.zeros(num_classes, dtype=torch.float, device=device)
    precision = torch.zeros(num_classes, dtype=torch.float, device=device)
    f1 = torch.zeros(num_classes, dtype=torch.float, device=device)
    for i in range(num_classes):
        recall[i] = (
            TP[i].float() / (TP[i] + FN[i]).float()
            if (TP[i] + FN[i]) > 0
            else torch.tensor(0.0, device=device)
        )
        precision[i] = (
            TP[i].float() / (TP[i] + FP[i]).float()
            if (TP[i] + FP[i]) > 0
            else torch.tensor(0.0, device=device)
        )
        if recall[i] + precision[i] > 0:
            f1[i] = 2 * recall[i] * precision[i] / (recall[i] + precision[i])
        else:
            f1[i] = torch.tensor(0.0, device=device)
    # 只计算bin1~num_classes-1的macro_f1
    macro_f1 = f1[1:].mean().item()
    average_recall = recall[1:].mean().item()
    average_precision = precision[1:].mean().item()

    return recall, precision, f1, macro_f1, average_recall, average_precision


def get_l0_lambda(epoch, index, l0_warmup_start_epoch, l0_lambda_target, epoch_length):
    """
    计算l0 lambda，使用warmup策略
    """
    if epoch < l0_warmup_start_epoch:
        return 0.0
    elif epoch > l0_warmup_start_epoch:
        return l0_lambda_target
    else:
        return calculate_l0_lambda_with_warmup_fine_grained(
            index, l0_lambda_target, epoch_length
        )


def calculate_l0_lambda_with_warmup_fine_grained(index, l0_lambda_target, epoch_length):
    """
    计算当前的l0 lambda 按index在l0_warmup_start_epoch到l0_warmup_start_epoch + 1之间线性增加到l0_lambda_target
    """
    return l0_lambda_target * (index / epoch_length)


def count_unique_cell_types(h5ad_path, cell_type_col="cell_type"):
    """
    统计 h5ad 文件中 obs 的 cell_type 列的唯一值数量

    Args:
        h5ad_path (str): h5ad 文件路径
        cell_type_col (str): obs 中细胞类型列名（默认 "cell_type"）

    Returns:
        tuple: (count, names) - 唯一 cell_type 的数量和名称列表
    """
    adata = sc.read_h5ad(h5ad_path)

    if cell_type_col not in adata.obs.columns:
        raise ValueError(f"obs 中不存在列: {cell_type_col}")

    unique_celltypes = sorted(adata.obs[cell_type_col].astype(str).unique())

    print(f"Found {len(unique_celltypes)} unique cell types in {h5ad_path}:")
    for i, celltype in enumerate(unique_celltypes):
        print(f"  {i}: {celltype}")

    return len(unique_celltypes), unique_celltypes


def count_unique_cell_types_from_multiple_files(*h5ad_paths, cell_type_col="cell_type"):
    """
    统计多个h5ad文件中所有unique的cell_type数量（并集）

    Args:
        *h5ad_paths: 多个h5ad文件路径
        cell_type_col (str): obs中细胞类型列名

    Returns:
        int: 所有文件中唯一cell_type的总数量
        list: 所有unique的cell_type名称列表（按字母顺序排序）
    """
    all_celltypes = set()

    # 从所有h5ad文件中收集celltype
    for h5ad_path in h5ad_paths:
        adata = sc.read_h5ad(h5ad_path)

        if cell_type_col not in adata.obs.columns:
            raise ValueError(f"文件 {h5ad_path} 的obs中不存在列: {cell_type_col}")

        celltypes = adata.obs[cell_type_col].astype(str).unique()
        all_celltypes.update(celltypes)

    # 按字母顺序排序以确保稳定的映射
    sorted_celltypes = sorted(all_celltypes)

    print(
        f"Found {len(sorted_celltypes)} unique cell types across {len(h5ad_paths)} files:"
    )
    for i, celltype in enumerate(sorted_celltypes):
        print(f"  {i}: {celltype}")

    return len(sorted_celltypes), sorted_celltypes


def count_common_cell_types_from_multiple_files(*h5ad_paths, cell_type_col="cell_type"):
    """
    统计多个h5ad文件中共同的cell_type数量（交集）

    Args:
        *h5ad_paths: 多个h5ad文件路径
        cell_type_col (str): obs中细胞类型列名

    Returns:
        int: 所有文件共同的cell_type数量
        list: 共同的cell_type名称列表（按字母顺序排序）
    """
    if not h5ad_paths:
        return 0, []

    # 读取第一个文件的细胞类型作为初始集合
    first_adata = sc.read_h5ad(h5ad_paths[0])
    if cell_type_col not in first_adata.obs.columns:
        raise ValueError(f"文件 {h5ad_paths[0]} 的obs中不存在列: {cell_type_col}")

    common_celltypes = set(first_adata.obs[cell_type_col].astype(str).unique())

    # 与其他文件的细胞类型求交集
    for h5ad_path in h5ad_paths[1:]:
        adata = sc.read_h5ad(h5ad_path)
        if cell_type_col not in adata.obs.columns:
            raise ValueError(f"文件 {h5ad_path} 的obs中不存在列: {cell_type_col}")

        file_celltypes = set(adata.obs[cell_type_col].astype(str).unique())
        common_celltypes &= file_celltypes  # 求交集

    # 按字母顺序排序以确保稳定的映射
    sorted_common_celltypes = sorted(common_celltypes)

    print(
        f"Found {len(sorted_common_celltypes)} common cell types across {len(h5ad_paths)} files:"
    )
    for i, celltype in enumerate(sorted_common_celltypes):
        print(f"  {i}: {celltype}")

    return len(sorted_common_celltypes), sorted_common_celltypes


def extract_state_dict(maybe_state):
    """
    兼容多种保存方式：
    - {"model": state_dict, ...}  ← 你现在的保存方式（Fabric）
    - {"state_dict": state_dict, ...}  ← Lightning 常见
    - 直接就是 state_dict
    - 键带 "model." 前缀
    """
    if isinstance(maybe_state, dict):
        if "model" in maybe_state and isinstance(maybe_state["model"], dict):
            sd = maybe_state["model"]
        elif "state_dict" in maybe_state and isinstance(
            maybe_state["state_dict"], dict
        ):
            sd = maybe_state["state_dict"]
        else:
            # 可能就是 state_dict
            sd = maybe_state
    else:
        raise ValueError("Checkpoint 内容不是字典，无法解析 state_dict")

    # 去掉可能存在的前缀 "model." 或 "module."
    need_strip_prefixes = ("model.", "module.")
    if any(any(k.startswith(p) for p in need_strip_prefixes) for k in sd.keys()):
        new_sd = {}
        for k, v in sd.items():
            for p in need_strip_prefixes:
                if k.startswith(p):
                    k = k[len(p) :]
                    break
            new_sd[k] = v
        sd = new_sd
    return sd


def extract_state_dict_with_encoder_prefix(maybe_state):
    """
    专门处理模型结构中有 encoder. 前缀，但预训练权重没有此前缀的情况。

    Args:
        maybe_state: 加载的checkpoint，可能是dict或直接的state_dict

    Returns:
        dict: 处理过前缀的state_dict，匹配当前模型结构
    """
    # 先用原有函数提取基本的state_dict
    sd = extract_state_dict(maybe_state)

    # 检查是否需要添加encoder前缀
    # 如果state_dict中的键没有encoder前缀，但我们需要encoder前缀
    has_encoder_prefix = any(k.startswith("encoder.") for k in sd.keys())

    if not has_encoder_prefix:
        # 需要为所有键添加encoder.前缀
        new_sd = {}
        for k, v in sd.items():
            new_key = f"encoder.{k}"
            new_sd[new_key] = v
        print(f"[LOAD] 为 {len(sd)} 个参数添加了 'encoder.' 前缀")
        return new_sd
    else:
        print("[LOAD] 检测到权重已有 'encoder.' 前缀，直接返回")
        return sd


def extract_state_dict_remove_encoder_prefix(maybe_state):
    """
    专门处理模型结构中没有 encoder. 前缀，但权重有此前缀的情况。

    Args:
        maybe_state: 加载的checkpoint，可能是dict或直接的state_dict

    Returns:
        dict: 处理过前缀的state_dict，匹配当前模型结构
    """
    # 先用原有函数提取基本的state_dict
    sd = extract_state_dict(maybe_state)

    # 检查是否需要移除encoder前缀
    has_encoder_prefix = any(k.startswith("encoder.") for k in sd.keys())

    if has_encoder_prefix:
        # 需要移除encoder.前缀
        new_sd = {}
        for k, v in sd.items():
            if k.startswith("encoder."):
                new_key = k[len("encoder.") :]  # 移除"encoder."前缀
                new_sd[new_key] = v
            else:
                new_sd[k] = v
        print(
            f"[LOAD] 为 {len([k for k in sd.keys() if k.startswith('encoder.')])} 个参数移除了 'encoder.' 前缀"
        )
        return new_sd
    else:
        print("[LOAD] 检测到权重没有 'encoder.' 前缀，直接返回")
        return sd


def report_loading_result(load_info):
    missing = list(load_info.missing_keys)
    unexpected = list(load_info.unexpected_keys)
    print(f"[LOAD] missing_keys: {len(missing)} | unexpected_keys: {len(unexpected)}")
    if missing:
        print("  ´ missing:", missing)
    if unexpected:
        print("   unexpected:", unexpected)


def sample_weight_norms(model, sd, k=5):
    """
    随机抽样 k 个双方都存在的参数，打印加载前后的范数变化。
    范数有变化 => 基本可确认成功写入。
    """
    with torch.no_grad():
        common_keys = [name for name, _ in model.named_parameters() if name in sd]
        if not common_keys:
            print("[LOAD] 没有找到与 checkpoint 对齐的公共参数名，无法做范数对比。")
            return
        sample = random.sample(common_keys, min(k, len(common_keys)))
        print("[LOAD] 抽样参数范数对比（加载前 -> 加载后）：")
        for name in sample:
            p = dict(model.named_parameters())[name]
            before = p.detach().float().norm().item()
            # 暂存当前权重
            old = p.detach().cpu().clone()
            # 用 ckpt 覆盖一次
            p.copy_(sd[name].to(p.device).to(p.dtype))
            after = p.detach().float().norm().item()
            print(f"  - {name}: {before:.6f} -> {after:.6f}")
            # 还原（只用于对比；真正的加载在 load_state_dict 里会再做一次）
            p.copy_(old.to(p.device).to(p.dtype))


def draw_expr_emb_analysis(E, epoch, ckpt_dir, iteration=0):
    """
    Draw t-SNE and UMAP visualization for expression embeddings.

    Args:
        E: Expression embeddings tensor
        epoch: Current epoch number
        ckpt_dir: Checkpoint directory for saving plots
        is_master: Whether this is the master process
        iteration: Current iteration number
    """
    # t-SNE visualization
    try:
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE

        tsne = TSNE(n_components=2, random_state=0, perplexity=30, n_iter=1000)
        E_np = E.cpu().numpy()
        E_tsne = tsne.fit_transform(E_np)
        plt.figure(figsize=(6, 6))
        plt.scatter(E_tsne[:, 0], E_tsne[:, 1], s=2, alpha=0.5)
        plt.title(f"expr_emb t-SNE (epoch {epoch}, iteration {iteration})")
        plt.tight_layout()
        tsne_dir = os.path.join(ckpt_dir, "tsne_vis")
        os.makedirs(tsne_dir, exist_ok=True)
        plt.savefig(
            os.path.join(
                tsne_dir,
                f"expr_emb_tsne_epoch{epoch}_iteration{iteration}.png",
            )
        )
        tsne_path = os.path.join(
            tsne_dir, f"expr_emb_tsne_epoch{epoch}_iteration{iteration}.png"
        )
        logging.info(f"[Embedding Analysis] t-SNE plot saved:\n  {tsne_path}")
        # New: upload t-SNE image to wandb
        wandb.log(
            {
                "tsne": wandb.Image(tsne_path),
            }
        )
        plt.close()

        # New: UMAP visualization
        import umap

        reducer = umap.UMAP(n_components=2, random_state=0)
        E_umap = reducer.fit_transform(E_np)
        plt.figure(figsize=(6, 6))
        plt.scatter(E_umap[:, 0], E_umap[:, 1], s=2, alpha=0.5)
        plt.title(f"expr_emb UMAP (epoch {epoch}, iteration {iteration})")
        plt.tight_layout()
        umap_path = os.path.join(
            tsne_dir, f"expr_emb_umap_epoch{epoch}_iteration{iteration}.png"
        )
        plt.savefig(umap_path)
        wandb.log(
            {
                "umap": wandb.Image(umap_path),
            }
        )
        plt.close()
        logging.info("[Embedding Analysis] t-SNE and UMAP plots saved")
    except Exception as e:
        logging.error(f"[Embedding Analysis] t-SNE failed: {e}")


def draw_continuous_pred_label_scatter(
    all_masked_preds, all_masked_labels, epoch, ckpt_dir, iteration=0
):
    """
    Draw scatter plot of predictions vs labels for continuous values.

    Args:
        all_masked_preds: List of prediction tensors
        all_masked_labels: List of label tensors
        epoch: Current epoch number
        ckpt_dir: Checkpoint directory for saving plots
        is_master: Whether this is the master process
        iteration: Current iteration number
    """
    # --------- New: draw pred-label scatter plot and upload to wandb (only draw once at the end of validate)
    if len(all_masked_preds) > 0:
        import matplotlib.pyplot as plt
        import torch

        preds = torch.cat(all_masked_preds, dim=0).numpy().flatten()
        labels = torch.cat(all_masked_labels, dim=0).numpy().flatten()
        plt.figure(figsize=(6, 6))
        plt.scatter(labels, preds, s=2, alpha=0.5)
        plt.xlabel("Label")
        plt.ylabel("Prediction")
        plt.title(f"Pred vs Label (epoch {epoch}, iter {iteration})")
        plt.tight_layout()
        scatter_dir = os.path.join(ckpt_dir, "scatter_vis")
        os.makedirs(scatter_dir, exist_ok=True)
        scatter_path = os.path.join(
            scatter_dir, f"pred_vs_label_epoch{epoch}_iter{iteration}.png"
        )
        plt.savefig(scatter_path)
        wandb.log(
            {
                "pred_vs_label_scatter": wandb.Image(scatter_path),
            }
        )
        plt.close()


def check_moe_collapse(model, epoch, iteration):
    """
    检查MoE塌缩情况并记录到日志

    Args:
        model: The model to check for MoE collapse
        epoch: current epoch
        iteration: current iteration
    """
    try:
        # 检查模型是否有MoE塌缩检测功能
        if not hasattr(model, "check_moe_collapse"):
            return

        print(f"\n[Epoch {epoch}, Iter {iteration}] 检查MoE塌缩状态...")

        # 获取塌缩检测结果
        collapse_results = model.check_moe_collapse(threshold=0.8)

        if not collapse_results:
            logging.info("No MoE layers found or MoE function not enabled")
            return

        # 统计塌缩情况
        total_layers = len(collapse_results)
        collapsed_layers = sum(
            1 for result in collapse_results.values() if result["is_collapsed"]
        )
        healthy_layers = total_layers - collapsed_layers

        # Log to console
        logging.info(
            f"MoE status summary: total_layers={total_layers}, "
            f"collapsed_layers={collapsed_layers}, healthy_layers={healthy_layers}"
        )

        # If there's collapse, print detailed report
        if collapsed_layers > 0:
            logging.warning("⚠️  MoE collapse detected! Detailed information:")
            for layer_name, result in collapse_results.items():
                if result["is_collapsed"]:
                    logging.warning(
                        f"  🚨 {layer_name}: collapse_ratio={result['collapse_ratio']:.4f}, "
                        f"entropy={result['entropy']:.4f}"
                    )

                    # Find the most used expert
                    usage_ratios = result["expert_usage_ratio"]
                    max_expert_idx = usage_ratios.index(max(usage_ratios))
                    logging.warning(
                        f"     Most used expert: Expert-{max_expert_idx} "
                        f"(usage_rate: {usage_ratios[max_expert_idx]:.4f})"
                    )
        else:
            logging.info("✅ All MoE layers are healthy")

        # Log to wandb (if enabled)
        if wandb.run is not None:
            wandb_logs = {
                "moe/total_layers": total_layers,
                "moe/collapsed_layers": collapsed_layers,
                "moe/healthy_layers": healthy_layers,
                "moe/collapse_ratio": (
                    collapsed_layers / total_layers if total_layers > 0 else 0.0
                ),
            }

            # Log detailed information for each layer
            for layer_name, result in collapse_results.items():
                layer_key = layer_name.replace("/", "_").replace("-", "_")
                wandb_logs[f"moe_layers/{layer_key}/collapse_ratio"] = result[
                    "collapse_ratio"
                ]
                wandb_logs[f"moe_layers/{layer_key}/entropy"] = result["entropy"]
                wandb_logs[f"moe_layers/{layer_key}/is_collapsed"] = int(
                    result["is_collapsed"]
                )

            wandb.log(wandb_logs, step=iteration)

        logging.info(
            f"MoE collapse detection completed [Epoch {epoch}, Iter {iteration}]\n"
        )

    except Exception as e:
        logging.error(f"MoE collapse detection error: {e}")
        import traceback

        traceback.print_exc()
