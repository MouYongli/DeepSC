import logging
import math
import os
import os.path as osp
import random
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import scanpy as sc
import torch
import torch.nn.functional as F
from anndata import AnnData
from torch import nn
from torch.nn.modules.loss import _WeightedLoss
from torch.optim.lr_scheduler import _LRScheduler

import sys
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
    use_hydra: bool = True,
) -> str:
    """
    Setup unified logging configuration.

    Args:
        log_path: Directory to store log files
        log_name: Base name for the log file
        rank: Process rank for distributed training (-1 for single process)
        add_timestamp: Whether to add timestamp to log filename
        log_level: Logging level
        use_hydra: Whether running under Hydra (will only redirect stdout/stderr)

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

    # Only setup logging handlers if not using Hydra
    if not use_hydra:
        # Set logging level based on rank
        if rank in [-1, 0]:
            level = getattr(logging, log_level.upper())
        else:
            level = logging.WARN

        # Configure logging with file handler
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s",
                datefmt="[%X]",
            )
        )

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)
        stream_handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s",
                datefmt="[%X]",
            )
        )

        logging.basicConfig(
            level=level,
            handlers=[file_handler, stream_handler],
            force=True,  # Reset any existing logging configuration
        )

        logger = logging.getLogger()
        logger.info(f"Log file initialized: {log_file}")

    # Redirect stdout and stderr to log file (only for master rank)
    if rank in [-1, 0]:

        class TeeOutput:
            """Redirect print output to both console and log file"""

            def __init__(self, file_path, original_stream):
                self.file = open(file_path, "a" if use_hydra else "w", buffering=1)
                self.original = original_stream

            def write(self, message):
                self.file.write(message)
                self.file.flush()
                self.original.write(message)
                self.original.flush()

            def flush(self):
                self.file.flush()
                self.original.flush()

        sys.stdout = TeeOutput(log_file, sys.__stdout__)
        sys.stderr = TeeOutput(log_file, sys.__stderr__)

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
            "chunk_idx": chunk_idx,  # Save chunk index
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
        chunk_idx=chunk_idx,  # Save chunk index
    )


def get_reduced(tensor, current_device, dest_device, world_size):
    """
    Gather variables or tensors from different GPUs to the main GPU and get the mean
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
    Gather variables or tensors from different GPUs to the main GPU and get the mean, needs to be 2D tensor
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
    Merge inference results from different processes
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
    Calculate MSE Loss only at positions where mask=True.

    Args:
        pred (Tensor): model output, shape = (B, T)
        target (Tensor): target value, shape = (B, T)
        mask (BoolTensor): mask, True indicates positions to calculate
        reduction (str): "mean", "sum" or "none"

    Returns:
        Tensor: loss value
    """
    loss_fn = loss_fn
    elementwise_loss = loss_fn(pred, target)  # shape: (B, T)

    masked_loss = elementwise_loss[mask]  # Only take masked parts

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
    Partitioned weighted MSE Loss, calculated only at positions where mask=True.
    Weight ranges:
        target == 1       → 0.2
        1 < target < 3    → 0.5
        3 <= target < 5   → 1.0
        target >= 5       → 2.0
    Normalized after weighting to prevent gradient explosion or collapse.
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
    Base-2 exponentially weighted MSE Loss, calculated only at positions where mask=True.
    weight = 2^(target - shift)
    shift can adjust weight starting point, default is 0.
    """
    loss_fn = loss_fn
    elementwise_loss = loss_fn(pred, target)  # shape: (B, T)

    # Generate base-2 exponential weights
    weights = torch.pow(1.8, target - 1)

    # Calculate only where mask=True
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
        # Upload recall, precision, f1 for each category to wandb
        if "wandb" in globals() and wandb.run is not None:
            log_dict = {}
            for i in range(num_classes):
                log_dict[f"recall/bin{i}"] = recall[i].item()
                log_dict[f"precision/bin{i}"] = precision[i].item()
                log_dict[f"f1/bin{i}"] = f1[i].item()
            log_dict["val/macro_f1"] = macro_f1
            # Average only for non-class-0 (bin0)
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
    Calculate prediction distribution.
    Args:
        final: Predicted classes (tensor)
        valid_mask: Valid position mask (tensor, bool)
        num_bins: Number of bins (int)
        topk: If None, return distribution of first num_bins bins; otherwise return top-k bins with highest prediction ratios, their indices and ratios
    Returns:
        If topk is None, return [(bin_number, ratio), ...] of length num_bins
        Otherwise, return [(bin_number, ratio), ...] of length topk
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
            # No warmup, use cosine annealing directly
            return [
                self.eta_min
                + (base_lr - self.eta_min)
                * (1 + math.cos(math.pi * self.T_cur / self.T_i))
                / 2
                for base_lr in self.base_lrs
            ]
        else:
            # Has warmup, use original logic
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


class LDAMLoss(nn.Module):
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30, ignore_index=-100):
        super(LDAMLoss, self).__init__()
        # Exclude padding class (class 0), only compute margin for valid classes
        valid_cls_num_list = cls_num_list[1:]  # Exclude class 0
        m_list = 1.0 / np.sqrt(np.sqrt(valid_cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        # Set margin to 0 for padding class (class 0)
        self.m_list = torch.zeros(len(cls_num_list), dtype=torch.float32)
        self.m_list[1:] = torch.tensor(
            m_list, dtype=torch.float32
        )  # Class 0 margin is 0
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
    Check if model gradient flow is normal.
    Args:
        model: nn.Module
        loss_tensor: loss
        verbose: Whether to print detailed info
        retain_graph: Whether to retain computation graph
        backward_fn: Custom backward function (e.g. fabric.backward)
    """
    print("=" * 60)
    print("➡️ [Check Start] Gradient flow during backpropagation...")

    # Save original gradient state
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
        print(f"[ERROR] Backpropagation failed: {e}")
        # Restore original gradients
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
            print(f"[ERROR] Error checking gradient for parameter {name}: {e}")
            no_grad_names.append(name)

    print("-" * 60)
    print(f"✅ Parameters with valid gradients: {len(ok_grad_names)}")
    print(f"⚠️ Parameters with zero gradients: {len(zero_grad_names)}")
    print(f"❌ Parameters with None gradients: {len(no_grad_names)}")
    print("=" * 60)

    # Restore original gradients
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
    Compute gating matrix M from Gumbel Softmax output y
    Args:
        y: Gumbel Softmax output, shape: (batch, g, g, 3)
    Returns:
        M: Gating matrix, shape: (batch, g, g)
    """
    return y[..., 0] * (-1) + y[..., 1] * 0 + y[..., 2] * (+1)


def print_m_matrix(epoch, index, M):
    print(f"\n=== Epoch {epoch}, Iteration {index}: M Matrix ===")
    # Only print M matrix of first sample in first batch
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


# Compute TP, FP, FN for each class
def compute_classification_metrics(valid_preds, valid_labels, num_classes, device):
    """
    Compute TP, FP, FN, recall, precision, f1, macro_f1 for each class
    Returns: recall, precision, f1, macro_f1
    Note: Does not compute metrics for bin0 (class 0)
    """
    # True Positives (TP): predicted as i and actually is i
    TP = torch.zeros(num_classes, dtype=torch.long, device=device)
    for i in range(num_classes):
        TP[i] = ((valid_preds == i) & (valid_labels == i)).sum()

    # False Positives (FP): predicted as i but actually not i
    FP = torch.zeros(num_classes, dtype=torch.long, device=device)
    for i in range(num_classes):
        FP[i] = ((valid_preds == i) & (valid_labels != i)).sum()

    # False Negatives (FN): actually is i but predicted as not i
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
    # Only compute macro_f1 for bin1~num_classes-1
    macro_f1 = f1[1:].mean().item()
    average_recall = recall[1:].mean().item()
    average_precision = precision[1:].mean().item()

    return recall, precision, f1, macro_f1, average_recall, average_precision


def count_unique_cell_types(h5ad_path, cell_type_col="cell_type"):
    """
    Count unique cell types in h5ad file's obs cell_type column

    Args:
        h5ad_path (str): Path to h5ad file
        cell_type_col (str): Cell type column name in obs (default "cell_type")

    Returns:
        tuple: (count, names) - Number of unique cell types and list of names
    """
    adata = sc.read_h5ad(h5ad_path)

    if cell_type_col not in adata.obs.columns:
        raise ValueError(f"Column does not exist in obs: {cell_type_col}")

    unique_celltypes = sorted(adata.obs[cell_type_col].astype(str).unique())

    print(f"Found {len(unique_celltypes)} unique cell types in {h5ad_path}:")
    for i, celltype in enumerate(unique_celltypes):
        print(f"  {i}: {celltype}")

    return len(unique_celltypes), unique_celltypes


def count_unique_cell_types_from_multiple_files(*h5ad_paths, cell_type_col="cell_type"):
    """
    Count all unique cell types across multiple h5ad files (union)

    Args:
        *h5ad_paths: Multiple h5ad file paths
        cell_type_col (str): Cell type column name in obs

    Returns:
        int: Total number of unique cell types across all files
        list: List of all unique cell type names (sorted alphabetically)
    """
    all_celltypes = set()

    # Collect celltypes from all h5ad files
    for h5ad_path in h5ad_paths:
        adata = sc.read_h5ad(h5ad_path)

        if cell_type_col not in adata.obs.columns:
            raise ValueError(
                f"Column does not exist in obs of file {h5ad_path}: {cell_type_col}"
            )

        celltypes = adata.obs[cell_type_col].astype(str).unique()
        all_celltypes.update(celltypes)

    # Sort alphabetically to ensure stable mapping
    sorted_celltypes = sorted(all_celltypes)

    print(
        f"Found {len(sorted_celltypes)} unique cell types across {len(h5ad_paths)} files:"
    )
    for i, celltype in enumerate(sorted_celltypes):
        print(f"  {i}: {celltype}")

    return len(sorted_celltypes), sorted_celltypes


def count_common_cell_types_from_multiple_files(*h5ad_paths, cell_type_col="cell_type"):
    """
    Count common cell types across multiple h5ad files (intersection)

    Args:
        *h5ad_paths: Multiple h5ad file paths
        cell_type_col (str): Cell type column name in obs

    Returns:
        int: Number of common cell types across all files
        list: List of common cell type names (sorted alphabetically)
    """
    if not h5ad_paths:
        return 0, []

    # Read cell types from first file as initial set
    first_adata = sc.read_h5ad(h5ad_paths[0])
    if cell_type_col not in first_adata.obs.columns:
        raise ValueError(
            f"Column does not exist in obs of file {h5ad_paths[0]}: {cell_type_col}"
        )

    common_celltypes = set(first_adata.obs[cell_type_col].astype(str).unique())

    # Compute intersection with cell types from other files
    for h5ad_path in h5ad_paths[1:]:
        adata = sc.read_h5ad(h5ad_path)
        if cell_type_col not in adata.obs.columns:
            raise ValueError(
                f"Column does not exist in obs of file {h5ad_path}: {cell_type_col}"
            )

        file_celltypes = set(adata.obs[cell_type_col].astype(str).unique())
        common_celltypes &= file_celltypes  # Intersection

    # Sort alphabetically to ensure stable mapping
    sorted_common_celltypes = sorted(common_celltypes)

    print(
        f"Found {len(sorted_common_celltypes)} common cell types across {len(h5ad_paths)} files:"
    )
    for i, celltype in enumerate(sorted_common_celltypes):
        print(f"  {i}: {celltype}")

    return len(sorted_common_celltypes), sorted_common_celltypes


def extract_state_dict(maybe_state):
    """
    Compatible with various save formats:
    - {"model": state_dict, ...}  ← Current save format (Fabric)
    - {"state_dict": state_dict, ...}  ← Common in Lightning
    - Direct state_dict
    - Keys with "model." prefix
    """
    if isinstance(maybe_state, dict):
        if "model" in maybe_state and isinstance(maybe_state["model"], dict):
            sd = maybe_state["model"]
        elif "state_dict" in maybe_state and isinstance(
            maybe_state["state_dict"], dict
        ):
            sd = maybe_state["state_dict"]
        else:
            # Possibly a direct state_dict
            sd = maybe_state
    else:
        raise ValueError(
            "Checkpoint content is not a dictionary, cannot parse state_dict"
        )

    # Remove possible prefixes "model." or "module."
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
    Handle case where model structure has encoder. prefix but pretrained weights don't.

    Args:
        maybe_state: Loaded checkpoint, may be dict or direct state_dict

    Returns:
        dict: Processed state_dict with prefix matching current model structure
    """
    # Extract basic state_dict using existing function
    sd = extract_state_dict(maybe_state)

    # Check if encoder prefix needs to be added
    # If keys in state_dict don't have encoder prefix, but we need it
    has_encoder_prefix = any(k.startswith("encoder.") for k in sd.keys())

    if not has_encoder_prefix:
        # Need to add encoder. prefix to all keys
        new_sd = {}
        for k, v in sd.items():
            new_key = f"encoder.{k}"
            new_sd[new_key] = v
        print(f"[LOAD] Added 'encoder.' prefix to {len(sd)} parameters")
        return new_sd
    else:
        print(
            "[LOAD] Detected weights already have 'encoder.' prefix, returning directly"
        )
        return sd


def extract_state_dict_remove_encoder_prefix(maybe_state):
    """
    Handle case where model structure doesn't have encoder. prefix but weights do.

    Args:
        maybe_state: Loaded checkpoint, may be dict or direct state_dict

    Returns:
        dict: Processed state_dict with prefix matching current model structure
    """
    # Extract basic state_dict using existing function
    sd = extract_state_dict(maybe_state)

    # Check if encoder prefix needs to be removed
    has_encoder_prefix = any(k.startswith("encoder.") for k in sd.keys())

    if has_encoder_prefix:
        new_sd = {}
        for k, v in sd.items():
            if k.startswith("encoder."):
                new_key = k[len("encoder.") :]  # Remove "encoder." prefix
                new_sd[new_key] = v
            else:
                new_sd[k] = v
        print(
            f"[LOAD] Removed 'encoder.' prefix from {len([k for k in sd.keys() if k.startswith('encoder.')])} parameters"
        )
        return new_sd
    else:
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

    with torch.no_grad():
        common_keys = [name for name, _ in model.named_parameters() if name in sd]
        if not common_keys:
            print(
                "[LOAD] No common parameter names found matching checkpoint, cannot compare norms."
            )
            return
        sample = random.sample(common_keys, min(k, len(common_keys)))
        print("[LOAD] Sampled parameter norm comparison (before -> after loading):")
        for name in sample:
            p = dict(model.named_parameters())[name]
            before = p.detach().float().norm().item()
            # Store current weights temporarily
            old = p.detach().cpu().clone()
            # Overwrite with checkpoint once
            p.copy_(sd[name].to(p.device).to(p.dtype))
            after = p.detach().float().norm().item()
            print(f"  - {name}: {before:.6f} -> {after:.6f}")
            # Restore (only for comparison; actual loading happens in load_state_dict)
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


def load_checkpoint(
    ckpt_file_path,
    model,
    optimizer=None,
    scheduler=None,
    fabric=None,
    is_master=True,
    resume_training=True,
):
    """
    Load checkpoint from file and restore model, optimizer, scheduler states.

    Args:
        ckpt_file_path: Path to the checkpoint file
        model: Model to restore
        optimizer: Optimizer to restore (optional)
        scheduler: Scheduler to restore (optional)
        fabric: Fabric instance for distributed training (optional)
        is_master: Whether this is the master process
        resume_training: Whether to resume training state (epoch, iteration, etc.)

    Returns:
        dict: Dictionary containing checkpoint information:
            - 'loaded': bool, whether checkpoint was loaded successfully
            - 'epoch': int, last epoch (if resume_training=True)
            - 'iteration': int, last iteration (if resume_training=True)
            - 'chunk_idx': int, last chunk index (if resume_training=True)
            - 'wandb_run_id': str, wandb run id (if available)
            - 'wandb_config': dict, wandb config (if available)
    """
    if not os.path.exists(ckpt_file_path):
        if is_master:
            logging.warning(f"[WARN] Checkpoint not found: {ckpt_file_path}")
        return {
            "loaded": False,
            "epoch": 1,
            "iteration": 0,
            "chunk_idx": 0,
            "wandb_run_id": None,
            "wandb_config": None,
        }

    try:
        # Load state dict
        if fabric is not None:
            remainder = fabric.load(ckpt_file_path)
        else:
            remainder = torch.load(ckpt_file_path, map_location="cpu")

        # Restore model, optimizer, scheduler parameters
        if "model" in remainder:
            model.load_state_dict(remainder["model"])
        if "optimizer" in remainder and optimizer is not None:
            optimizer.load_state_dict(remainder["optimizer"])
        if (
            "scheduler" in remainder
            and scheduler is not None
            and remainder["scheduler"] is not None
        ):
            scheduler.load_state_dict(remainder["scheduler"])

        # Prepare return values
        result = {
            "loaded": True,
            "epoch": remainder.get("epoch", 1) if resume_training else 1,
            "iteration": remainder.get("iteration", 0) if resume_training else 0,
            "chunk_idx": remainder.get("chunk_idx", 0) if resume_training else 0,
            "wandb_run_id": remainder.get("wandb_run_id", None),
            "wandb_config": remainder.get("wandb_config", None),
        }

        if is_master:
            logging.info(f"[INFO] Checkpoint loaded successfully from {ckpt_file_path}")
            if resume_training:
                logging.info(
                    f"[INFO] Resume training from epoch={result['epoch']}, "
                    f"chunk={result['chunk_idx']}, iter={result['iteration']}"
                )

        return result

    except Exception as e:
        if is_master:
            logging.error(
                f"[ERROR] Failed to load checkpoint from {ckpt_file_path}: {e}"
            )
        return {
            "loaded": False,
            "epoch": 1,
            "iteration": 0,
            "chunk_idx": 0,
            "wandb_run_id": None,
            "wandb_config": None,
        }


def load_checkpoint_cta_test(checkpoint_path, model, fabric, is_master=True):
    """
    Load model checkpoint for Cell Type Annotation testing.

    This function loads only the model weights and cell type mappings,
    without loading optimizer/scheduler states (unlike load_checkpoint).

    Args:
        checkpoint_path: Path to the checkpoint file
        model: Model to load weights into
        fabric: Fabric instance for distributed loading
        is_master: Whether this is the master process

    Returns:
        dict: Dictionary containing:
            - cell_type_count: Number of cell types
            - type2id: Mapping from cell type name to ID
            - id2type: Mapping from ID to cell type name
            - common_celltypes: Sorted list of cell type names
            - epoch: Training epoch when checkpoint was saved
            - eval_loss: Evaluation loss when checkpoint was saved
    """
    assert os.path.exists(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"

    if fabric.global_rank == 0:
        print(f"[LOAD] Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    else:
        checkpoint = None

    # Broadcast checkpoint to all processes
    checkpoint = fabric.broadcast(checkpoint, src=0)

    # Load model state dict
    load_info = model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    if is_master:
        print(f"Checkpoint loaded successfully from epoch {checkpoint['epoch']}")
        print(f"Checkpoint eval loss: {checkpoint['eval_loss']:.4f}")
        report_loading_result(load_info)

    # Extract cell type mappings from checkpoint
    cell_type_count = checkpoint["cell_type_count"]
    type2id = checkpoint["type2id"]
    id2type = checkpoint["id2type"]
    common_celltypes = sorted(type2id.keys())

    print(f"Loaded cell type count: {cell_type_count}")
    print(f"Loaded cell types: {common_celltypes}")

    return {
        "cell_type_count": cell_type_count,
        "type2id": type2id,
        "id2type": id2type,
        "common_celltypes": common_celltypes,
        "epoch": checkpoint["epoch"],
        "eval_loss": checkpoint["eval_loss"],
    }


def restore_wandb_session(wandb_run_id, wandb_config, args, is_master=True):
    """
    Restore wandb session from checkpoint information.

    Args:
        wandb_run_id: Saved wandb run ID
        wandb_config: Saved wandb configuration
        args: Training arguments
        is_master: Whether this is the master process

    Returns:
        bool: Whether wandb session was restored successfully
    """
    if not is_master or wandb_run_id is None:
        return False

    try:
        logging.info(f"[INFO] Found saved wandb run_id: {wandb_run_id}")
        logging.info("[INFO] Using original run_id to restore wandb session...")

        if wandb_config:
            wandb.init(
                id=wandb_run_id,
                resume="allow",
                project=wandb_config.get("project", args.logging.wandb_project),
                entity=wandb_config.get("entity", args.logging.wandb_team),
                name=wandb_config.get(
                    "name",
                    f"{args.logging.run_name}, lr: {args.learning_rate}",
                ),
                tags=wandb_config.get("tags", args.logging.tags),
                config=wandb_config.get("config", dict(args)),
            )
        else:
            wandb.init(
                id=wandb_run_id,
                resume="allow",
                project=args.logging.wandb_project,
                entity=args.logging.wandb_team,
                name=f"{args.logging.run_name}, lr: {args.learning_rate}",
                tags=args.logging.tags,
                config=dict(args),
            )

        logging.info(
            f"✅ Wandb restored! Project: {wandb.run.project}, Entity: {wandb.run.entity}, Run ID: {wandb.run.id}"
        )
        logging.info(f"🔗 Wandb URL: {wandb.run.url}")
        return True

    except Exception as e:
        logging.error(f"[ERROR] Failed to restore wandb session: {e}")
        return False


def check_moe_collapse(model, epoch, iteration):
    """
    Check MoE collapse and log to console

    Args:
        model: The model to check for MoE collapse
        epoch: current epoch
        iteration: current iteration
    """
    try:
        # Check if model has MoE collapse detection capability
        if not hasattr(model, "check_moe_collapse"):
            return

        print(f"\n[Epoch {epoch}, Iter {iteration}] Checking MoE collapse status...")

        # Get collapse detection results
        collapse_results = model.check_moe_collapse(threshold=0.8)

        if not collapse_results:
            logging.info("No MoE layers found or MoE function not enabled")
            return

        # Count collapse statistics
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


import pandas as pd


def build_vocab_from_csv(csv_path, special_tokens=("<pad>", "<cls>", "<mlm>")):
    df = pd.read_csv(csv_path)

    # Must have feature_name and id columns
    assert {"feature_name", "id"}.issubset(
        df.columns
    ), "CSV must contain feature_name and id columns"
    df["feature_name"] = df["feature_name"].astype(str)
    df["id"] = df["id"].astype(int)

    # Gene name -> original id
    gene2id_raw = dict(zip(df["feature_name"], df["id"]))

    # Reserve special tokens
    vocab2id = {}
    vocab2id[special_tokens[0]] = 0  # <pad> fixed at 0
    start_offset = 1  # All other gene ids +1

    # Add genes
    for g, i in gene2id_raw.items():
        vocab2id[g] = i + start_offset

    # Assign remaining special tokens
    max_id = max(vocab2id.values())
    for j, tok in enumerate(special_tokens[1:], start=1):  # <cls>, <eoc>
        vocab2id[tok] = max_id + j

    # Reverse mapping
    id2vocab = {i: g for g, i in vocab2id.items()}
    return vocab2id, id2vocab, special_tokens[0], vocab2id[special_tokens[0]]


def build_gene_ids_for_dataset(genes, vocab2id, pad_token="<pad>"):
    pad_id = vocab2id[pad_token]
    gene_ids = torch.tensor([vocab2id.get(g, pad_id) for g in genes], dtype=int)
    n_hit = int((gene_ids != pad_id).sum())
    print(
        f"[Mapping] Matched {n_hit}/{len(genes)} genes, unmapped genes use <pad>={pad_id}"
    )
    return gene_ids


# code from scGPT/scgpt/utils/util.py
def compute_perturbation_metrics(
    results: Dict,
    ctrl_adata: AnnData,
    non_zero_genes: bool = False,
    return_raw: bool = False,
) -> Dict:
    """
    Given results from a model run and the ground truth, compute metrics

    Args:
        results (:obj:`Dict`): The results from a model run
        ctrl_adata (:obj:`AnnData`): The adata of the control condtion
        non_zero_genes (:obj:`bool`, optional): Whether to only consider non-zero
            genes in the ground truth when computing metrics
        return_raw (:obj:`bool`, optional): Whether to return the raw metrics or
            the mean of the metrics. Default is False.

    Returns:
        :obj:`Dict`: The metrics computed
    """
    from scipy.stats import pearsonr

    # metrics:
    #   Pearson correlation of expression on all genes, on DE genes,
    #   Pearson correlation of expression change on all genes, on DE genes,

    metrics_across_genes = {
        "pearson": [],
        "pearson_de": [],
        "pearson_delta": [],
        "pearson_de_delta": [],
    }

    metrics_across_conditions = {
        "pearson": [],
        "pearson_delta": [],
    }

    conditions = np.unique(results["pert_cat"])
    # assert not "ctrl" in conditions, "ctrl should not be in test conditions"
    condition2idx = {c: np.where(results["pert_cat"] == c)[0] for c in conditions}

    mean_ctrl = np.array(ctrl_adata.X.mean(0)).flatten()  # (n_genes,)
    assert ctrl_adata.X.max() <= 1000, "gene expression should be log transformed"

    true_perturbed = results["truth"]  # (n_cells, n_genes)
    assert true_perturbed.max() <= 1000, "gene expression should be log transformed"
    true_mean_perturbed_by_condition = np.array(
        [true_perturbed[condition2idx[c]].mean(0) for c in conditions]
    )  # (n_conditions, n_genes)
    print(
        f"true_mean_perturbed_by_condition shape: {true_mean_perturbed_by_condition.shape}"
    )
    print(f"mean_ctrl shape: {mean_ctrl.shape}")
    true_mean_delta_by_condition = true_mean_perturbed_by_condition - mean_ctrl
    zero_rows = np.where(np.all(true_mean_perturbed_by_condition == 0, axis=1))[
        0
    ].tolist()
    zero_cols = np.where(np.all(true_mean_perturbed_by_condition == 0, axis=0))[
        0
    ].tolist()

    pred_perturbed = results["pred"]  # (n_cells, n_genes)
    pred_mean_perturbed_by_condition = np.array(
        [pred_perturbed[condition2idx[c]].mean(0) for c in conditions]
    )  # (n_conditions, n_genes)
    print(
        f"pred_mean_perturbed_by_condition shape: {pred_mean_perturbed_by_condition.shape}"
    )
    print(f"mean_ctrl shape: {mean_ctrl.shape}")
    pred_mean_delta_by_condition = pred_mean_perturbed_by_condition - mean_ctrl

    def corr_over_genes(x, y, conditions, res_list, skip_rows=[], non_zero_mask=None):
        """compute pearson correlation over genes for each condition"""
        for i, c in enumerate(conditions):
            if i in skip_rows:
                continue
            x_, y_ = x[i], y[i]
            if non_zero_mask is not None:
                x_ = x_[non_zero_mask[i]]
                y_ = y_[non_zero_mask[i]]
            res_list.append(pearsonr(x_, y_)[0])

    corr_over_genes(
        true_mean_perturbed_by_condition,
        pred_mean_perturbed_by_condition,
        conditions,
        metrics_across_genes["pearson"],
        zero_rows,
        non_zero_mask=true_mean_perturbed_by_condition != 0 if non_zero_genes else None,
    )
    corr_over_genes(
        true_mean_delta_by_condition,
        pred_mean_delta_by_condition,
        conditions,
        metrics_across_genes["pearson_delta"],
        zero_rows,
        non_zero_mask=true_mean_perturbed_by_condition != 0 if non_zero_genes else None,
    )

    def find_DE_genes(adata, condition, geneid2idx, non_zero_genes=False, top_n=20):
        """
        Find the DE genes for a condition
        """
        key_components = next(
            iter(adata.uns["rank_genes_groups_cov_all"].keys())
        ).split("_")
        assert len(key_components) == 3, "rank_genes_groups_cov_all key is not valid"

        condition_key = "_".join([key_components[0], condition, key_components[2]])

        de_genes = adata.uns["rank_genes_groups_cov_all"][condition_key]
        if non_zero_genes:
            de_genes = adata.uns["top_non_dropout_de_20"][condition_key]
            # de_genes = adata.uns["rank_genes_groups_cov_all"][condition_key]
            # de_genes = de_genes[adata.uns["non_zeros_gene_idx"][condition_key]]
            # assert len(de_genes) > top_n

        de_genes = de_genes[:top_n]

        de_idx = [geneid2idx[i] for i in de_genes]

        return de_idx, de_genes

    geneid2idx = dict(zip(ctrl_adata.var.index.values, range(len(ctrl_adata.var))))
    de_idx = {
        c: find_DE_genes(ctrl_adata, c, geneid2idx, non_zero_genes)[0]
        for c in conditions
    }
    mean_ctrl_de = np.array(
        [mean_ctrl[de_idx[c]] for c in conditions]
    )  # (n_conditions, n_diff_genes)

    true_mean_perturbed_by_condition_de = np.array(
        [
            true_mean_perturbed_by_condition[i, de_idx[c]]
            for i, c in enumerate(conditions)
        ]
    )  # (n_conditions, n_diff_genes)
    zero_rows_de = np.where(np.all(true_mean_perturbed_by_condition_de == 0, axis=1))[
        0
    ].tolist()
    true_mean_delta_by_condition_de = true_mean_perturbed_by_condition_de - mean_ctrl_de

    pred_mean_perturbed_by_condition_de = np.array(
        [
            pred_mean_perturbed_by_condition[i, de_idx[c]]
            for i, c in enumerate(conditions)
        ]
    )  # (n_conditions, n_diff_genes)
    pred_mean_delta_by_condition_de = pred_mean_perturbed_by_condition_de - mean_ctrl_de

    corr_over_genes(
        true_mean_perturbed_by_condition_de,
        pred_mean_perturbed_by_condition_de,
        conditions,
        metrics_across_genes["pearson_de"],
        zero_rows_de,
    )
    corr_over_genes(
        true_mean_delta_by_condition_de,
        pred_mean_delta_by_condition_de,
        conditions,
        metrics_across_genes["pearson_de_delta"],
        zero_rows_de,
    )

    if not return_raw:
        for k, v in metrics_across_genes.items():
            metrics_across_genes[k] = np.mean(v)
        for k, v in metrics_across_conditions.items():
            metrics_across_conditions[k] = np.mean(v)
    metrics = metrics_across_genes

    return metrics
