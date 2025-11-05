"""
Learning rate scheduler utilities for training.

This module provides functions to create various learning rate schedulers
with warmup and decay strategies.
"""

import math

from torch.optim.lr_scheduler import LinearLR, SequentialLR

from .utils import CosineAnnealingWarmRestartsWithDecayAndLinearWarmup


def create_warmup_cosine_scheduler(
    optimizer,
    num_epochs,
    steps_per_epoch,
    batch_size,
    world_size,
    grad_accumulation_steps,
    warmup_ratio=0.03,
    T_0_multiplier=3,
    decay=0.9,
):
    """
    Create a learning rate scheduler with linear warmup and cosine annealing.

    The scheduler consists of two phases:
    1. Linear warmup: Learning rate increases linearly from 1% to 100%
    2. Cosine annealing with restarts: Learning rate follows a cosine curve with periodic restarts

    Args:
        optimizer: PyTorch optimizer
        num_epochs: Total number of training epochs
        steps_per_epoch: Number of optimization steps per epoch
            (usually: dataset_size / (batch_size * world_size * grad_accumulation_steps))
        batch_size: Batch size per GPU
        world_size: Number of GPUs
        grad_accumulation_steps: Number of gradient accumulation steps
        warmup_ratio: Proportion of total steps for warmup phase (default: 0.03)
        T_0_multiplier: Multiplier for the initial period of cosine annealing (default: 3)
        decay: Decay factor for cosine annealing restarts (default: 0.9)

    Returns:
        torch.optim.lr_scheduler.SequentialLR: Combined warmup + cosine annealing scheduler
    """
    # Calculate total training steps
    total_steps = num_epochs * steps_per_epoch

    # Calculate warmup steps
    warmup_steps = math.ceil(total_steps * warmup_ratio)

    # Phase 1: Linear warmup
    linear_warmup = LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
    )

    # Phase 2: Cosine annealing with restarts and decay
    cosine_anneal = CosineAnnealingWarmRestartsWithDecayAndLinearWarmup(
        optimizer,
        T_0=warmup_steps * T_0_multiplier,
        T_mult=1,
        warmup_steps=0,  # No additional warmup after the initial phase
        decay=decay,
    )

    # Combine warmup and cosine annealing
    scheduler = SequentialLR(
        optimizer,
        schedulers=[linear_warmup, cosine_anneal],
        milestones=[warmup_steps],
    )

    return scheduler


def create_scheduler_from_args(optimizer, args, world_size):
    """
    Create a scheduler from configuration arguments.

    This is a convenience wrapper around create_warmup_cosine_scheduler
    that extracts parameters from a config object.

    Args:
        optimizer: PyTorch optimizer
        args: Configuration object with the following attributes:
            - epoch: Number of epochs
            - batch_size: Batch size per GPU
            - grad_acc: Gradient accumulation steps
            - warmup_ratio: Warmup ratio (optional, default: 0.03)
        world_size: Number of GPUs/processes

    Returns:
        torch.optim.lr_scheduler: Learning rate scheduler
    """
    # Calculate steps per epoch (hardcoded 100000 dataset size for now)
    # TODO: Make this configurable
    steps_per_epoch = math.ceil(100000 / (args.batch_size * world_size * args.grad_acc))

    warmup_ratio = getattr(args, "warmup_ratio", 0.03)

    return create_warmup_cosine_scheduler(
        optimizer=optimizer,
        num_epochs=args.epoch,
        steps_per_epoch=steps_per_epoch,
        batch_size=args.batch_size,
        world_size=world_size,
        grad_accumulation_steps=args.grad_acc,
        warmup_ratio=warmup_ratio,
    )
