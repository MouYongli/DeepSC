"""
DeepSC Model Pre-training Script.

This module provides functionality for pre-training DeepSC (Deep Learning-based
Semantic Communication) models using PyTorch Lightning Fabric for distributed training.

The script supports multi-GPU and multi-node training with automatic mixed precision
and distributed data parallel strategy.

Author: DeepSC Team
"""

import hydra
import torch.nn as nn
from lightning.fabric import Fabric
from lightning.fabric.strategies import DDPStrategy
from omegaconf import DictConfig

from deepsc.train.trainer import Trainer
from deepsc.utils.utils import setup_logging


@hydra.main(
    version_base=None, config_path="../../../configs/pretrain", config_name="pretrain"
)
def pretrain(cfg: DictConfig) -> None:
    """
    Main pre-training function for DeepSC models.

    This function initializes the distributed training environment, sets up logging,
    instantiates the model from configuration, and starts the training process.

    Args:
        cfg (DictConfig): Hydra configuration object containing all training parameters
                         including model configuration, device settings, and training
                         hyperparameters.

    Returns:
        None

    Note:
        The function uses PyTorch Lightning Fabric for distributed training setup
        and supports CUDA acceleration with mixed precision (bf16) training.
    """
    # Initialize PyTorch Lightning Fabric for distributed training
    fabric = Fabric(
        accelerator="cuda",  # Use CUDA for GPU acceleration
        devices=cfg.num_device,  # Number of devices per node
        num_nodes=cfg.num_nodes,  # Number of nodes for distributed training
        strategy=DDPStrategy(find_unused_parameters=False),  # Distributed data parallel
        precision="bf16-mixed",  # Mixed precision training for efficiency
    )
    print(f"cfg: {cfg.model}")
    fabric.launch()

    # Initialize logging system with fabric's global rank
    setup_logging(rank=fabric.global_rank, log_path="./logs")

    # Note: wandb initialization will be handled in trainer after checkpoint check
    # This approach prevents creating empty wandb runs if we can resume from checkpoint

    # Instantiate the model from Hydra configuration
    model: nn.Module = hydra.utils.instantiate(cfg.model)
    # Ensure model parameters are in float32 for stable training
    model = model.float()

    # Create trainer instance and start training
    trainer = Trainer(cfg, fabric=fabric, model=model)
    trainer.train()


if __name__ == "__main__":
    """Entry point for the pre-training script."""
    pretrain()
