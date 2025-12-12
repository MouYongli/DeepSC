"""
Perturbation Prediction Fine-tuning with Hydra Configuration
Uses existing pp.yaml configuration from /home/angli/DeepSC/configs/pp/
"""

import os
import sys
from pathlib import Path

import hydra
from hydra.utils import instantiate
from lightning.fabric import Fabric
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from deepsc.finetune.perturbation_finetune import PerturbationPredictor


@hydra.main(version_base=None, config_path="../../../configs/pp", config_name="pp")
def main(cfg: DictConfig):
    """
    Main training function using Hydra configuration

    Args:
        cfg: Hydra configuration loaded from configs/pp/pp.yaml
    """

    # Print configuration
    print("=" * 80)
    print("DeepSC Perturbation Prediction Fine-tuning")
    print("=" * 80)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # Setup Fabric for distributed training
    # Determine precision based on config
    if hasattr(cfg, 'amp') and cfg.amp:
        precision = "16-mixed"
    else:
        precision = "32-true"

    fabric = Fabric(
        accelerator="auto",
        devices=getattr(cfg, 'num_device', 1),
        precision=precision,
    )

    # Launch distributed setup
    fabric.launch()

    if fabric.global_rank == 0:
        print(f"\nDataset: {cfg.data_name}")
        print(f"Pretrained model: {cfg.pretrained_model}")
        if cfg.pretrained_model:
            print(f"Pretrained path: {cfg.pretrained_model_path}")
        print(f"Batch size: {cfg.batch_size}")
        print(f"Learning rate: {cfg.learning_rate}")
        print(f"Epochs: {cfg.epoch}")
        print(f"Devices: {cfg.num_device}")
        print(f"Seed: {cfg.seed}")
        print(f"Include zero gene: {cfg.include_zero_gene}")
        print(f"Data length: {cfg.data_length}")
        print("=" * 80)

    # Instantiate model using Hydra
    # The model config is in configs/pp/model/deepsc.yaml
    if fabric.global_rank == 0:
        print("\nInstantiating model...")

    model = instantiate(cfg.model)

    if fabric.global_rank == 0:
        print(f"Model instantiated: {type(model).__name__}")
        print(f"  - Embedding dim: {cfg.model.embedding_dim}")
        print(f"  - Num layers: {cfg.model.num_layers}")
        print(f"  - Num heads: {cfg.model.num_heads}")
        print(f"  - Use MoE regressor: {cfg.model.use_moe_regressor}")

    # Initialize predictor
    predictor = PerturbationPredictor(
        args=cfg,
        fabric=fabric,
        model=model
    )

    # Train the model
    if fabric.global_rank == 0:
        print("\nStarting training...")

    predictor.train()

    # Generate visualization plots
    if fabric.global_rank == 0:
        print("\nGenerating visualization plots...")
        predictor.plot_predictions()

        print("\nTraining completed!")
        print(f"Results saved to: {predictor.output_dir}")


if __name__ == "__main__":
    main()
