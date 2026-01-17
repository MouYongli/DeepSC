#!/usr/bin/env python
"""
Simple runner for perturbation prediction - Pure PyTorch, no Fabric
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate


@hydra.main(version_base=None, config_path="configs/pp", config_name="pp")
def main(cfg: DictConfig):
    """Main function"""

    print("="*80)
    print("DeepSC Perturbation Prediction - Pure PyTorch Version")
    print("="*80)
    print(f"\nDataset: {cfg.data_name}")
    print(f"Epochs: {cfg.epoch}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Learning rate: {cfg.learning_rate}")
    print("="*80)

    # Import here to avoid Fabric initialization issues
    from deepsc.finetune.perturbation_pytorch import PerturbationPredictorPyTorch

    # Instantiate model
    print("\nInstantiating model...")
    model = instantiate(cfg.model)
    print(f"✓ Model created: {type(model).__name__}")

    # Create predictor (Pure PyTorch, no Fabric!)
    print("\nInitializing predictor...")
    predictor = PerturbationPredictorPyTorch(
        args=cfg,
        model=model,
        device='cuda'
    )
    print("✓ Predictor initialized")

    # Train
    predictor.train()

    print("\n" + "="*80)
    print("DONE!")
    print("="*80)


if __name__ == "__main__":
    main()
