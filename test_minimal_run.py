#!/usr/bin/env python
"""Minimal test to find where the code is stuck"""

import sys
sys.path.insert(0, 'src')

print("Step 1: Importing modules...")
from deepsc.finetune.perturbation_finetune import PerturbationPredictor
from lightning.fabric import Fabric
print("✓ Import successful")

print("\nStep 2: Loading config...")
from omegaconf import OmegaConf
cfg = OmegaConf.load('configs/pp/pp.yaml')
cfg.epoch = 1
cfg.batch_size = 4
cfg.grad_acc = 1
print(f"✓ Config loaded: {cfg.data_name}")

print("\nStep 3: Setting up Fabric...")
fabric = Fabric(accelerator="auto", devices=1, precision="32-true")
fabric.launch()
print("✓ Fabric initialized")

print("\nStep 4: Instantiating model...")
from hydra.utils import instantiate
model = instantiate(cfg.model)
print(f"✓ Model created: {type(model).__name__}")

print("\nStep 5: Creating PerturbationPredictor...")
print("  (This might take a while - loading data and checkpoint)")
predictor = PerturbationPredictor(
    args=cfg,
    fabric=fabric,
    model=model
)
print("✓ Predictor created successfully!")

print("\n" + "="*60)
print("SUCCESS! All initialization complete.")
print("="*60)
