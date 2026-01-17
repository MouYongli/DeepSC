"""
Example script for running perturbation prediction fine-tuning with DeepSC
Based on scGPT's perturbation prediction workflow
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from lightning.fabric import Fabric

from deepsc.finetune.perturbation_finetune import PerturbationPredictor
from deepsc.models.deepsc.model import DeepSCModel  # Adjust based on your model structure


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Fine-tune DeepSC model for perturbation prediction"
    )

    # Data settings
    parser.add_argument(
        "--data_name",
        type=str,
        default="norman",
        choices=["norman", "adamson", "replogle_rpe1_essential"],
        help="Name of the perturbation dataset"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data",
        help="Path to data directory"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="simulation",
        help="Data split to use"
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to CSV file for vocabulary"
    )

    # Model settings
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=512,
        help="Embedding dimension"
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=12,
        help="Number of transformer layers"
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=8,
        help="Number of attention heads"
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=512,
        help="Hidden dimension in feedforward network"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate"
    )

    # Training settings
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Training batch size"
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=64,
        help="Evaluation batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--grad_acc",
        type=int,
        default=1,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--early_stop",
        type=int,
        default=10,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--schedule_interval",
        type=int,
        default=1,
        help="Learning rate scheduler step interval"
    )
    parser.add_argument(
        "--scheduler_gamma",
        type=float,
        default=0.9,
        help="Learning rate scheduler gamma"
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        default=True,
        help="Use automatic mixed precision"
    )

    # Data processing settings
    parser.add_argument(
        "--include_zero_gene",
        type=str,
        default="all",
        choices=["all", "batch-wise"],
        help="How to handle zero-expression genes"
    )
    parser.add_argument(
        "--data_length",
        type=int,
        default=1536,
        help="Maximum sequence length"
    )

    # Pretrained model
    parser.add_argument(
        "--pretrained_model",
        action="store_true",
        help="Load pretrained model"
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        help="Path to pretrained model checkpoint"
    )

    # Logging and checkpointing
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="Logging interval (in batches)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory (optional, will use time-stamped dir if not specified)"
    )

    # Perturbations to plot
    parser.add_argument(
        "--perts_to_plot",
        type=str,
        nargs="+",
        help="List of perturbations to visualize (e.g., 'SAMD1+ZBTB1')"
    )

    # Distributed training
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of devices to use"
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu", "tpu"],
        help="Accelerator type"
    )

    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()

    # Setup Fabric for distributed training
    fabric = Fabric(
        accelerator=args.accelerator,
        devices=args.devices,
        precision="16-mixed" if args.amp else "32-true",
    )

    # Launch distributed setup
    fabric.launch()

    # Print configuration
    if fabric.global_rank == 0:
        print("=" * 80)
        print("DeepSC Perturbation Prediction Fine-tuning")
        print("=" * 80)
        print(f"Dataset: {args.data_name}")
        print(f"Batch size: {args.batch_size}")
        print(f"Learning rate: {args.learning_rate}")
        print(f"Epochs: {args.epoch}")
        print(f"Devices: {args.devices}")
        print(f"Seed: {args.seed}")
        print("=" * 80)

    # Initialize model
    # NOTE: You need to adjust this based on your actual DeepSC model structure
    # This is a placeholder - replace with your actual model initialization
    model = DeepSCModel(
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        # Add other model-specific parameters
    )

    # Initialize predictor
    predictor = PerturbationPredictor(
        args=args,
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
