"""
Fine-tuning utility functions for model training.

This module provides utilities for configuring fine-tuning modes and
managing trainable parameters during the fine-tuning process.
"""


def setup_finetune_mode(model, finetune_mode="full", is_master=False):
    """
    Configure the fine-tuning mode based on the configuration.

    Supports two modes:
    - full: Full parameter fine-tuning (all model parameters)
    - head_only: Only fine-tune the classification head

    Args:
        model: Model with 'encoder' and 'cls_decoder' attributes
        finetune_mode: Fine-tuning mode, "full" or "head_only"
        is_master: Whether this is the master process (for logging)

    Raises:
        ValueError: If finetune_mode is not recognized
    """
    if finetune_mode == "full":
        # Full fine-tuning: train all parameters
        if is_master:
            print("=" * 80)
            print("Fine-tuning Mode: FULL PARAMETER FINE-TUNING")
            print("All model parameters will be trained.")
            print("=" * 80)
        # All parameters are trainable by default, no need to change

    elif finetune_mode == "head_only":
        # Head-only fine-tuning: freeze encoder, only train classification head
        if is_master:
            print("=" * 80)
            print("Fine-tuning Mode: CLASSIFICATION HEAD ONLY")
            print("Only the classification head will be trained.")
            print("=" * 80)

        # Freeze all encoder parameters
        for param in model.encoder.parameters():
            param.requires_grad = False

        # Ensure cls_decoder parameters are trainable
        for param in model.cls_decoder.parameters():
            param.requires_grad = True
    else:
        raise ValueError(
            f"Unknown finetune_mode: {finetune_mode}. "
            f"Must be one of: 'full', 'head_only'"
        )


def get_trainable_parameters(model, finetune_mode="full", is_master=False):
    """
    Get the list of trainable parameters based on the fine-tuning mode.

    Args:
        model: Model with parameters
        finetune_mode: Fine-tuning mode, "full" or "head_only"
        is_master: Whether this is the master process (for logging)

    Returns:
        iterator: Iterator of trainable parameters
    """
    if finetune_mode == "head_only":
        # Only return cls_decoder parameters
        trainable_params = model.cls_decoder.parameters()
    else:
        trainable_params = model.parameters()

    # Count and print trainable parameters
    if is_master:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params_list = list(trainable_params)
        trainable_count = sum(p.numel() for p in trainable_params_list)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_count:,}")
        print(f"Trainable ratio: {trainable_count / total_params * 100:.2f}%")
        print("=" * 80)
        return iter(trainable_params_list)

    return trainable_params
