#!/usr/bin/env python3
"""
Script to inspect checkpoint file structure and layer information
"""

import os
from collections import OrderedDict

import torch


def inspect_checkpoint(checkpoint_path):
    """Inspect checkpoint file structure"""
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file {checkpoint_path} does not exist!")
        return

    print(f"Loading checkpoint from: {checkpoint_path}")
    print("=" * 80)

    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Display checkpoint top-level keys
        print("\n📁 Checkpoint top-level keys:")
        print("-" * 40)
        for key in checkpoint.keys():
            if isinstance(checkpoint[key], dict):
                print(f"  {key}: dict with {len(checkpoint[key])} items")
            elif isinstance(checkpoint[key], torch.Tensor):
                print(f"  {key}: tensor {checkpoint[key].shape}")
            else:
                print(f"  {key}: {type(checkpoint[key])}")

        # Check model parameter location
        state_dict = None
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            print(f"\n🔍 Found 'state_dict' with {len(state_dict)} parameters")
        elif "model" in checkpoint:
            model_data = checkpoint["model"]
            print("\n🔍 Found 'model' key, checking its structure...")
            print(f"   Type: {type(model_data)}")

            # If model is a dict, it might be state_dict
            if isinstance(model_data, dict):
                # Check if it contains tensor parameters
                tensor_keys = [
                    k for k, v in model_data.items() if isinstance(v, torch.Tensor)
                ]
                if tensor_keys:
                    state_dict = model_data
                    print(
                        f"   Found {len(tensor_keys)} tensor parameters in 'model' key"
                    )
                else:
                    print(
                        f"   'model' key contains {len(model_data)} items but no tensors"
                    )
                    # Try to view internal model structure
                    for k, v in list(model_data.items())[:5]:
                        print(f"     {k}: {type(v)}")
            else:
                print(f"   'model' key is not a dict, type: {type(model_data)}")

        if state_dict is None:
            print(
                "\n🔍 No usable state_dict found, treating entire checkpoint as state_dict"
            )
            state_dict = checkpoint

        print("\n🧠 Model layers and parameters:")
        print("-" * 80)

        # Only display tensor parameters
        total_params = 0
        for name, param in state_dict.items():
            if isinstance(param, torch.Tensor):
                param_count = param.numel()
                total_params += param_count
                print(
                    f"  {name:60} | {str(param.shape):25} | {param_count:>12,} params"
                )

        print(f"\n🎯 Total parameters: {total_params:,}")

        # Check if there's a 'model.' prefix
        tensor_keys = [k for k, v in state_dict.items() if isinstance(v, torch.Tensor)]
        if tensor_keys:
            has_model_prefix = any(k.startswith("model.") for k in tensor_keys)
            print(f"🏷️  Has 'model.' prefix: {has_model_prefix}")

        # Display layers grouped by module
        print("\n📊 Layers grouped by module:")
        print("-" * 80)
        layer_groups = OrderedDict()
        for name, param in state_dict.items():
            if isinstance(param, torch.Tensor):
                module_name = name.split(".")[0] if "." in name else name
                if module_name not in layer_groups:
                    layer_groups[module_name] = {"count": 0, "params": 0}

                layer_groups[module_name]["count"] += 1
                layer_groups[module_name]["params"] += param.numel()

        for module, info in layer_groups.items():
            print(
                f"  {module:40} | {info['count']:3} layers | {info['params']:>12,} params"
            )

        return checkpoint, state_dict

    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return None, None


if __name__ == "__main__":
    checkpoint_path = "/home/angli/baseline/DeepSC/results/latest_checkpoint.ckpt"
    checkpoint, state_dict = inspect_checkpoint(checkpoint_path)
