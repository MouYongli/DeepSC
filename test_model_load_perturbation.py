import os
import random

import hydra
import torch
import torch.nn as nn
from lightning.fabric import Fabric
from lightning.fabric.strategies import DDPStrategy
from omegaconf import DictConfig

from deepsc.utils.utils import setup_logging
from src.deepsc.utils import count_unique_cell_types


def _extract_state_dict(maybe_state):
    """
    Compatible with multiple save formats:
    - {"model": state_dict, ...}  <- Your current save format (Fabric)
    - {"state_dict": state_dict, ...}  <- Common in Lightning
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
            # Possibly the state_dict itself
            sd = maybe_state
    else:
        raise ValueError("Checkpoint content is not a dict, cannot parse state_dict")

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


def _report_loading_result(load_info):
    missing = list(load_info.missing_keys)
    unexpected = list(load_info.unexpected_keys)
    print(f"[LOAD] missing_keys: {len(missing)} | unexpected_keys: {len(unexpected)}")
    if missing:
        print("  - (first 10) missing:", missing[:10])
    if unexpected:
        print("  - (first 10) unexpected:", unexpected[:10])


def _sample_weight_norms(model, sd, k=5):
    """
    Randomly sample k parameters that exist in both sides, print norm changes before/after loading.
    Norm changes => basically confirms successful write.
    """
    with torch.no_grad():
        common_keys = [name for name, _ in model.named_parameters() if name in sd]
        if not common_keys:
            print(
                "[LOAD] No common parameter names found between checkpoint and model, cannot compare norms."
            )
            return
        sample = random.sample(common_keys, min(k, len(common_keys)))
        print("[LOAD] Sampled parameter norm comparison (before -> after loading):")
        for name in sample:
            p = dict(model.named_parameters())[name]
            before = p.detach().float().norm().item()
            # Store current weights
            old = p.detach().cpu().clone()
            # Override with ckpt once
            p.copy_(sd[name].to(p.device).to(p.dtype))
            after = p.detach().float().norm().item()
            print(f"  - {name}: {before:.6f} -> {after:.6f}")
            # Restore (only for comparison; actual loading will be done again in load_state_dict)
            p.copy_(old.to(p.device).to(p.dtype))


@hydra.main(version_base=None, config_path="./configs/finetune", config_name="finetune")
def finetune(cfg: DictConfig):
    cfg.cell_type_count = count_unique_cell_types(cfg.data_path, cfg.obs_celltype_col)
    fabric = Fabric(
        accelerator="cuda",
        devices=cfg.num_device,
        num_nodes=cfg.num_nodes,
        strategy=DDPStrategy(
            find_unused_parameters=True, gradient_as_bucket_view=False
        ),
        precision="bf16-mixed",
    )
    fabric.launch()
    setup_logging(rank=fabric.global_rank, log_path="./logs")

    # === Instantiate model ===
    model: nn.Module = hydra.utils.instantiate(cfg.model)
    # (Optional) Don't convert to float first, Fabric/AMP will handle it; if you insist, do it after load
    # model = model.float()

    # === BEGIN: Test whether loading is successful ===
    # 1) Path
    ckpt_path = "/home/angli/baseline/DeepSC/results/latest_checkpoint.ckpt"
    assert os.path.exists(ckpt_path), f"Cannot find ckpt: {ckpt_path}"

    # 2) Only read on rank0 to CPU to reduce pressure; then broadcast (optional)
    if fabric.global_rank == 0:
        print(f"[LOAD] Reading checkpoint: {ckpt_path}")
        raw = torch.load(ckpt_path, map_location="cpu")
        state_dict = _extract_state_dict(raw)
    else:
        raw = None
        state_dict = None

    # 3) Broadcast to all processes
    state_dict = fabric.broadcast(state_dict, src=0)

    # 4) Print sampled comparison (optional, but intuitive)
    _sample_weight_norms(model, state_dict, k=5)

    # 5) Actually load (strict=False: allows newly added embedding layers to remain empty)
    load_info = model.load_state_dict(state_dict, strict=False)
    _report_loading_result(load_info)

    # 6) Simple shape verification: match keys that exist in both, pick one to compare shape
    common = [
        (k, v.shape, dict(model.named_parameters())[k].shape)
        for k, v in state_dict.items()
        if k in dict(model.named_parameters())
    ]
    if common:
        k0, s_ckpt, s_model = common[0]
        print(
            f"[LOAD] Shape example: {k0}: ckpt {tuple(s_ckpt)} vs model {tuple(s_model)}"
        )

    # If needed, convert dtype now
    # model = model.float()

    # 7) At this point, we can basically confirm whether loading was successful; if only verifying, just return:
    if cfg.get("only_check_loading", False):
        print("[LOAD] Only checking loading, program ends.")
        return
    # === END: Test whether loading is successful ===

    # === Below is the regular training process ===
    # Better to build optimizer after setup, to avoid param mismatch after DDP/Fabric wraps parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    model, optimizer = fabric.setup(model, optimizer)

    # Actual business Trainer (if needed)
    # Note: You commented out PerturbationPrediction; uncomment if you need it
    # trainer = PerturbationPrediction(cfg, fabric=fabric, model=model)
    # trainer.train()


if __name__ == "__main__":
    finetune()
