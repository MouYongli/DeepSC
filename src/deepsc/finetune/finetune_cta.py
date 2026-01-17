import os
import shutil

import hydra
import torch.nn as nn
from lightning.fabric import Fabric
from lightning.fabric.strategies import DDPStrategy
from omegaconf import DictConfig

from deepsc.finetune.cell_type_annotation import CellTypeAnnotation
from deepsc.models.deepsc.model import DeepSCClassifier
from deepsc.utils.utils import setup_logging
from src.deepsc.utils import (
    count_common_cell_types_from_multiple_files,
    count_unique_cell_types,
)


@hydra.main(
    version_base=None, config_path="../../../configs/finetune", config_name="finetune"
)
def finetune(cfg: DictConfig):
    # initialize fabric
    fabric = Fabric(
        accelerator="cuda",
        devices=cfg.num_device,
        num_nodes=cfg.num_nodes,
        strategy=DDPStrategy(
            find_unused_parameters=True,
            gradient_as_bucket_view=False,  # More stable, avoids edge cases from bucket view
        ),
        precision="bf16-mixed",
    )
    fabric.launch()
    # initialize log (Hydra auto-manages working directory)
    # use_hydra=True to only redirect stdout/stderr, not reconfigure logging
    setup_logging(
        rank=fabric.global_rank, log_path=".", log_name="finetune", use_hydra=True
    )

    # wandb initialization will be handled in trainer after checkpoint check
    # This way we don't create empty runs if we can resume

    # Get cell type count based on dataset usage
    use_separated_datasets = getattr(cfg, "seperated_train_eval_dataset", True)

    if use_separated_datasets:
        # Use intersection of two separate datasets
        print("Getting common cell type count from separated datasets...")
        actual_cell_type_count, cell_type_names = (
            count_common_cell_types_from_multiple_files(
                cfg.data_path, cfg.data_path_eval, cell_type_col=cfg.obs_celltype_col
            )
        )
    else:
        # Use all cell types from single dataset
        print("Getting cell type count from single dataset...")
        actual_cell_type_count, cell_type_names = count_unique_cell_types(
            cfg.data_path, cell_type_col=cfg.obs_celltype_col
        )

    # model = select_model(cfg)
    # instantiate model
    model: nn.Module = hydra.utils.instantiate(cfg.model)
    encoder = model.float()
    model = DeepSCClassifier(
        deepsc_encoder=encoder,
        n_cls=actual_cell_type_count,  # Use actual celltype count
        num_layers_cls=3,
        cell_emb_style="avg-pool",
    )
    model = model.float()
    trainer = CellTypeAnnotation(cfg, fabric=fabric, model=model)
    trainer.train()

    # After training, copy hydra logs to training output directory
    if fabric.global_rank == 0 and trainer.output_dir:
        try:
            # Get current hydra output directory
            hydra_output_dir = os.getcwd()

            # Find finetune_0.log file
            log_file = os.path.join(hydra_output_dir, "finetune_0.log")
            if os.path.exists(log_file):
                dest_log = os.path.join(trainer.log_dir, "finetune.log")
                shutil.copy2(log_file, dest_log)
                print(f"Copied log file to: {dest_log}")

            # Copy hydra configuration
            hydra_config_dir = os.path.join(hydra_output_dir, ".hydra")
            if os.path.exists(hydra_config_dir):
                dest_config_dir = os.path.join(trainer.output_dir, "config")
                shutil.copytree(hydra_config_dir, dest_config_dir, dirs_exist_ok=True)
                print(f"Copied config to: {dest_config_dir}")
        except Exception as e:
            print(f"Warning: Failed to copy logs/config: {e}")


if __name__ == "__main__":
    finetune()
