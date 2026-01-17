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
            gradient_as_bucket_view=False,  # More stable, avoids edge issues from bucket view
        ),
        precision="bf16-mixed",
    )
    fabric.launch()

    # Get cell type count based on dataset usage method
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
        n_cls=actual_cell_type_count,  # Use the actual cell type count
        num_layers_cls=3,
        cell_emb_style="avg-pool",
    )
    model = model.float()
    trainer = CellTypeAnnotation(cfg, fabric=fabric, model=model)

    # initialize log after trainer is created (so we can use trainer.log_dir)
    # use_hydra=True to only redirect stdout/stderr, not reconfigure logging
    if fabric.global_rank == 0:
        setup_logging(
            rank=fabric.global_rank,
            log_path=trainer.log_dir,
            log_name="finetune",
            use_hydra=True,
        )

    trainer.train()

    # All files are already in the correct directory, no need to copy
    # Hydra config is in .hydra/ directory
    # Training checkpoints are in checkpoints/ directory
    # Logs are in logs/ directory
    # Visualizations are in visualizations/ directory


if __name__ == "__main__":
    finetune()
