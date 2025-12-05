import hydra
import torch.nn as nn
from lightning.fabric import Fabric
from lightning.fabric.strategies import DDPStrategy
from omegaconf import DictConfig, OmegaConf

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
    # Merge task-specific configs to top level for backward compatibility
    # This allows cell_type_annotation.py to access configs via cfg.data_path
    # instead of cfg.tasks.data_path
    if "tasks" in cfg:
        # Disable struct mode to allow merging new keys
        OmegaConf.set_struct(cfg, False)
        task_cfg = OmegaConf.to_container(cfg.tasks, resolve=True)
        cfg = OmegaConf.merge(cfg, task_cfg)
        # Remove the tasks key to avoid confusion
        if "tasks" in cfg:
            del cfg["tasks"]
        # Re-enable struct mode to prevent accidental key additions
        OmegaConf.set_struct(cfg, True)

    # initialize fabric
    fabric = Fabric(
        accelerator="cuda",
        devices=cfg.num_device,
        num_nodes=cfg.num_nodes,
        strategy=DDPStrategy(
            find_unused_parameters=True,
            gradient_as_bucket_view=False,  # 更稳一些，避免 bucket 视图带来的边缘问题
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

    # Get task type from config
    task_type = getattr(cfg, "task_type", "cell_type_annotation")
    print(f"Running task: {task_type}")

    if task_type == "cell_type_annotation":
        # Cell type annotation task
        # 根据数据集使用方式获取细胞类型数量
        use_separated_datasets = getattr(cfg, "seperated_train_eval_dataset", True)

        if use_separated_datasets:
            # 使用两个分开数据集的交集
            print("Getting common cell type count from separated datasets...")
            actual_cell_type_count, cell_type_names = (
                count_common_cell_types_from_multiple_files(
                    cfg.data_path,
                    cfg.data_path_eval,
                    cell_type_col=cfg.obs_celltype_col,
                )
            )
        else:
            # 使用单个数据集的全部细胞类型
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
            n_cls=actual_cell_type_count,  # 使用实际的celltype数量
            num_layers_cls=3,
            cell_emb_style="avg-pool",
        )
        model = model.float()
        trainer = CellTypeAnnotation(cfg, fabric=fabric, model=model)
        trainer.train()
        # test_ckpt = TestCkpt(cfg, fabric=fabric, model=model)
    else:
        raise ValueError(
            f"Unknown task type: {task_type}. Supported types: ['cell_type_annotation']"
        )


if __name__ == "__main__":
    finetune()
