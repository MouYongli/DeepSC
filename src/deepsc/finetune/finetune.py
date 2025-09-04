import hydra
import torch.nn as nn
from lightning.fabric import Fabric
from lightning.fabric.strategies import DDPStrategy
from omegaconf import DictConfig

from deepsc.finetune.cell_type_annotation import CellTypeAnnotation
from deepsc.models.deepsc_new.model import DeepSCClassifier
from deepsc.utils.utils import setup_logging
from src.deepsc.utils import count_unique_cell_types_from_multiple_files


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
            gradient_as_bucket_view=False,  # 更稳一些，避免 bucket 视图带来的边缘问题
        ),
        precision="bf16-mixed",
    )
    fabric.launch()
    # initialize log
    setup_logging(rank=fabric.global_rank, log_path="./logs")

    # wandb initialization will be handled in trainer after checkpoint check
    # This way we don't create empty runs if we can resume

    # 提前获取实际的celltype数量
    print("Getting actual cell type count from data files...")
    actual_cell_type_count, cell_type_names = (
        count_unique_cell_types_from_multiple_files(
            cfg.data_path, cfg.data_path_eval, cell_type_col=cfg.obs_celltype_col
        )
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


if __name__ == "__main__":
    finetune()
