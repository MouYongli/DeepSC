import hydra
import torch.nn as nn
from lightning.fabric import Fabric
from lightning.fabric.strategies import DDPStrategy
from omegaconf import DictConfig

from deepsc.utils.utils import setup_logging
from src.deepsc.finetune.pp_new import PPNEW


@hydra.main(version_base=None, config_path="../../../configs/pp", config_name="pp")
def finetune(cfg: DictConfig):
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

    # instantiate model first
    model: nn.Module = hydra.utils.instantiate(cfg.model)
    model = model.float()

    # create trainer (which sets up output directories)
    trainer = PPNEW(cfg, fabric=fabric, model=model)

    # initialize log to trainer's log directory
    # use_hydra=True to only redirect stdout/stderr, not reconfigure logging
    if fabric.global_rank == 0:
        setup_logging(rank=fabric.global_rank, log_path=trainer.log_dir, use_hydra=True)
    else:
        setup_logging(
            rank=fabric.global_rank, log_path="./logs", use_hydra=True
        )  # fallback for non-master

    # wandb initialization will be handled in trainer after checkpoint check
    # This way we don't create empty runs if we can resume

    trainer.train()


if __name__ == "__main__":
    finetune()
