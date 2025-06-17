import hydra
import torch.nn as nn
from lightning.fabric import Fabric
from lightning.fabric.strategies import FSDPStrategy
from omegaconf import DictConfig

import wandb
from deepsc.train.trainer import Trainer
from deepsc.utils.utils import setup_logging


@hydra.main(
    version_base=None, config_path="../../../configs/pretrain", config_name="pretrain"
)
def pretrain(cfg: DictConfig):
    # initialize fabric
    strategy = FSDPStrategy(state_dict_type="sharded")
    fabric = Fabric(
        accelerator="cuda",
        devices=cfg.num_device,
        strategy=strategy,
        precision="bf16-mixed",
    )
    fabric.launch()
    # initialize log
    setup_logging(fabric.global_rank, "./logs")

    # wandb only in master
    if fabric.global_rank == 0:
        wandb.init(project=cfg.get("wandb_project", "DeepSC"), config=dict(cfg))

    # model = select_model(cfg)
    # instantiate model
    model: nn.Module = hydra.utils.instantiate(cfg.model)
    model = model.float()
    trainer = Trainer(cfg, fabric=fabric, model=model)
    trainer.train()


if __name__ == "__main__":
    pretrain()
