import hydra
import torch.nn as nn
from lightning.fabric import Fabric
from lightning.fabric.strategies import DDPStrategy
from omegaconf import DictConfig

from deepsc.train.trainer import Trainer
from deepsc.utils.utils import setup_logging


@hydra.main(
    version_base=None, config_path="../../../configs/pretrain", config_name="pretrain"
)
def pretrain(cfg: DictConfig):
    # initialize fabric
    fabric = Fabric(
        accelerator="cuda",
        devices=cfg.num_device,
        num_nodes=cfg.num_nodes,
        strategy=DDPStrategy(find_unused_parameters=True),
        precision="bf16-mixed",
    )
    fabric.launch()
    # initialize log
    setup_logging(rank=fabric.global_rank, log_path="./logs")

    # wandb initialization will be handled in trainer after checkpoint check
    # This way we don't create empty runs if we can resume

    # model = select_model(cfg)
    # instantiate model
    model: nn.Module = hydra.utils.instantiate(cfg.model)
    model = model.float()
    trainer = Trainer(cfg, fabric=fabric, model=model)
    trainer.train()


if __name__ == "__main__":
    pretrain()
