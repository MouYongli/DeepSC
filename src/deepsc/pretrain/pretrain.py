import hydra
import torch.nn as nn
from lightning.fabric import Fabric
from omegaconf import DictConfig

from deepsc.train.trainer import Trainer
from deepsc.utils.utils import setup_logging


@hydra.main(
    version_base=None, config_path="../../../configs/pretrain", config_name="pretrain"
)
def pretrain(cfg: DictConfig):
    # initialize fabric
    fabric = Fabric(accelerator="cuda", devices=cfg.num_device, strategy="ddp")
    fabric.launch()
    # initialize log
    setup_logging(fabric.global_rank, "./logs")

    # model = select_model(cfg)
    # instantiate model
    model: nn.Module = hydra.utils.instantiate(cfg.model)

    trainer = Trainer(cfg, fabric=fabric, model=model)
    trainer.train()

    print(f"run in {cfg.fabric.global_rank}")


if __name__ == "__main__":
    pretrain()
