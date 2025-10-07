import hydra
import torch.nn as nn
from lightning.fabric import Fabric
from lightning.fabric.strategies import DDPStrategy
from omegaconf import DictConfig

from deepsc.finetune.grn_inference import GRNInference
from deepsc.utils.utils import setup_logging


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

    # model = select_model(cfg)
    # instantiate model
    model: nn.Module = hydra.utils.instantiate(cfg.model)
    model = model.float()
    trainer = GRNInference(cfg, fabric=fabric, model=model)
    trainer.inference()
    # test_ckpt = TestCkpt(cfg, fabric=fabric, model=model)


if __name__ == "__main__":
    finetune()
