import hydra
import torch.nn as nn
from lightning.fabric import Fabric
from omegaconf import DictConfig

from deepsc.models.scbert import PerformerLM
from deepsc.train.trainer import Trainer
from deepsc.utils.utils import setup_logging


def select_model(args):
    if args.model_type == "scbert":
        return PerformerLM(
            num_tokens=args.num_bin + 2,
            dim=args.hidden_dim,
            depth=6,
            max_seq_len=args.num_gene + 1,
            heads=10,
            local_attn_heads=0,
            g2v_position_emb=args.pos_embed,
        )
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")


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
