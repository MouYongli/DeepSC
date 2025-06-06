from lightning.fabric import Fabric

import argparse
from deepsc.models.scbert import PerformerLM
from deepsc.train.trainer import Trainer
from deepsc.utils.utils import setup_logging


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--bin_num", type=int, default=5, help="number of bins")
    parser.add_argument(
        "--model_type",
        type=str,
        default="scbert",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/angli/baseline/DeepSC/data/3ac/mapped_batch_data/"
        "1d84333c-0327-4ad6-be02-94fee81154ff_sparse_preprocessed.pth",
        help="Path of data for pretraining.",
    )
    # TODO: if path is dir or file, handle differently
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Number of batch size."
    )
    parser.add_argument("--epoch", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--gene_num", type=int, default=60664, help="Number of genes.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate."
    )
    parser.add_argument(
        "--grad_acc", type=int, default=32, help="Number of gradient accumulation."
    )
    parser.add_argument(
        "--valid_every",
        type=int,
        default=1,
        help="Number of training epochs between twice validation.",
    )
    parser.add_argument(
        "--pos_embed", action="store_true", help="Using Gene2vec encoding or not."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="llm_pretrained",
        help="Pretrained model name.",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="./ckpts",
        help="Directory of checkpoint to save.",
    )
    parser.add_argument(
        "--mask_prob", type=float, default=0.15, help="Probability of masking."
    )
    parser.add_argument(
        "--replace_prob",
        type=float,
        default=0.9,
        help="Probability of replacing with [MASK] token for masking.",
    )
    parser.add_argument(
        "--num_device",
        type=int,
        default=2,
        help="number of devices that should be included",
    )
    # TODO: check if mask_prob and replace_prob necessary in all models
    return parser.parse_args()


def select_model(args):
    if args.model_type == "scbert":
        return PerformerLM(
            num_tokens=args.bin_num + 2,
            dim=200,
            depth=6,
            max_seq_len=args.gene_num + 1,
            heads=10,
            local_attn_heads=0,
            g2v_position_emb=args.pos_embed,
        )
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")


def pretrain():
    args = parse_args()

    """     master_addr = os.getenv("MASTER_ADDR", "127.0.0.1")
    master_port = os.getenv("MASTER_PORT", "12345")
    rank = int(os.getenv("SLURM_PROCID", "0"))
    world_size = int(os.getenv("SLURM_NTASKS", "1"))
    local_rank = int(os.getenv("SLURM_LOCALID", "-1")) """

    # 初始化通信组，在fabric的时候可以省略
    """  dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        rank=rank,
        world_size=world_size
    ) """
    # 设置进程要使用的GPU 在使用fabric的时候可以省略
    # torch.cuda.set_device(local_rank)

    fabric = Fabric(accelerator="cuda", devices=args.num_device, strategy="ddp")
    fabric.launch()

    args.fabric = fabric
    setup_logging(fabric.global_rank, "./logs")
    args.model = select_model(args)
    trainer = Trainer(args)
    trainer.train()

    print(f"run in {args.fabric.global_rank}")


if __name__ == "__main__":
    pretrain()
