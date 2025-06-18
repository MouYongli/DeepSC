import hydra
import torch
from omegaconf import OmegaConf


@hydra.main(
    version_base=None, config_path="../../configs/pretrain", config_name="pretrain"
)
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    # 用hydra.utils.instantiate实例化模型
    import hydra.utils

    model = hydra.utils.instantiate(cfg.model)
    print(model)

    # 测试前向传播
    batch_size = 2
    g = cfg.model.num_genes
    gene_ids = torch.randint(0, cfg.model.num_genes, (batch_size, g))
    expression = torch.rand(batch_size, g)

    # 统计可学习参数数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of learnable parameters: {total_params}")


if __name__ == "__main__":
    main()
