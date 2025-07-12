import hydra
import torch
from gears import PertData
from omegaconf import DictConfig


@hydra.main(
    version_base=None,
    config_path="/home/angli/baseline/DeepSC/configs/finetune/",
    config_name="finetune",
)
def main(cfg: DictConfig):
    # 用hydra配置实例化PertData
    pert_data = PertData(cfg.task.data_path)
    pert_data.load(data_name=cfg.task.data_name)
    print(pert_data.adata)
    pert_data.prepare_split(split=cfg.task.split, seed=1)
    pert_data.get_dataloader(
        batch_size=cfg.batch_size, test_batch_size=cfg.test_batch_size
    )
    # 加载特殊token，比如eoc, pad到基因-id的vocab，
    # 检查数据集中的每个基因是否在vocab里面，把结果记到adata里面
    # 获取adata里面的所有基因名
    # 将没在gene-vocab里面出现的基因名记为pad
    print(pert_data.dataloader)

    # %%
    train_loader = pert_data.dataloader["train_loader"]

    # %%
    for batch, batch_data in enumerate(train_loader):
        # gear 上的载入pert flag的方法已经更改，需要适配
        print(batch)
        print(batch_data)
        print(len(batch_data.y))
        print(batch_data.x.shape)
        print(batch_data.y.shape)
        x: torch.Tensor = batch_data.x  # (batch_size * n_genes, 2)
        ori_gene_values = x[:, 0]
        target_gene_values = batch_data.y  # (batch_size, n_genes)
        print(ori_gene_values.shape)
        print(target_gene_values.shape)
        # 最后传入scgpt的是每个geneid和pert_flag的map，每个geneid和original value的map,
        # 每个geneid和target_value的map用于loss的计算

        # 计算pearson系数在evaluation里面，
        # 在evaluation里面，模型调用model.pred_perturb(),然后返回pred_和truth两部分，调用pearson的函数来计算
        # 还用了patience来决定early stop
        break


if __name__ == "__main__":
    main()
