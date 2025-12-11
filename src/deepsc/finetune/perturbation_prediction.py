"""
Perturbation Prediction Fine-tuning with Hydra Configuration
Uses existing pp.yaml configuration from /home/angli/DeepSC/configs/pp/
"""

import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from gears import PertData
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from tqdm import tqdm

import hydra
from hydra.utils import instantiate
from lightning.fabric import Fabric
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from deepsc.utils import (
    build_gene_ids_for_dataset,
    build_vocab_from_csv,
    extract_state_dict,
    report_loading_result,
    sample_weight_norms,
)
from src.deepsc.utils import (
    CosineAnnealingWarmRestartsWithDecayAndLinearWarmup,
    compute_perturbation_metrics,
    seed_all,
)

class PPNEW:
    # Question:在这个脚本中，基因的顺序为乱序，不确定这样会不会对结果有影响
    def __init__(self, args, fabric, model):
        self.args = args
        self.fabric = fabric
        self.model = model
        self.world_size = self.fabric.world_size
        seed_all(args.seed + self.fabric.global_rank)
        self.is_master = self.fabric.global_rank == 0
        self.setup_output_directory()
        if args.pretrained_model:
            self.load_pretrained_model()

        self.optimizer = Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.model, self.optimizer = self.fabric.setup(self.model, self.optimizer)
        self.init_loss_fn()
        self.scheduler = self.create_scheduler(self.optimizer, self.args)
        self.vocab, self.id2vocab, self.pad_token, self.pad_value = (
            build_vocab_from_csv(self.args.csv_path)
        )
        self.prepare_data()
        self.gene_ids = build_gene_ids_for_dataset(self.original_genes, self.vocab)
        self.name2col = {g: i for i, g in enumerate(self.original_genes)}
        self.valid_gene_mask = self.gene_ids != 0
        # 创建有效基因的索引列表（只包含在vocab中有对应的基因）
        self.valid_gene_ids = torch.arange(self.num_genes)[self.valid_gene_mask]
        self.num_valid_genes = len(self.valid_gene_ids)
        if self.is_master:
            print(f"Total genes in dataset: {self.num_genes}")
            print(f"Valid genes (in vocab): {self.num_valid_genes}")
            print(f"Invalid genes (not in vocab): {self.num_genes - self.num_valid_genes}")

        self.perts_to_plot = ["KCTD16+ctrl"]
        # 比较self.name2col和self.node_map是不是内容一样的字典
        if hasattr(self, "node_map"):
            name2col_equal = self.name2col == self.node_map
            if self.is_master:
                print("self.name2col == self.node_map:", name2col_equal)
        else:
            if self.is_master:
                print("self does not have attribute 'node_map'")
        # print(torch.unique(self.gene_ids, return_counts=True))
        # unique_vals, counts = np.unique(self.gene_ids, return_counts=True)
        # print("unique_vals:",unique_vals)
        # print("counts:",counts)

    def setup_output_directory(self):
        """
        使用Hydra的输出目录,并在其中创建checkpoints、logs和visualizations子目录
        """
        if self.is_master:
            # 使用Hydra的输出目录(通过HydraConfig获取,即使在fork后的子进程中也有效)
            from hydra.core.hydra_config import HydraConfig

            try:
                hydra_cfg = HydraConfig.get()
                self.output_dir = hydra_cfg.runtime.output_dir
            except:
                # 如果不是在Hydra下运行,使用当前目录
                self.output_dir = os.getcwd()

            # 创建checkpoints、logs和visualizations子目录
            self.ckpt_dir = os.path.join(self.output_dir, "checkpoints")
            self.log_dir = os.path.join(self.output_dir, "logs")
            self.vis_dir = os.path.join(self.output_dir, "visualizations")
            os.makedirs(self.ckpt_dir, exist_ok=True)
            os.makedirs(self.log_dir, exist_ok=True)
            os.makedirs(self.vis_dir, exist_ok=True)

            print(f"Output directory: {self.output_dir}")
            print(f"  - Checkpoints: {self.ckpt_dir}")
            print(f"  - Logs: {self.log_dir}")
            print(f"  - Visualizations: {self.vis_dir}")
        else:
            # 非master进程设置为None
            self.output_dir = None
            self.ckpt_dir = None
            self.log_dir = None
            self.vis_dir = None

        # 广播输出目录到所有进程
        self.output_dir = self.fabric.broadcast(self.output_dir, src=0)
        self.ckpt_dir = self.fabric.broadcast(self.ckpt_dir, src=0)
        self.log_dir = self.fabric.broadcast(self.log_dir, src=0)
        self.vis_dir = self.fabric.broadcast(self.vis_dir, src=0)
        
    def load_pretrained_model(self):
        ckpt_path = self.args.pretrained_model_path
        assert os.path.exists(ckpt_path), f"找不到 ckpt: {ckpt_path}"

        # 2) 只在 rank0 读取到 CPU，减少压力；再广播（可选）
        if self.fabric.global_rank == 0:
            print(f"[LOAD] 读取 checkpoint: {ckpt_path}")
            raw = torch.load(ckpt_path, map_location="cpu")
            state_dict = extract_state_dict(raw)
        else:
            raw = None
            state_dict = None

        # 3) 广播到所有进程
        state_dict = self.fabric.broadcast(state_dict, src=0)
        # 4) 打印抽样对比（可选，但很直观）
        sample_weight_norms(self.model, state_dict, k=5)

        # 5) 真正加载（strict=False：允许你新增的 embedding 层留空）
        load_info = self.model.load_state_dict(state_dict, strict=False)
        report_loading_result(load_info)
    
    def create_scheduler(self, optimizer, args):

        # 如果使用恒定学习率，返回None
        if getattr(args, "use_constant_lr", False):
            if self.is_master:
                print("Using constant learning rate (no scheduler)")
            return None

        total_steps = args.epoch * math.ceil(
            (100000) / (args.batch_size * self.world_size * args.grad_acc)
        )
        warmup_ratio = self.args.warmup_ratio
        warmup_steps = math.ceil(total_steps * warmup_ratio)
        main_steps = total_steps - warmup_steps
        linear_warmup = LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
        )
        cosine_anneal = CosineAnnealingWarmRestartsWithDecayAndLinearWarmup(
            optimizer,
            T_0=warmup_steps * 3,
            T_mult=1,
            warmup_steps=0,
            decay=0.9,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[linear_warmup, cosine_anneal],
            milestones=[warmup_steps],
        )
        return scheduler
    
    def prepare_data(self):
        # 从配置文件读取data_name,如果没有配置则使用默认值"norman"
        data_name = getattr(self.args, "data_name", "norman")
        split = getattr(self.args, "split", "simulation")
        data_path = getattr(self.args, "data_path", "./data")

        pert_data = PertData(data_path)
        self.pert_data = pert_data
        pert_data.load(data_name=data_name)
        pert_data.prepare_split(split=split, seed=1)
        pert_data.get_dataloader(
            batch_size=self.args.batch_size, test_batch_size=self.args.batch_size
        )
        self.original_genes = pert_data.adata.var["gene_name"].tolist()
        self.num_genes = len(self.original_genes)

        self.train_loader = pert_data.dataloader["train_loader"]
        self.valid_loader = pert_data.dataloader["val_loader"]
        self.train_loader, self.valid_loader = self.fabric.setup_dataloaders(
            self.train_loader, self.valid_loader
        )
        self.node_map = pert_data.node_map
    
    def init_loss_fn(self):
        self.criterion_mse = nn.MSELoss()

    def construct_pert_flags(self, batch_data, batch_size, n_genes, device):
        """
        Construct perturbation flags from batch_data.pert for new version of gears.

        Args:
            batch_data: Batch data from gears dataloader
            batch_size: Batch size
            n_genes: Number of genes
            device: Torch device

        Returns:
            pert_flags: torch.Tensor of shape (batch_size, n_genes) with 1 for perturbed genes, 0 otherwise
        """
        pert_flags = torch.zeros(batch_size, n_genes, device=device, dtype=torch.long)

        for r, p in enumerate(batch_data.pert):
            for g in p.split("+"):
                if g and g != "ctrl":
                    j = self.name2col.get(g, -1)
                    if j != -1:
                        pert_flags[r, j] = 1

        return pert_flags


    def train(self):
        """
        Train the model for one epoch.
        """
        for epoch in range(self.args.epoch):
            self.model.train()
            total_loss, total_mse = 0.0, 0.0

            # 设置进度条
            data_iter = self.train_loader
            if self.is_master:
                data_iter = tqdm(
                    self.train_loader,
                    desc="TRAIN",
                    ncols=150,
                    position=0,
                )
            for batch, batch_data in enumerate(data_iter):
                batch_size = len(batch_data.y)
                device = batch_data.x.device
                x: torch.Tensor = batch_data.x  # (batch_size * n_genes, 2)
                ori_gene_values = x[:, 0].view(batch_size, self.num_genes)
                target_gene_values = batch_data.y  # (batch_size, n_genes)
                pert_flags = self.construct_pert_flags(batch_data, batch_size, self.num_genes, device)
                print(batch_data.pert_idx)
                print(batch_data.pert)
                print("Training batch:", batch)
                print("Batch data keys:", batch_data.keys())
                if self.args.include_zero_gene in ["all", "batch-wise"]:
                    if self.args.include_zero_gene == "all":
                        # 只使用有效基因（在vocab中有对应的基因）
                        input_gene_ids = self.valid_gene_ids.to(device)
                    else:
                        # 得到在ori_gene_values或target_gene_values(t)中至少有一个非零的基因索引
                        ori_nonzero_gene_ids = (
                            ori_gene_values.nonzero()[:, 1].flatten().unique()
                        )
                        target_nonzero_gene_ids = target_gene_values.nonzero()[:, 1].flatten().unique()
                        # 合并两个集合，取并集
                        nonzero_gene_ids = (
                            torch.cat([ori_nonzero_gene_ids, target_nonzero_gene_ids])
                            .unique()
                            .sort()[0]
                        )
                        # 过滤掉无效基因：只保留在vocab中有对应的基因
                        valid_mask = torch.isin(nonzero_gene_ids, self.valid_gene_ids.to(device))
                        input_gene_ids = nonzero_gene_ids[valid_mask]
                    if len(input_gene_ids) > self.args.data_length:
                        input_gene_ids = torch.randperm(
                            len(input_gene_ids), device=device
                        )[: self.args.data_length]
                input_values = ori_gene_values[:, input_gene_ids]
                input_pert_flags = pert_flags[:, input_gene_ids]
                target_values = target_gene_values[:, input_gene_ids]
                print("Input gene ids:", input_gene_ids)
                print("Input values shape:", input_values.shape)
                print("Input pert flags shape:", input_pert_flags.shape)
                print("Target values shape:", target_values.shape)
                break

@hydra.main(version_base=None, config_path="../../../configs/pp", config_name="pp")
def main(cfg: DictConfig):
    """
    Main training function using Hydra configuration

    Args:
        cfg: Hydra configuration loaded from configs/pp/pp.yaml
    """

    # Print configuration
    print("=" * 80)
    print("DeepSC Perturbation Prediction Fine-tuning (PPNEW)")
    print("=" * 80)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # Setup Fabric for distributed training
    # Determine precision based on config
    if hasattr(cfg, 'amp') and cfg.amp:
        precision = "16-mixed"
    else:
        precision = "32-true"

    fabric = Fabric(
        accelerator="auto",
        devices=getattr(cfg, 'num_device', 1),
        precision=precision,
    )

    # Launch distributed setup
    fabric.launch()

    if fabric.global_rank == 0:
        print(f"\nDataset: {cfg.data_name}")
        print(f"Pretrained model: {cfg.pretrained_model}")
        if cfg.pretrained_model:
            print(f"Pretrained path: {cfg.pretrained_model_path}")
        print(f"Batch size: {cfg.batch_size}")
        print(f"Learning rate: {cfg.learning_rate}")
        print(f"Epochs: {cfg.epoch}")
        print(f"Devices: {cfg.num_device}")
        print(f"Seed: {cfg.seed}")
        print(f"Include zero gene: {cfg.include_zero_gene}")
        print(f"Data length: {cfg.data_length}")
        print("=" * 80)

    # Instantiate model using Hydra
    # The model config is in configs/pp/model/deepsc.yaml
    if fabric.global_rank == 0:
        print("\nInstantiating model...")

    model = instantiate(cfg.model)

    if fabric.global_rank == 0:
        print(f"Model instantiated: {type(model).__name__}")
        print(f"  - Embedding dim: {cfg.model.embedding_dim}")
        print(f"  - Num layers: {cfg.model.num_layers}")
        print(f"  - Num heads: {cfg.model.num_heads}")
        print(f"  - Use MoE regressor: {cfg.model.use_moe_regressor}")

    # Initialize PPNEW predictor
    predictor = PPNEW(
        args=cfg,
        fabric=fabric,
        model=model
    )

    # Train the model
    if fabric.global_rank == 0:
        print("\nStarting training...")

    # Note: The PPNEW class needs to implement train() and plot_predictions() methods
    # predictor.train()

    # # Generate visualization plots
    # if fabric.global_rank == 0:
    #     print("\nGenerating visualization plots...")
    #     predictor.plot_predictions()

    #     print("\nTraining completed!")
    #     print(f"Results saved to: {predictor.output_dir}")

    if fabric.global_rank == 0:
        print("\nPPNEW predictor initialized successfully!")
        print("Note: Please implement train() and plot_predictions() methods in PPNEW class")

    predictor.train()
if __name__ == "__main__":
    main()
