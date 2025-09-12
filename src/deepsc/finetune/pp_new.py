import math
import os

import numpy as np
import torch
import torch.nn as nn
from gears import PertData
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from tqdm import tqdm

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

    def init_loss_fn(self):
        self.criterion_mse = nn.MSELoss()

    def create_scheduler(self, optimizer, args):

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
        data_name = "adamson"
        split = "simulation"
        pert_data = PertData("./data")  # TODO: change to the data path
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

    def map_raw_id_to_vocab_id(self, raw_ids, gene_ids):
        """
        Map raw gene indices to vocab indices, equivalent to scGPT's map_raw_id_to_vocab_id

        Args:
            raw_ids: torch.Tensor, raw gene indices (positions in dataset)
            gene_ids: torch.Tensor or np.ndarray, vocab indices for each gene

        Returns:
            torch.Tensor: mapped vocab indices
        """
        if isinstance(raw_ids, torch.Tensor):
            device = raw_ids.device
            dtype = raw_ids.dtype

            # Ensure gene_ids is on the same device
            if isinstance(gene_ids, torch.Tensor):
                gene_ids = gene_ids.to(device)
            else:
                gene_ids = torch.as_tensor(gene_ids, device=device)

            # Map raw indices to vocab indices
            mapped_ids = gene_ids[raw_ids]
            return mapped_ids.to(dtype=dtype)
        else:
            raise ValueError("raw_ids must be torch.Tensor")

    def construct_pert_flags_for_pert(self, data, batch_size, device):
        """构造每个样本的扰动标记矩阵"""
        pert_flags = torch.zeros(
            batch_size, self.num_genes, device=device, dtype=torch.long
        )

        for r, p in enumerate(data.pert):
            for g in p.split("+"):
                if g and g != "ctrl":
                    j = self.name2col.get(g, -1)
                    if j != -1:
                        pert_flags[r, j] = 1

        return pert_flags

    def construct_pert_flags_for_de(self, data, batch_size, device):
        """构造每个样本的扰动标记矩阵"""
        de_flags = torch.zeros(
            batch_size, self.num_genes, device=device, dtype=torch.long
        )

        for r, p in enumerate(data.de_idx):
            for g_idx in p:
                de_flags[r, g_idx] = 1

        return de_flags

    def _construct_pert_flags(self, data, batch_size, device):
        """构造每个样本的扰动标记矩阵"""
        pert_flags_full = torch.zeros(
            batch_size, self.num_genes, device=device, dtype=torch.long
        )

        # 标记原始扰动基因 (pert_flags=1)
        for r, p in enumerate(data.pert):
            for g in p.split("+"):
                if g and g != "ctrl":
                    j = self.name2col.get(g, -1)
                    if j != -1:
                        pert_flags_full[r, j] = 1

        # 标记受扰动影响较大的基因 (pert_flags=2)
        for r, p in enumerate(data.de_idx):
            for g_idx in p:
                pert_flags_full[r, g_idx] = 2

        return pert_flags_full

    def _discretize_expression(self, input_values, num_bins=5):
        """对表达值进行离散化分箱"""
        batch_size = input_values.shape[0]
        discrete_input_bins = torch.zeros_like(input_values, dtype=torch.long)

        for i in range(batch_size):
            row_vals = input_values[i]
            valid_mask = row_vals != -1.0
            if valid_mask.any():
                valid_vals = row_vals[valid_mask]
                min_val = valid_vals.min()
                max_val = valid_vals.max()
                norm = (valid_vals - min_val) / (max_val - min_val + 1e-8)
                bins = torch.floor(norm * (num_bins - 1)).long()
                bins = torch.clamp(bins, 0, num_bins - 1) + 1
                discrete_input_bins[i][valid_mask] = bins

        return discrete_input_bins

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
                pert_flags = self.construct_pert_flags_for_pert(
                    batch_data, batch_size, device
                )
                de_flags = self.construct_pert_flags_for_de(
                    batch_data, batch_size, device
                )
                all_pert_flags = self._construct_pert_flags(
                    batch_data, batch_size, device
                )
                x: torch.Tensor = batch_data.x  # (batch_size * n_genes, 2)
                ori_gene_values = x[:, 0].view(batch_size, self.num_genes)
                target_gene_values = batch_data.y  # (batch_size, n_genes)
                if self.args.include_zero_gene in ["all", "batch-wise"]:
                    if self.args.include_zero_gene == "all":
                        input_gene_ids = torch.arange(
                            self.num_genes, device=device, dtype=torch.long
                        )
                    else:
                        input_gene_ids = (
                            ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
                        )  # 得到整个batch里面没有0表达的基因原始索引
                    if len(input_gene_ids) > self.args.data_length:
                        input_gene_ids = torch.randperm(
                            len(input_gene_ids), device=device
                        )[: self.args.data_length]
                    input_values = ori_gene_values[:, input_gene_ids]
                    target_values = target_gene_values[:, input_gene_ids]
                    input_pert_flags = pert_flags[:, input_gene_ids]
                    all_pert_flags = all_pert_flags[:, input_gene_ids]
                    discrete_input_bins = self._discretize_expression(input_values)
                    # Apply gene ID mapping like scGPT
                    mapped_input_gene_ids = self.map_raw_id_to_vocab_id(
                        input_gene_ids, self.gene_ids
                    )
                    mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

                    regression_output, y, gene_emb, expr_emb = self.model(
                        gene_ids=mapped_input_gene_ids,
                        expression_bin=discrete_input_bins,
                        normalized_expr=input_values,
                        input_pert_flags=all_pert_flags,
                    )
                    loss = self.criterion_mse(regression_output, target_values)

                    self.fabric.backward(loss)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1e2)
                    self.optimizer.step()
                    self.scheduler.step()  # 每次optimizer.step()后更新学习率
                    self.optimizer.zero_grad()

                    # 累积损失用于记录
                    total_loss += loss.item()
                    total_mse += loss.item()

                    # 更新进度条 - 这就是您要添加的代码！
                    if self.is_master:
                        data_iter.set_postfix(
                            loss=loss.item(),
                            avg_loss=total_loss / (batch + 1),
                        )
            res = self.evaluate()
            val_metrics = compute_perturbation_metrics(
                res,
                self.pert_data.adata[self.pert_data.adata.obs["condition"] == "ctrl"],
            )
            print("val_metrics at epoch 1: ")
            print(val_metrics)

    def evaluate(self):
        self.model.eval()
        pert_cat = []
        pred = []
        truth = []
        pred_de = []
        truth_de = []
        results = {}
        data_iter = self.valid_loader
        if self.is_master:
            data_iter = tqdm(
                self.valid_loader,
                desc="EVAL ",
                ncols=150,
                position=1,
            )
        with torch.no_grad():
            for batch, batch_data in enumerate(data_iter):
                batch_size = len(batch_data.y)
                pert_cat.extend(batch_data.pert)
                t = batch_data.y
                device = batch_data.x.device
                all_pert_flags = self._construct_pert_flags(
                    batch_data, batch_size, device
                )
                x: torch.Tensor = batch_data.x  # (batch_size * n_genes, 2)
                ori_gene_values = x[:, 0].view(batch_size, self.num_genes)
                if self.args.include_zero_gene in ["all", "batch-wise"]:
                    if self.args.include_zero_gene == "all":
                        input_gene_ids = torch.arange(
                            self.num_genes, device=device, dtype=torch.long
                        )
                    else:
                        input_gene_ids = (
                            ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
                        )  # 得到整个batch里面没有0表达的基因原始索引
                    if len(input_gene_ids) > self.args.data_length:
                        input_gene_ids = torch.randperm(
                            len(input_gene_ids), device=device
                        )[: self.args.data_length]
                    input_values = ori_gene_values[:, input_gene_ids]
                    input_pert_flags = all_pert_flags[:, input_gene_ids]
                    discrete_input_bins = self._discretize_expression(input_values)
                    # Apply gene ID mapping like scGPT
                    mapped_input_gene_ids = self.map_raw_id_to_vocab_id(
                        input_gene_ids, self.gene_ids
                    )
                    mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

                    regression_output, _, _, _ = self.model(
                        gene_ids=mapped_input_gene_ids,
                        expression_bin=discrete_input_bins,
                        normalized_expr=input_values,
                        input_pert_flags=input_pert_flags,
                    )
                    pred_gene_values = ori_gene_values.clone()  # 使用原始值作为基础
                    pred_gene_values[:, input_gene_ids] = (
                        regression_output  # 只更新选中基因的预测
                    )

                    pred.extend(pred_gene_values.cpu())
                    truth.extend(t.cpu())
                    for itr, de_idx in enumerate(batch_data.de_idx):
                        pred_de.append(pred_gene_values[itr, de_idx])
                        truth_de.append(t[itr, de_idx])
        results["pert_cat"] = np.array(pert_cat)
        pred = torch.stack(pred)
        truth = torch.stack(truth)
        results["pred"] = pred.detach().cpu().numpy().astype(np.float64)
        results["truth"] = truth.detach().cpu().numpy().astype(np.float64)
        results["pred_de"] = (
            torch.stack(pred_de).detach().cpu().numpy().astype(np.float64)
        )
        results["truth_de"] = (
            torch.stack(truth_de).detach().cpu().numpy().astype(np.float64)
        )
        return results

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
