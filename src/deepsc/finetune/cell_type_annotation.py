import math
import os

import numpy as np
import scanpy as sc
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import (
    LinearLR,
    SequentialLR,
)
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from deepsc.data.dataset import GeneExpressionDatasetMapped
from src.deepsc.data import DataCollator
from src.deepsc.utils import (
    CosineAnnealingWarmRestartsWithDecayAndLinearWarmup,
    extract_state_dict_with_encoder_prefix,
    report_loading_result,
    sample_weight_norms,
    seed_all,
)


class CellTypeAnnotation:
    def __init__(self, args, fabric, model):
        self.args = args
        self.fabric = fabric
        self.model = model
        self.world_size = self.fabric.world_size
        seed_all(args.seed + self.fabric.global_rank)
        self.is_master = self.fabric.global_rank == 0
        if self.args.pretrained_model_path and self.args.load_pretrained_model:
            self.load_pretrained_model()
        self.optimizer = Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.model, self.optimizer = self.fabric.setup(self.model, self.optimizer)
        self.init_loss_fn()
        self.scheduler = self.create_scheduler(self.optimizer, self.args)

    def init_loss_fn(self):
        if self.args.cls_loss_type == "standard":
            self.criterion_cls = nn.CrossEntropyLoss()
        elif self.args.cls_loss_type == "per_bin":
            self.criterion_cls = self.calculate_mean_ce_loss

    def calculate_per_bin_ce_loss(self, logits, discrete_expr_label, ignore_index=-100):
        """
        logits: (batch, seq_len, num_bins)
        discrete_expr_label: (batch, seq_len)
        返回: (num_bins,) 每个bin的平均交叉熵损失
        计算平均时不包括bin0
        """
        num_bins = self.args.cell_type_count
        ce_losses = []
        logits_flat = logits.reshape(-1, num_bins)
        labels_flat = discrete_expr_label.reshape(-1)
        for i in range(0, num_bins):  # 从bin0开始，到num_bins-1
            # 只统计label为i且不是ignore_index的样本
            mask = (labels_flat == i) & (labels_flat != ignore_index)
            if mask.sum() == 0:
                ce_losses.append(torch.tensor(0.0, device=logits.device))
                continue
            logits_i = logits_flat[mask]
            labels_i = labels_flat[mask]
            ce = nn.CrossEntropyLoss(reduction="mean")
            ce_loss = ce(logits_i, labels_i)
            ce_losses.append(ce_loss)
        if len(ce_losses) == 0:
            return torch.tensor(0.0, device=logits.device)
        return torch.stack(ce_losses)  # (num_bins,)

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

    def calculate_mean_ce_loss(self, logits, discrete_expr_label):
        per_bin_ce_loss = self.calculate_per_bin_ce_loss(logits, discrete_expr_label)
        mean_ce_loss = (
            per_bin_ce_loss.mean()
            if per_bin_ce_loss is not None
            else torch.tensor(0.0, device=logits.device)
        )
        return mean_ce_loss

    def build_dataset_sampler_from_h5ad(self):
        print("Building dataset sampler from h5ad file...")
        # adata = sc.read_h5ad(self.args.data_path)

        # # 根据样本索引划分
        # train_idx, test_idx = train_test_split(
        #     range(adata.n_obs),
        #     test_size=0.1,  # 20% 测试集
        #     random_state=42,  # 固定随机种子
        # )

        # # 划分数据集
        # adata_train = adata[train_idx].copy()
        # adata_test = adata[test_idx].copy()
        adata_train = sc.read_h5ad(self.args.data_path)
        adata_test = sc.read_h5ad(self.args.data_path_eval)

        self.train_dataset = GeneExpressionDatasetMapped(
            h5ad=adata_train,
            csv_path=self.args.csv_path,
            var_name_col=self.args.var_name_in_h5ad,
            obs_celltype_col=self.args.obs_celltype_col,
        )
        self.eval_dataset = GeneExpressionDatasetMapped(
            h5ad=adata_test,
            csv_path=self.args.csv_path,
            var_name_col=self.args.var_name_in_h5ad,
            obs_celltype_col=self.args.obs_celltype_col,
        )
        self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
        self.eval_sampler = DistributedSampler(self.eval_dataset, shuffle=True)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            sampler=self.train_sampler,
            num_workers=8,
            collate_fn=DataCollator(
                do_padding=True,
                pad_token_id=0,
                pad_value=0,
                do_mlm=False,
                do_binning=True,
                max_length=self.args.sequence_length,
                num_genes=self.args.model.num_genes,
                num_bins=self.args.model.num_bins,
                use_max_cell_length=False,
                cell_type=True,
            ),
        )
        self.eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.args.batch_size,
            sampler=self.eval_sampler,
            num_workers=8,
            collate_fn=DataCollator(
                do_padding=True,
                pad_token_id=0,
                pad_value=0,
                do_mlm=False,
                do_binning=True,
                max_length=self.args.sequence_length,
                num_genes=self.args.model.num_genes,
                num_bins=self.args.model.num_bins,
                use_max_cell_length=False,
                cell_type=True,
            ),
        )
        self.train_loader, self.eval_loader = self.fabric.setup_dataloaders(
            self.train_loader, self.eval_loader
        )

    def each_training_iteration(self, data, is_accumulating):
        gene = data["gene"]
        discrete_expr = data["masked_discrete_expr"]
        continuous_expr = data["masked_continuous_expr"]
        cell_type_id = data["cell_type_id"]
        model_outputs = self.model(
            gene_ids=gene,
            value_log1p=continuous_expr,
            value_binned=discrete_expr,  # 使用新映射后的数据
            return_encodings=False,
            return_mask_prob=True,
            return_gate_weights=False,
        )
        logits, regression_output, y, gene_emb, expr_emb, cls_output = model_outputs
        output_dict = {
            "mlm_output": logits,
            "cls_output": cls_output,
            "cell_emb": self.model._get_cell_emb_from_layer(
                self.model.encoder.fused_emb_proj(
                    torch.cat([gene_emb, expr_emb], dim=-1)
                ),
                y.sum(dim=-1).mean(dim=2) if y is not None else None,
            ),
        }
        loss = 0.0
        loss_cls = self.criterion_cls(output_dict["cls_output"], cell_type_id)
        loss = loss + loss_cls

        error_rate = 1 - (
            (output_dict["cls_output"].argmax(1) == cell_type_id).sum().item()
        ) / cell_type_id.size(0)
        if is_accumulating:
            with self.fabric.no_backward_sync(self.model, enabled=is_accumulating):
                self.fabric.backward(loss / self.args.grad_acc)
        else:
            self.fabric.backward(loss / self.args.grad_acc)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1e2)
            self.optimizer.step()
            self.scheduler.step()  # 每次optimizer.step()后更新学习率
            self.optimizer.zero_grad()

        return loss_cls, error_rate

    def evaluate(self, model: nn.Module, loader: DataLoader) -> float:
        model.eval()
        total_loss = 0.0
        total_error = 0.0
        total_dab = 0.0
        total_num = 0
        predictions = []
        cell_type_ids = []
        data_iter = loader
        if self.is_master:
            data_iter = tqdm(
                loader,
                desc="Evaluate[Finetune Cell Type Annotation]",
                ncols=150,
                position=1,
            )
        with torch.no_grad():
            for i, data in enumerate(data_iter):
                gene = data["gene"]
                discrete_expr = data["masked_discrete_expr"]
                continuous_expr = data["masked_continuous_expr"]
                cell_type_id = data["cell_type_id"]
                model_outputs = self.model(
                    gene_ids=gene,
                    value_log1p=continuous_expr,
                    value_binned=discrete_expr,  # 使用新映射后的数据
                    return_encodings=False,
                    return_mask_prob=True,
                    return_gate_weights=False,
                )
                logits, regression_output, y, gene_emb, expr_emb, cls_output = (
                    model_outputs
                )
                output_dict = {
                    "mlm_output": logits,
                    "cls_output": cls_output,
                    "cell_emb": model._get_cell_emb_from_layer(
                        model.encoder.fused_emb_proj(
                            torch.cat([gene_emb, expr_emb], dim=-1)
                        ),
                        y.sum(dim=-1).mean(dim=2) if y is not None else None,
                    ),
                }
                output_values = output_dict["cls_output"]
                loss = self.criterion_cls(output_values, cell_type_id)
                total_loss += loss.item() * len(gene)
                accuracy = (output_values.argmax(1) == cell_type_id).sum().item()
                total_error += (1 - accuracy / len(gene)) * len(gene)
                total_num += len(gene)
                preds = output_values.argmax(1).detach().cpu()
                predictions.append(preds)
                cell_type_ids.append(cell_type_id.detach().cpu())
            from sklearn.metrics import (
                accuracy_score,
                f1_score,
                precision_score,
                recall_score,
            )

            y_true = torch.cat(cell_type_ids, dim=0).numpy()
            y_pred = torch.cat(predictions, dim=0).numpy()
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average="macro")
            recall = recall_score(y_true, y_pred, average="macro")
            macro_f1 = f1_score(y_true, y_pred, average="macro")
            print("唯一的真实 cell_type_ids:", np.unique(y_true))
            print("唯一的预测 predictions:", np.unique(y_pred))

        if self.is_master:
            print(
                f"Evaluation Loss: {total_loss / total_num:.4f}, "
                f"Error Rate: {total_error / total_num:.4f}, "
                f"Accuracy: {accuracy:.4f}, "
                f"Precision: {precision:.4f}, "
                f"Recall: {recall:.4f}, "
                f"Macro F1: {macro_f1:.4f}"
            )

        return total_loss / total_num, total_error / total_num

    def train(self):
        # 首先构建dataloader（如果还没有构建）
        if not hasattr(self, "train_loader"):
            self.build_dataset_sampler_from_h5ad()
        print("train_loader: ", len(self.train_loader.dataset))
        print("eval_loader: ", len(self.eval_loader.dataset))
        start_epoch = self.last_epoch if hasattr(self, "last_epoch") else 1
        for epoch in range(start_epoch, self.args.epoch + 1):
            self.epoch = epoch  # 更新类变量epoch
            self.model.train()
            data_iter = self.train_loader
            if self.is_master:
                data_iter = tqdm(
                    self.train_loader,
                    desc=f"Epoch {epoch} [Finetune Cell Type Annotation]",
                    ncols=150,
                    position=1,
                )
            # 获取第一个batch并打印其结构
            for i, data in enumerate(data_iter):
                is_accumulating = (i + 1) % self.args.grad_acc == 0
                loss_cls, error_rate = self.each_training_iteration(
                    data, is_accumulating
                )
                if self.is_master:
                    data_iter.set_postfix(
                        loss_cls=loss_cls.item(),
                        error_rate=error_rate,
                    )
            self.fabric.print(f"Epoch {epoch} [Finetune Cell Type Annotation]")
            self.fabric.print(f"Epoch {epoch} [Finetune Cell Type Annotation]")
            self.evaluate(self.model, self.eval_loader)

    def load_pretrained_model(self):
        ckpt_path = self.args.pretrained_model_path
        assert os.path.exists(ckpt_path), f"找不到 ckpt: {ckpt_path}"

        # 2) 只在 rank0 读取到 CPU，减少压力；再广播（可选）
        if self.fabric.global_rank == 0:
            print(f"[LOAD] 读取 checkpoint: {ckpt_path}")
            raw = torch.load(ckpt_path, map_location="cpu")
            state_dict = extract_state_dict_with_encoder_prefix(raw)
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
