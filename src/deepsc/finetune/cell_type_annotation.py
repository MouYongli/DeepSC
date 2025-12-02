import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.optim.lr_scheduler import (
    LinearLR,
    SequentialLR,
)
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from datetime import datetime
from deepsc.data.dataset import (  # GeneExpressionDatasetMapped,; create_global_celltype_mapping,
    GeneExpressionDatasetMappedWithGlobalCelltype,
)
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
        self.epoch = 0  # 初始化epoch变量用于绘图

        # 创建带时间戳的输出目录
        self.setup_output_directory()

        # 首先构建数据集以确定正确的celltype数量
        self.build_dataset_sampler_from_h5ad()

        if self.args.pretrained_model_path and self.args.load_pretrained_model:
            self.load_pretrained_model()

        # Configure fine-tuning mode
        self.setup_finetune_mode()

        self.optimizer = Adam(
            self.get_trainable_parameters(), lr=self.args.learning_rate
        )
        self.model, self.optimizer = self.fabric.setup(self.model, self.optimizer)
        self.init_loss_fn()
        self.scheduler = self.create_scheduler(self.optimizer, self.args)

    def setup_output_directory(self):
        """
        创建带时间戳的输出目录用于保存checkpoint和log
        """
        if self.is_master:
            # 创建基础目录
            base_dir = "/home/angli/DeepSC/results/cell_type_annotation"
            os.makedirs(base_dir, exist_ok=True)

            # 创建带时间戳的子目录
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = os.path.join(base_dir, timestamp)
            os.makedirs(self.output_dir, exist_ok=True)

            # 创建checkpoints和logs子目录
            self.ckpt_dir = os.path.join(self.output_dir, "checkpoints")
            self.log_dir = os.path.join(self.output_dir, "logs")
            self.vis_dir = os.path.join(self.output_dir, "visualizations")
            os.makedirs(self.ckpt_dir, exist_ok=True)
            os.makedirs(self.log_dir, exist_ok=True)
            os.makedirs(self.vis_dir, exist_ok=True)

            print(f"Output directory created: {self.output_dir}")
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

    def setup_finetune_mode(self):
        """
        Configure the fine-tuning mode based on the configuration.
        Supports two modes:
        - full: Full parameter fine-tuning
        - head_only: Only fine-tune the classification head
        """
        finetune_mode = getattr(self.args, "finetune_mode", "full")

        if finetune_mode == "full":
            # Full fine-tuning: train all parameters
            if self.is_master:
                print("=" * 80)
                print("Fine-tuning Mode: FULL PARAMETER FINE-TUNING")
                print("All model parameters will be trained.")
                print("=" * 80)
            # All parameters are trainable by default, no need to change

        elif finetune_mode == "head_only":
            # Head-only fine-tuning: freeze encoder, only train classification head
            if self.is_master:
                print("=" * 80)
                print("Fine-tuning Mode: CLASSIFICATION HEAD ONLY")
                print("Only the classification head will be trained.")
                print("=" * 80)

            # Freeze all encoder parameters
            for param in self.model.encoder.parameters():
                param.requires_grad = False

            # Ensure cls_decoder parameters are trainable
            for param in self.model.cls_decoder.parameters():
                param.requires_grad = True
        else:
            raise ValueError(
                f"Unknown finetune_mode: {finetune_mode}. "
                f"Must be one of: 'full', 'head_only'"
            )

    def get_trainable_parameters(self):
        """
        Get the list of trainable parameters based on the fine-tuning mode.
        """
        finetune_mode = getattr(self.args, "finetune_mode", "full")

        if finetune_mode == "head_only":
            # Only return cls_decoder parameters
            trainable_params = self.model.cls_decoder.parameters()
        else:
            trainable_params = self.model.parameters()

        # Count and print trainable parameters
        if self.is_master:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params_list = list(trainable_params)
            trainable_count = sum(p.numel() for p in trainable_params_list)
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_count:,}")
            print(f"Trainable ratio: {trainable_count / total_params * 100:.2f}%")
            print("=" * 80)
            return iter(trainable_params_list)

        return trainable_params

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
        num_bins = self.cell_type_count
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
        """
        Create learning rate scheduler based on configuration.

        Supported scheduler types (controlled by args.lr_scheduler_type):
        - 'constant': No scheduler, constant learning rate
        - 'cosine': Cosine annealing with warmup and restarts (default)
        """
        scheduler_type = getattr(args, "lr_scheduler_type", "cosine")

        if scheduler_type == "constant":
            # No scheduler needed for constant learning rate
            if self.is_master:
                print("=" * 80)
                print(f"Using CONSTANT learning rate: {args.learning_rate}")
                print("No learning rate scheduling will be applied.")
                print("=" * 80)
            return None

        elif scheduler_type == "cosine":
            # Original cosine annealing with warmup
            total_steps = args.epoch * math.ceil(
                (100000) / (args.batch_size * self.world_size * args.grad_acc)
            )
            warmup_ratio = self.args.warmup_ratio
            warmup_steps = math.ceil(total_steps * warmup_ratio)

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
            if self.is_master:
                print("=" * 80)
                print("Using COSINE ANNEALING with warmup:")
                print(f"  - Warmup steps: {warmup_steps}")
                print(f"  - Total steps: {total_steps}")
                print(f"  - Initial LR: {args.learning_rate}")
                print("=" * 80)
            return scheduler

        else:
            raise ValueError(
                f"Unknown lr_scheduler_type: {scheduler_type}. "
                f"Must be one of: 'constant', 'cosine'"
            )

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

        # 检查是否使用分开的数据集
        use_separated_datasets = getattr(
            self.args, "seperated_train_eval_dataset", True
        )

        if use_separated_datasets:
            # 使用两个分开的数据集
            print("Using separated train/eval datasets...")
            adata_train = sc.read_h5ad(self.args.data_path)
            adata_test = sc.read_h5ad(self.args.data_path_eval)
        else:
            # 使用单个数据集进行分层采样分割
            print("Using single dataset with stratified split...")
            adata = sc.read_h5ad(self.args.data_path)

            # 获取细胞类型标签
            cell_types = adata.obs[self.args.obs_celltype_col].astype(str)

            # 进行分层采样，确保每个细胞类型在训练集和测试集中的比例相同
            test_size = getattr(self.args, "test_size", 0.2)  # 默认20%测试集
            train_idx, test_idx = train_test_split(
                range(adata.n_obs),
                test_size=test_size,
                random_state=42,
                stratify=cell_types,  # 按细胞类型分层采样
            )

            # 分割数据集
            adata_train = adata[train_idx].copy()
            adata_test = adata[test_idx].copy()

            print(
                f"分层采样结果：训练集 {len(train_idx)} 个细胞，测试集 {len(test_idx)} 个细胞"
            )

            # 验证分层采样效果
            train_type_counts = adata_train.obs[
                self.args.obs_celltype_col
            ].value_counts()
            test_type_counts = adata_test.obs[self.args.obs_celltype_col].value_counts()
            print("分层采样验证（训练集 vs 测试集比例）:")
            for celltype in train_type_counts.index:
                if celltype in test_type_counts.index:
                    train_ratio = train_type_counts[celltype] / len(train_idx)
                    test_ratio = test_type_counts[celltype] / len(test_idx)
                    print(f"  {celltype}: {train_ratio:.3f} vs {test_ratio:.3f}")

        print(f"训练数据集大小: {adata_train.n_obs}")
        print(f"测试数据集大小: {adata_test.n_obs}")

        # 获取训练集和测试集的细胞类型交集（只训练和评估共同拥有的类型）
        train_celltypes = set(
            adata_train.obs[self.args.obs_celltype_col].astype(str).unique()
        )
        test_celltypes = set(
            adata_test.obs[self.args.obs_celltype_col].astype(str).unique()
        )

        # 计算交集 - 这是我们真正关心的细胞类型
        common_celltypes = train_celltypes & test_celltypes
        print(f"训练集细胞类型数量: {len(train_celltypes)}")
        print(f"测试集细胞类型数量: {len(test_celltypes)}")
        print(f"共同细胞类型数量: {len(common_celltypes)}")
        print(f"共同细胞类型: {sorted(common_celltypes)}")

        # 基于交集创建映射表（只包含交集中的类型）
        self.common_celltypes = sorted(common_celltypes)
        self.type2id = {
            celltype: idx for idx, celltype in enumerate(self.common_celltypes)
        }
        self.id2type = {
            idx: celltype for idx, celltype in enumerate(self.common_celltypes)
        }

        # 细胞类型数量就是交集大小
        self.cell_type_count = len(common_celltypes)
        print(f"使用细胞类型数量（交集）: {self.cell_type_count}")

        # 过滤训练和测试数据，只保留交集中的细胞类型
        print("Filtering datasets to keep only common cell types...")
        train_mask = (
            adata_train.obs[self.args.obs_celltype_col]
            .astype(str)
            .isin(common_celltypes)
        )
        test_mask = (
            adata_test.obs[self.args.obs_celltype_col]
            .astype(str)
            .isin(common_celltypes)
        )

        adata_train_filtered = adata_train[train_mask].copy()
        adata_test_filtered = adata_test[test_mask].copy()

        print(f"训练数据：{adata_train.n_obs} -> {adata_train_filtered.n_obs} 个细胞")
        print(f"测试数据：{adata_test.n_obs} -> {adata_test_filtered.n_obs} 个细胞")

        self.train_dataset = GeneExpressionDatasetMappedWithGlobalCelltype(
            h5ad=adata_train_filtered,
            csv_path=self.args.csv_path,
            var_name_col=self.args.var_name_in_h5ad,
            obs_celltype_col=self.args.obs_celltype_col,
            global_type2id=self.type2id,
            global_id2type=self.id2type,
        )
        self.eval_dataset = GeneExpressionDatasetMappedWithGlobalCelltype(
            h5ad=adata_test_filtered,
            csv_path=self.args.csv_path,
            var_name_col=self.args.var_name_in_h5ad,
            obs_celltype_col=self.args.obs_celltype_col,
            global_type2id=self.type2id,
            global_id2type=self.id2type,
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
        cls_output = self.model(
            gene_ids=gene,
            value_log1p=continuous_expr,
            value_binned=discrete_expr,
        )
        loss = 0.0
        loss_cls = self.criterion_cls(cls_output, cell_type_id)
        loss = loss + loss_cls

        error_rate = 1 - (
            (cls_output.argmax(1) == cell_type_id).sum().item()
        ) / cell_type_id.size(0)
        if is_accumulating:
            with self.fabric.no_backward_sync(self.model, enabled=is_accumulating):
                self.fabric.backward(loss / self.args.grad_acc)
        else:
            self.fabric.backward(loss / self.args.grad_acc)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1e2)
            self.optimizer.step()
            if self.scheduler is not None:
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
                cls_output = model(
                    gene_ids=gene,
                    value_log1p=continuous_expr,
                    value_binned=discrete_expr,
                )
                loss = self.criterion_cls(cls_output, cell_type_id)
                total_loss += loss.item() * len(gene)
                accuracy = (cls_output.argmax(1) == cell_type_id).sum().item()
                total_error += (1 - accuracy / len(gene)) * len(gene)
                total_num += len(gene)
                preds = cls_output.argmax(1).detach().cpu()
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

            # 现在所有类型都是交集，直接计算即可
            eval_labels = np.arange(self.cell_type_count)
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(
                y_true, y_pred, average="macro", labels=eval_labels, zero_division=0
            )
            recall = recall_score(
                y_true, y_pred, average="macro", labels=eval_labels, zero_division=0
            )
            macro_f1 = f1_score(
                y_true, y_pred, average="macro", labels=eval_labels, zero_division=0
            )

        if self.is_master:
            print(
                f"Evaluation Loss: {total_loss / total_num:.4f}, "
                f"Error Rate: {total_error / total_num:.4f}, "
                f"Accuracy: {accuracy:.4f}, "
                f"Precision: {precision:.4f}, "
                f"Recall: {recall:.4f}, "
                f"Macro F1: {macro_f1:.4f}"
            )

            # 绘制评估图表
            self.plot_evaluation_charts(y_true, y_pred)

        return total_loss / total_num, total_error / total_num

    def process_evaluation_data(self, y_true, y_pred):
        """
        处理评估数据，计算指标并准备绘图所需的数据

        Returns:
            dict: 包含处理后的数据和指标，如果没有有效类别则返回None
        """
        from sklearn.metrics import classification_report

        # 现在所有类型都是交集，直接使用即可
        eval_labels = np.arange(self.cell_type_count)  # 0 到 cell_type_count-1
        target_names = [self.id2type[i] for i in eval_labels]

        # 计算分类指标
        report = classification_report(
            y_true,
            y_pred,
            labels=eval_labels,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        )

        # 提取指标数据
        metrics_data = {
            "categories": target_names,
            "recalls": [report[name]["recall"] for name in target_names],
            "precisions": [report[name]["precision"] for name in target_names],
            "f1_scores": [report[name]["f1-score"] for name in target_names],
            "supports": [report[name]["support"] for name in target_names],
        }

        # 计算类别占比
        total_samples = sum(metrics_data["supports"])
        metrics_data["proportions"] = [
            s / total_samples for s in metrics_data["supports"]
        ]
        metrics_data["unique_labels"] = eval_labels
        metrics_data["y_true"] = y_true
        metrics_data["y_pred"] = y_pred

        return metrics_data

    def plot_evaluation_charts(self, y_true, y_pred):
        """
        绘制评估图表：分类指标详情和混淆矩阵
        """
        from sklearn.metrics import confusion_matrix

        # 处理评估数据
        processed_data = self.process_evaluation_data(y_true, y_pred)
        if processed_data is None:
            print("Warning: No valid categories found for plotting")
            return

        # 解包处理后的数据
        categories = processed_data["categories"]
        recalls = processed_data["recalls"]
        precisions = processed_data["precisions"]
        f1_scores = processed_data["f1_scores"]
        supports = processed_data["supports"]
        proportions = processed_data["proportions"]
        unique_labels = processed_data["unique_labels"]
        y_true = processed_data["y_true"]
        y_pred = processed_data["y_pred"]

        # 使用新的可视化目录
        save_dir = self.vis_dir

        # 绘制图1：分类指标详情（4个子图）
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(
            f"Classification Metrics by Cell Type - Epoch {self.epoch}",
            fontsize=16,
            fontweight="bold",
        )

        # 子图1: Recall
        bars1 = ax1.bar(range(len(categories)), recalls, color="skyblue", alpha=0.8)
        ax1.set_title("Recall by Cell Type")
        ax1.set_ylabel("Recall")
        ax1.set_xticks(range(len(categories)))
        ax1.set_xticklabels(categories, rotation=45, ha="right")
        ax1.set_ylim(0, 1)
        ax1.grid(axis="y", alpha=0.3)
        # 添加数值标签
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        # 子图2: Precision
        bars2 = ax2.bar(
            range(len(categories)), precisions, color="lightcoral", alpha=0.8
        )
        ax2.set_title("Precision by Cell Type")
        ax2.set_ylabel("Precision")
        ax2.set_xticks(range(len(categories)))
        ax2.set_xticklabels(categories, rotation=45, ha="right")
        ax2.set_ylim(0, 1)
        ax2.grid(axis="y", alpha=0.3)
        # 添加数值标签
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        # 子图3: F1 Score
        bars3 = ax3.bar(
            range(len(categories)), f1_scores, color="lightgreen", alpha=0.8
        )
        ax3.set_title("F1 Score by Cell Type")
        ax3.set_ylabel("F1 Score")
        ax3.set_xticks(range(len(categories)))
        ax3.set_xticklabels(categories, rotation=45, ha="right")
        ax3.set_ylim(0, 1)
        ax3.grid(axis="y", alpha=0.3)
        # 添加数值标签
        for i, bar in enumerate(bars3):
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        # 子图4: 类别占比
        bars4 = ax4.bar(range(len(categories)), proportions, color="gold", alpha=0.8)
        ax4.set_title("Class Proportion")
        ax4.set_ylabel("Proportion")
        ax4.set_xticks(range(len(categories)))
        ax4.set_xticklabels(categories, rotation=45, ha="right")
        ax4.set_ylim(0, max(proportions) * 1.1 if proportions else 1)
        ax4.grid(axis="y", alpha=0.3)
        # 添加数值标签
        for i, bar in enumerate(bars4):
            height = bar.get_height()
            ax4.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + height * 0.05,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        plt.tight_layout()
        metrics_file = os.path.join(
            save_dir, f"classification_metrics_epoch_{self.epoch}.png"
        )
        plt.savefig(metrics_file, dpi=300, bbox_inches="tight")
        plt.close()

        # 绘制图2：混淆矩阵（只包含真实标签中的类别）
        cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm_df = pd.DataFrame(
            cm_normalized,
            index=categories[: cm.shape[0]],
            columns=categories[: cm.shape[1]],
        )

        plt.figure(figsize=(10, 10))
        sns.heatmap(
            cm_df, annot=True, fmt=".2f", cmap="Blues", cbar_kws={"shrink": 0.8}
        )
        plt.title(
            f"Confusion Matrix (Normalized) - Epoch {self.epoch}",
            fontsize=14,
            fontweight="bold",
        )
        plt.ylabel("True Label", fontsize=12)
        plt.xlabel("Predicted Label", fontsize=12)

        confusion_file = os.path.join(
            save_dir, f"confusion_matrix_epoch_{self.epoch}.png"
        )
        plt.savefig(confusion_file, dpi=300, bbox_inches="tight")
        plt.close()

        # 创建并保存指标表格
        metrics_df = pd.DataFrame(
            {
                "Cell Type": categories,
                "Recall": recalls,
                "Precision": precisions,
                "F1 Score": f1_scores,
                "Support": supports,
                "Proportion": proportions,
            }
        )
        csv_file = os.path.join(
            save_dir, f"classification_metrics_epoch_{self.epoch}.csv"
        )
        # 将数值列格式化为保留2位小数
        metrics_df["Recall"] = metrics_df["Recall"].round(2)
        metrics_df["Precision"] = metrics_df["Precision"].round(2)
        metrics_df["F1 Score"] = metrics_df["F1 Score"].round(2)
        metrics_df["Proportion"] = metrics_df["Proportion"].round(2)
        metrics_df.to_csv(csv_file, index=False)

        print("Evaluation plots saved to:")
        print(f"  - Metrics: {metrics_file}")
        print(f"  - Confusion Matrix: {confusion_file}")
        print(f"  - CSV Report: {csv_file}")

    def train(self):
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
                is_accumulating = (i + 1) % self.args.grad_acc != 0
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
            eval_loss, eval_error = self.evaluate(self.model, self.eval_loader)

            # 保存checkpoint
            self.save_checkpoint(epoch, eval_loss, eval_error)

    def save_checkpoint(self, epoch, eval_loss, eval_error):
        """
        保存checkpoint

        Args:
            epoch: 当前epoch
            eval_loss: 评估损失
            eval_error: 评估错误率
        """
        if not self.is_master:
            return

        # 准备checkpoint数据
        checkpoint = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "eval_loss": eval_loss,
            "eval_error": eval_error,
            "cell_type_count": self.cell_type_count,
            "type2id": self.type2id,
            "id2type": self.id2type,
        }

        # 保存当前epoch的checkpoint
        ckpt_path = os.path.join(self.ckpt_dir, f"epoch_{epoch}.ckpt")
        torch.save(checkpoint, ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

        # 保存为latest checkpoint (覆盖)
        latest_path = os.path.join(self.ckpt_dir, "latest_checkpoint.ckpt")
        torch.save(checkpoint, latest_path)
        print(f"Saved latest checkpoint: {latest_path}")

        # 保存训练日志
        log_file = os.path.join(self.log_dir, "training_log.txt")
        with open(log_file, "a") as f:
            f.write(f"Epoch {epoch}: Loss={eval_loss:.4f}, Error={eval_error:.4f}\n")

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
