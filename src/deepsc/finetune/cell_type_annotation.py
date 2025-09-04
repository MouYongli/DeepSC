import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import torch
import torch.nn as nn

# from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.optim.lr_scheduler import (
    LinearLR,
    SequentialLR,
)
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from deepsc.data.dataset import (  # GeneExpressionDatasetMapped,
    GeneExpressionDatasetMappedWithGlobalCelltype,
    create_global_celltype_mapping,
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

        # 首先构建数据集以确定正确的celltype数量
        self.build_dataset_sampler_from_h5ad()

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

        # 保存仅训练集的细胞类型信息（用于公平评估）
        print("Collecting training cell types...")
        train_celltypes = set(
            adata_train.obs[self.args.obs_celltype_col].astype(str).unique()
        )
        print(f"训练集细胞类型: {sorted(train_celltypes)}")

        # 创建统一的celltype映射表
        print("Creating global celltype mapping...")
        self.type2id, self.id2type = create_global_celltype_mapping(
            adata_train, adata_test, obs_celltype_col=self.args.obs_celltype_col
        )

        # 保存仅训练集的类型ID集合，用于评估时的公平性检查
        self.train_only_label_ids = set()
        for celltype_name in train_celltypes:
            if celltype_name in self.type2id:
                self.train_only_label_ids.add(self.type2id[celltype_name])
        print(f"训练集类型对应的ID: {sorted(self.train_only_label_ids)}")
        print(f"训练集类型ID数量: {len(self.train_only_label_ids)}")

        # 同时保存测试集类型ID，用于调试
        test_celltypes = set(
            adata_test.obs[self.args.obs_celltype_col].astype(str).unique()
        )
        self.test_only_label_ids = set()
        for celltype_name in test_celltypes:
            if celltype_name in self.type2id:
                self.test_only_label_ids.add(self.type2id[celltype_name])
        print(f"测试集类型对应的ID: {sorted(self.test_only_label_ids)}")
        print(f"测试集类型ID数量: {len(self.test_only_label_ids)}")

        # 计算共同类型ID
        common_type_ids = self.train_only_label_ids & self.test_only_label_ids
        print(f"理论上的共同类型ID: {sorted(common_type_ids)}")
        print(f"理论上的共同类型ID数量: {len(common_type_ids)}")

        # 保存celltype数量信息
        self.cell_type_count = len(self.type2id)
        print(f"Confirmed cell_type_count: {self.cell_type_count}")

        self.train_dataset = GeneExpressionDatasetMappedWithGlobalCelltype(
            h5ad=adata_train,
            csv_path=self.args.csv_path,
            var_name_col=self.args.var_name_in_h5ad,
            obs_celltype_col=self.args.obs_celltype_col,
            global_type2id=self.type2id,
            global_id2type=self.id2type,
        )
        self.eval_dataset = GeneExpressionDatasetMappedWithGlobalCelltype(
            h5ad=adata_test,
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

            # 使用训练集类型与测试真实标签的交集，确保公平评估
            unique_true_early = np.unique(y_true)

            # 获取仅训练集的细胞类型
            if hasattr(self, "train_only_label_ids"):
                train_label_ids_early = self.train_only_label_ids
            else:
                # 备选方案：假设所有类型都在训练集中（向后兼容）
                train_label_ids_early = set(range(self.cell_type_count))

            # 只评估训练时见过且测试中存在的类别
            meaningful_labels_early = sorted(
                train_label_ids_early & set(unique_true_early)
            )
            eval_labels = np.array(meaningful_labels_early)

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

        # 确定要评估的类别：训练集和测试集都有的类型
        if hasattr(self, "train_only_label_ids") and hasattr(
            self, "test_only_label_ids"
        ):
            # 使用训练集和测试集的交集
            eval_labels = sorted(self.train_only_label_ids & self.test_only_label_ids)
        else:
            # 后备方案：使用测试集中出现的所有类别
            eval_labels = sorted(np.unique(y_true))

        if not eval_labels:
            print("Warning: No valid evaluation labels found")
            return None

        eval_labels = np.array(eval_labels)

        # 获取类别名称映射
        id2type = getattr(self, "id2type", {i: f"Type_{i}" for i in eval_labels})
        target_names = [id2type.get(i, f"Type_{i}") for i in eval_labels]

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
            "categories": [],
            "recalls": [],
            "precisions": [],
            "f1_scores": [],
            "supports": [],
        }

        for label in target_names:
            if label in report and isinstance(report[label], dict):
                metrics_data["categories"].append(label)
                metrics_data["recalls"].append(report[label]["recall"])
                metrics_data["precisions"].append(report[label]["precision"])
                metrics_data["f1_scores"].append(report[label]["f1-score"])
                metrics_data["supports"].append(report[label]["support"])

        if not metrics_data["categories"]:
            return None

        # 计算类别占比
        total_samples = sum(metrics_data["supports"])
        metrics_data["proportions"] = [
            s / total_samples for s in metrics_data["supports"]
        ]
        metrics_data["unique_labels"] = eval_labels
        metrics_data["y_true"] = y_true
        metrics_data["y_pred"] = y_pred

        print(f"📊 评估了 {len(metrics_data['categories'])} 个细胞类型")
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

        # 创建保存图表的目录
        save_dir = (
            self.args.save_dir
            if hasattr(self.args, "save_dir") and self.args.save_dir
            else "./evaluation_plots"
        )
        os.makedirs(save_dir, exist_ok=True)

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
