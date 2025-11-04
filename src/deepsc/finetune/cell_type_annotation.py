import os

import numpy as np
import scanpy as sc
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from deepsc.data import DataCollator
from deepsc.data.dataset import (
    GeneExpressionDatasetMappedWithGlobalCelltype,
)
from deepsc.utils import (
    calculate_mean_ce_loss,
    create_scheduler_from_args,
    extract_state_dict_with_encoder_prefix,
    get_trainable_parameters,
    plot_classification_metrics,
    report_loading_result,
    sample_weight_norms,
    seed_all,
    setup_finetune_mode,
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

        # Configure fine-tuning mode
        finetune_mode = getattr(self.args, "finetune_mode", "full")
        setup_finetune_mode(self.model, finetune_mode, self.is_master)

        self.optimizer = Adam(
            get_trainable_parameters(self.model, finetune_mode, self.is_master),
            lr=self.args.learning_rate,
        )
        self.model, self.optimizer = self.fabric.setup(self.model, self.optimizer)
        self.init_loss_fn()
        self.scheduler = create_scheduler_from_args(
            self.optimizer, self.args, self.world_size
        )

    def init_loss_fn(self):
        """Initialize loss function based on configuration."""
        if self.args.cls_loss_type == "standard":
            self.criterion_cls = nn.CrossEntropyLoss()
        elif self.args.cls_loss_type == "per_bin":
            # Use per-class CE loss from utils
            def per_class_loss_wrapper(logits, labels):
                return calculate_mean_ce_loss(
                    logits, labels, self.cell_type_count, ignore_index=-100
                )

            self.criterion_cls = per_class_loss_wrapper

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
        self.eval_sampler = DistributedSampler(self.eval_dataset, shuffle=False)
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
        logits, _, y, gene_emb, expr_emb, cls_output = model_outputs
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
            save_dir = (
                self.args.save_dir
                if hasattr(self.args, "save_dir") and self.args.save_dir
                else "./evaluation_plots"
            )
            plot_classification_metrics(
                y_true, y_pred, self.id2type, save_dir=save_dir, epoch=self.epoch
            )

        return total_loss / total_num, total_error / total_num

    def train(self):
        print("train_loader: ", len(self.train_loader.dataset))
        print("eval_loader: ", len(self.eval_loader.dataset))
        best_loss = float("inf")
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
            eval_loss, eval_error = self.evaluate(self.model, self.eval_loader)

            # Save best model
            if self.is_master and eval_loss < best_loss:
                best_loss = eval_loss
                self.save_checkpoint(epoch, eval_loss, is_best=True)
                print(f"Saved best model at epoch {epoch} with loss {eval_loss:.4f}")

            # Save checkpoint every N epochs if configured
            if self.is_master and hasattr(self.args, "save_every_n_epochs"):
                if epoch % self.args.save_every_n_epochs == 0:
                    self.save_checkpoint(epoch, eval_loss, is_best=False)

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

    def save_checkpoint(self, epoch, eval_loss, is_best=False):
        """Save model checkpoint."""
        save_dir = (
            self.args.save_dir
            if hasattr(self.args, "save_dir") and self.args.save_dir
            else "./results/cell_type_annotation"
        )
        os.makedirs(save_dir, exist_ok=True)

        # Prepare checkpoint data
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "eval_loss": eval_loss,
            "cell_type_count": self.cell_type_count,
            "type2id": self.type2id,
            "id2type": self.id2type,
        }

        # Save checkpoint
        if is_best:
            checkpoint_path = os.path.join(save_dir, "best_model.pt")
        else:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt")

        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
