import os
import pickle
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import torch
import torch.nn as nn
from omegaconf import DictConfig
from sklearn.metrics import confusion_matrix
from torch import Tensor

import sys

# 添加模型路径到系统路径
model_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../models/deepsc")
)
sys.path.append(model_path)
from model import DeepSC

# 设置scanpy绘图参数
sc.set_figure_params(figsize=(6, 6))


class ClsDecoder(nn.Module):
    """
    Decoder for classification task.
    """

    def __init__(
        self,
        d_model: int,
        n_cls: int,
        nlayers: int = 3,
        activation: callable = nn.ReLU,
    ):
        super().__init__()
        # module list
        self._decoder = nn.ModuleList()
        for i in range(nlayers - 1):
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(d_model))
        self.out_layer = nn.Linear(d_model, n_cls)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, embsize]
        """
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)


# 整体模型结构：encoder + classifier
class AnnotationModel(nn.Module):
    def __init__(self, encoder, embedding_dim, num_classes):
        super().__init__()
        self.encoder = encoder
        self.decoder = ClsDecoder(d_model=embedding_dim, n_cls=num_classes)

    def forward(self, gene_ids, expression_bin, normalized_expr):
        with torch.no_grad():  # 如果 encoder 不参与训练
            # 使用DeepSC encoder的forward方法获取embeddings
            outputs = self.encoder(
                gene_ids, expression_bin, normalized_expr, return_encodings=True
            )
            # 根据DeepSC的输出格式提取最终嵌入
            if len(outputs) >= 4:
                gene_emb, expr_emb = outputs[3], outputs[4]  # gene_emb, expr_emb
                # 取平均池化或者使用CLS token的位置
                embedding = (gene_emb + expr_emb).mean(
                    dim=1
                )  # (batch_size, embedding_dim)
            else:
                raise ValueError("Unexpected encoder output format")

        logits = self.decoder(embedding)
        return logits


def calculate_per_class_ce_loss(
    logits, labels, num_classes, class_weights=None, ignore_index=-100
):
    """
    按细胞类型分别计算交叉熵损失，类似于calculate_per_bin_ce_loss

    Args:
        logits: (batch_size, num_classes) 模型输出的logits
        labels: (batch_size,) 真实标签
        num_classes: 类别总数
        class_weights: (num_classes,) 每个类别的权重，可选
        ignore_index: 要忽略的标签索引

    Returns:
        per_class_losses: (num_classes,) 每个类别的平均交叉熵损失
        weighted_loss: 加权平均损失（标量）
    """
    device = logits.device
    per_class_losses = []
    valid_losses = []

    # 计算每个类别的损失
    for class_idx in range(num_classes):
        # 找到属于当前类别的样本
        class_mask = (labels == class_idx) & (labels != ignore_index)

        if class_mask.sum() == 0:
            # 如果当前类别没有样本，损失为0
            per_class_losses.append(torch.tensor(0.0, device=device))
            continue

        # 提取当前类别的logits和labels
        class_logits = logits[class_mask]
        class_labels = labels[class_mask]

        # 计算交叉熵损失
        ce_loss_fn = nn.CrossEntropyLoss(reduction="mean")
        class_loss = ce_loss_fn(class_logits, class_labels)
        per_class_losses.append(class_loss)
        valid_losses.append(class_loss)

    # 转换为tensor
    per_class_losses = torch.stack(per_class_losses)  # (num_classes,)

    # 计算加权平均损失
    if class_weights is not None:
        # 使用提供的类别权重
        class_weights = class_weights.to(device)
        # 只对有样本的类别计算加权损失
        valid_mask = per_class_losses > 0
        if valid_mask.sum() > 0:
            weighted_loss = (
                per_class_losses[valid_mask] * class_weights[valid_mask]
            ).mean()
        else:
            weighted_loss = torch.tensor(0.0, device=device)
    else:
        # 简单平均（只计算有样本的类别）
        if len(valid_losses) > 0:
            weighted_loss = torch.stack(valid_losses).mean()
        else:
            weighted_loss = torch.tensor(0.0, device=device)

    return per_class_losses, weighted_loss


def load_deepsc_from_checkpoint(checkpoint_path, config):
    """
    从checkpoint加载DeepSC模型

    Args:
        checkpoint_path: checkpoint文件路径
        config: 模型配置

    Returns:
        loaded_model: 加载的DeepSC模型
    """
    # 创建DeepSC模型实例
    model = DeepSC(
        embedding_dim=config.model.embedding_dim,
        num_genes=config.model.num_genes,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        attn_dropout=config.model.attn_dropout,
        ffn_dropout=config.model.ffn_dropout,
        fused=config.model.fused,
        num_bins=config.model.num_bins,
        alpha=config.model.alpha,
        mask_layer_start=config.model.mask_layer_start,
        enable_l0=config.model.enable_l0,
        enable_mse=config.model.enable_mse,
        enable_ce=config.model.enable_ce,
        num_layers_ffn=config.model.num_layers_ffn,
        use_moe_regressor=config.model.use_moe_regressor,
    )

    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # 如果checkpoint包含模型状态字典
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        print(
            f"Loaded model from checkpoint epoch {checkpoint.get('epoch', 'unknown')} (strict=False)"
        )
    elif "model" in checkpoint:
        # 处理包含 'model' 键的checkpoint格式
        model.load_state_dict(checkpoint["model"], strict=False)
        print(
            f"Loaded model from checkpoint epoch {checkpoint.get('epoch', 'unknown')} (strict=False)"
        )
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"], strict=False)
        print("Loaded model from checkpoint (strict=False)")
    else:
        # 假设整个checkpoint就是state_dict
        model.load_state_dict(checkpoint, strict=False)
        print("Loaded model state dict from checkpoint (strict=False)")

    # 设置为评估模式（用于特征提取）
    model.eval()

    # 冻结encoder参数
    for param in model.parameters():
        param.requires_grad = False

    return model


def create_umap_visualization(adata, save_dir, id2type, results):
    """
    创建UMAP可视化图，显示真实标签和预测标签

    Args:
        adata: AnnData对象，包含细胞数据
        save_dir: 保存目录
        id2type: 类别ID到类别名的映射
        results: 结果字典
    """
    print("Creating UMAP visualization...")

    # 计算UMAP (如果还没有计算过的话)
    if "X_umap" not in adata.obsm:
        print("Computing UMAP embedding...")
        sc.tl.pca(adata, svd_solver="arpack")
        sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
        sc.tl.umap(adata)

    # 设置颜色palette
    palette_ = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    palette_ = palette_ + palette_ + palette_  # 扩展颜色
    celltypes = list(id2type.values())
    palette_dict = {c: palette_[i % len(palette_)] for i, c in enumerate(celltypes)}

    # 绘制UMAP图
    with plt.rc_context({"figure.figsize": (12, 5), "figure.dpi": 300}):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 真实标签
        sc.pl.umap(
            adata,
            color="celltype",
            palette=palette_dict,
            show=False,
            ax=axes[0],
            frameon=False,
            title="True Cell Types",
        )

        # 预测标签
        sc.pl.umap(
            adata,
            color="predictions",
            palette=palette_dict,
            show=False,
            ax=axes[1],
            frameon=False,
            title="Predicted Cell Types",
        )

        plt.tight_layout()
        plt.savefig(save_dir / "umap_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()

    return str(save_dir / "umap_comparison.png")


def create_confusion_matrix(labels, predictions, id2type, save_dir):
    """
    创建混淆矩阵热图

    Args:
        labels: 真实标签数组
        predictions: 预测标签数组
        id2type: 类别ID到类别名的映射
        save_dir: 保存目录
    """
    print("Creating confusion matrix...")

    # 获取类别名称
    celltypes = [id2type[i] for i in sorted(id2type.keys())]

    # 计算混淆矩阵
    cm = confusion_matrix(labels, predictions)

    # 归一化到百分比
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # 转换为DataFrame以便可视化
    cm_df = pd.DataFrame(
        cm_normalized, index=celltypes[: cm.shape[0]], columns=celltypes[: cm.shape[1]]
    )

    # 绘制热图
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_df,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        cbar_kws={"label": "Normalized Frequency"},
        square=True,
    )
    plt.title("Confusion Matrix (Normalized)")
    plt.xlabel("Predicted Cell Type")
    plt.ylabel("True Cell Type")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    return str(save_dir / "confusion_matrix.png")


def create_class_performance_plot(labels, predictions, id2type, save_dir):
    """
    创建每个类别的性能指标图
    """
    from sklearn.metrics import classification_report

    print("Creating class performance plot...")

    # 计算每个类别的详细指标
    report = classification_report(
        labels,
        predictions,
        target_names=[id2type[i] for i in sorted(id2type.keys())],
        output_dict=True,
    )

    # 提取每个类别的指标
    class_names = [id2type[i] for i in sorted(id2type.keys())]
    precision = [report[name]["precision"] for name in class_names]
    recall = [report[name]["recall"] for name in class_names]
    f1_score = [report[name]["f1-score"] for name in class_names]

    # 创建性能指标条形图
    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(15, 8))

    bars1 = ax.bar(x - width, precision, width, label="Precision", alpha=0.8)
    bars2 = ax.bar(x, recall, width, label="Recall", alpha=0.8)
    bars3 = ax.bar(x + width, f1_score, width, label="F1-Score", alpha=0.8)

    ax.set_xlabel("Cell Types")
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Performance Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 添加数值标签
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)

    plt.tight_layout()
    plt.savefig(save_dir / "class_performance.png", dpi=300, bbox_inches="tight")
    plt.close()

    return str(save_dir / "class_performance.png")


def comprehensive_test_with_visualization(
    model, test_loader, device, num_classes, class_weights, cfg
):
    """
    comprehensive测试函数，包含详细的可视化和指标计算

    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        device: 设备
        num_classes: 类别数
        class_weights: 类别权重
        cfg: 配置

    Returns:
        detailed_results: 详细的测试结果
    """
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        f1_score,
        precision_score,
        recall_score,
    )

    print("Starting comprehensive testing with visualization...")

    # 创建保存目录
    save_dir = Path(cfg.output_dir) / "test_results"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 测试模型
    test_loss, test_acc, test_metrics, test_per_class_losses = evaluate(
        model, test_loader, device, num_classes, class_weights
    )

    # 获取所有预测结果
    model.eval()
    all_predictions = []
    all_labels = []
    all_gene_ids = []
    all_expressions = []

    with torch.no_grad():
        for batch in test_loader:
            gene_ids = batch["gene_ids"].to(device)
            expression_bin = batch["expression_bin"].to(device)
            normalized_expr = batch["normalized_expr"].to(device)
            cell_types = batch["cell_types"].to(device)

            logits = model(gene_ids, expression_bin, normalized_expr)
            _, predicted = torch.max(logits.data, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(cell_types.cpu().numpy())
            all_gene_ids.append(gene_ids.cpu().numpy())
            all_expressions.append(normalized_expr.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # 创建类别ID到类别名的映射
    id2type = {}
    # 从数据加载器中获取类别映射信息
    if hasattr(test_loader.dataset, "label_encoder"):
        label_encoder = test_loader.dataset.label_encoder
        for i, class_name in enumerate(label_encoder.classes_):
            id2type[i] = class_name
    else:
        # 如果没有label_encoder，创建默认映射
        for i in range(num_classes):
            id2type[i] = f"CellType_{i}"

    print(f"ID to type mapping: {id2type}")

    # 计算详细指标
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(
        all_labels, all_predictions, average="macro", zero_division=0
    )
    recall = recall_score(all_labels, all_predictions, average="macro", zero_division=0)
    macro_f1 = f1_score(all_labels, all_predictions, average="macro", zero_division=0)

    # 生成分类报告
    class_names = [id2type[i] for i in sorted(id2type.keys())]
    classification_rep = classification_report(
        all_labels, all_predictions, target_names=class_names, zero_division=0
    )

    print(f"\n{'='*50}")
    print("DETAILED TEST RESULTS")
    print(f"{'='*50}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision (macro): {precision:.4f}")
    print(f"Test Recall (macro): {recall:.4f}")
    print(f"Test F1-Score (macro): {macro_f1:.4f}")
    print(f"\n{classification_rep}")

    # 保存分类报告到文件
    with open(save_dir / "classification_report.txt", "w") as f:
        f.write("DETAILED TEST RESULTS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {accuracy:.4f}\n")
        f.write(f"Test Precision (macro): {precision:.4f}\n")
        f.write(f"Test Recall (macro): {recall:.4f}\n")
        f.write(f"Test F1-Score (macro): {macro_f1:.4f}\n\n")
        f.write(classification_rep)

    # 创建模拟的AnnData对象用于可视化
    print("Creating AnnData object for visualization...")

    # 合并所有表达数据
    all_expressions_combined = np.vstack(all_expressions)

    # 创建AnnData对象
    import anndata

    adata_test = anndata.AnnData(X=all_expressions_combined)

    # 添加观测数据
    adata_test.obs["celltype"] = [id2type[label] for label in all_labels]
    adata_test.obs["predictions"] = [id2type[pred] for pred in all_predictions]

    # 创建可视化
    print("Creating visualizations...")

    # 1. UMAP可视化
    umap_path = create_umap_visualization(adata_test, save_dir, id2type, {})

    # 2. 混淆矩阵
    cm_path = create_confusion_matrix(all_labels, all_predictions, id2type, save_dir)

    # 3. 类别性能图
    class_perf_path = create_class_performance_plot(
        all_labels, all_predictions, id2type, save_dir
    )

    # 保存详细结果
    detailed_results = {
        "test_loss": test_loss,
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1_macro": macro_f1,
        "per_class_losses": test_per_class_losses.cpu().numpy(),
        "predictions": all_predictions,
        "labels": all_labels,
        "id2type": id2type,
        "classification_report": classification_rep,
        "visualization_paths": {
            "umap": umap_path,
            "confusion_matrix": cm_path,
            "class_performance": class_perf_path,
        },
    }

    # 保存结果到pickle文件
    with open(save_dir / "detailed_results.pkl", "wb") as f:
        pickle.dump(detailed_results, f)

    print(f"All results and visualizations saved to: {save_dir}")

    return detailed_results


@hydra.main(
    version_base=None, config_path="../../../configs/finetune", config_name="finetune"
)
def finetune(cfg: DictConfig):
    """
    微调函数

    Args:
        cfg: 配置文件
    """
    from finetune_data_loader import create_data_loaders

    import time
    import wandb

    print(f"Starting cell type annotation fine-tuning with config: {cfg}")

    # 设置设备
    gpu_id = cfg.get("gpu_id", 0)  # 从配置中获取GPU ID，默认为0
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        print(f"Using device: {device} (GPU {gpu_id})")
        print(f"GPU {gpu_id} name: {torch.cuda.get_device_name(gpu_id)}")
        print(
            f"GPU {gpu_id} memory: {torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3:.1f} GB"
        )
    else:
        device = torch.device("cpu")
        print(f"Using device: {device}")

    # 设置随机种子
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.set_device(gpu_id)  # 设置当前GPU

    # 创建数据加载器
    print("Creating data loaders...")
    train_loader, val_loader, test_loader, num_classes = create_data_loaders(
        h5ad_path=cfg.data_path,
        gene_map_path="/home/angli/DeepSC/scripts/preprocessing/gene_map.csv",
        batch_size=cfg.batch_size,
        max_length=cfg.sequence_length,
        num_bins=cfg.model.num_bins,
        test_size=0.2,
        val_size=0.1,
        random_state=cfg.seed,
        num_workers=4,
    )
    print(f"Data loaders created successfully with {num_classes} classes")

    # 计算类别权重（处理不平衡）
    class_counts = []
    total_samples = 0
    for batch in train_loader:
        labels = batch["cell_types"]
        for class_idx in range(num_classes):
            count = (labels == class_idx).sum().item()
            if class_idx < len(class_counts):
                class_counts[class_idx] += count
            else:
                class_counts.append(count)
        total_samples += len(labels)

    # 计算每个类别的权重（反比于样本数）
    class_weights = torch.tensor(
        [
            total_samples / (num_classes * count) if count > 0 else 0.0
            for count in class_counts
        ],
        dtype=torch.float32,
    )
    print(f"Class weights calculated: {class_weights}")

    # 更新配置中的类别数
    cfg.num_classes = num_classes

    # 加载预训练的DeepSC模型
    print(f"Loading DeepSC model from checkpoint: {cfg.checkpoint_path}")
    encoder = load_deepsc_from_checkpoint(cfg.checkpoint_path, cfg)
    encoder = encoder.to(device)

    # 创建注释模型
    annotation_model = AnnotationModel(
        encoder=encoder, embedding_dim=cfg.model.embedding_dim, num_classes=num_classes
    )
    annotation_model = annotation_model.to(device)
    class_weights = class_weights.to(device)

    print(f"Created annotation model with {num_classes} classes")
    print(
        f"Model has {sum(p.numel() for p in annotation_model.parameters() if p.requires_grad)} trainable parameters"
    )

    # 创建优化器
    optimizer = torch.optim.AdamW(
        annotation_model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.get("weight_decay", 1e-5),
    )

    # 创建学习率调度器
    num_training_steps = len(train_loader) * cfg.epoch
    warmup_steps = int(num_training_steps * cfg.get("warmup_ratio", 0.1))

    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=num_training_steps - warmup_steps
    )
    scheduler = SequentialLR(
        optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_steps]
    )

    # 初始化wandb（如果配置了）
    if cfg.get("wandb_project"):
        wandb.init(
            project=cfg.wandb_project,
            name=cfg.get("run_name", f'finetune_{time.strftime("%Y%m%d_%H%M%S")}'),
            tags=cfg.get("tags", []),
            config=dict(cfg),
        )

    # 训练循环
    print("Starting training...")
    best_val_acc = 0.0
    patience = cfg.get("patience", 5)
    patience_counter = 0

    for epoch in range(cfg.epoch):
        # 训练阶段
        train_loss, train_acc, train_per_class_losses = train_epoch(
            annotation_model,
            train_loader,
            optimizer,
            device,
            scheduler,
            num_classes,
            class_weights,
        )

        # 验证阶段
        val_loss, val_acc, val_metrics, val_per_class_losses = evaluate(
            annotation_model, val_loader, device, num_classes, class_weights
        )

        print(f"Epoch {epoch+1}/{cfg.epoch}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"  Val F1: {val_metrics['f1']:.4f}")
        print(f"Val Precision: {val_metrics['precision']:.4f}")
        print(f" Val Recall: {val_metrics['recall']:.4f}")

        # 打印每个类别的损失（前5个类别）
        print(f"  Train per-class losses (top 5): {train_per_class_losses[:5]}")
        print(f"  Val per-class losses (top 5): {val_per_class_losses[:5]}")

        # 记录到wandb
        if cfg.get("wandb_project"):
            log_dict = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_metrics["f1"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "learning_rate": scheduler.get_last_lr()[0],
            }

            # 记录每个类别的损失
            for i, (train_class_loss, val_class_loss) in enumerate(
                zip(train_per_class_losses, val_per_class_losses)
            ):
                log_dict[f"train_class_{i}_loss"] = train_class_loss.item()
                log_dict[f"val_class_{i}_loss"] = val_class_loss.item()

            wandb.log(log_dict)

        # 早停检查
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            # 保存最佳模型
            torch.save(
                {
                    "model_state_dict": annotation_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": epoch + 1,
                    "best_val_acc": best_val_acc,
                    "config": cfg,
                },
                f"{cfg.ckpt_dir}/best_model.pth",
            )
            print(f"  New best model saved with val_acc: {best_val_acc:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(
                f"Early stopping triggered after {patience} epochs without improvement"
            )
            break

    # 最终测试 - 使用comprehensive测试函数
    print("Evaluating on test set with comprehensive visualization...")

    # 加载最佳模型
    best_checkpoint = torch.load(f"{cfg.ckpt_dir}/best_model.pth", weights_only=False)
    annotation_model.load_state_dict(best_checkpoint["model_state_dict"])

    # 执行comprehensive测试
    detailed_results = comprehensive_test_with_visualization(
        annotation_model, test_loader, device, num_classes, class_weights, cfg
    )

    # 记录最终结果到wandb
    if cfg.get("wandb_project"):
        final_log_dict = {
            "test_loss": detailed_results["test_loss"],
            "test_acc": detailed_results["test_accuracy"],
            "test_f1": detailed_results["test_f1_macro"],
            "test_precision": detailed_results["test_precision"],
            "test_recall": detailed_results["test_recall"],
        }

        # 记录测试集每个类别的损失
        for i, test_class_loss in enumerate(detailed_results["per_class_losses"]):
            final_log_dict[f"test_class_{i}_loss"] = test_class_loss

        # 上传可视化图片到wandb
        try:
            final_log_dict["test/umap_comparison"] = wandb.Image(
                detailed_results["visualization_paths"]["umap"],
                caption=f"UMAP comparison - Test F1: {detailed_results['test_f1_macro']:.3f}",
            )
            final_log_dict["test/confusion_matrix"] = wandb.Image(
                detailed_results["visualization_paths"]["confusion_matrix"],
                caption="Confusion Matrix",
            )
            final_log_dict["test/class_performance"] = wandb.Image(
                detailed_results["visualization_paths"]["class_performance"],
                caption="Per-class Performance Metrics",
            )
        except Exception as e:
            print(f"Warning: Could not upload images to wandb: {e}")

        wandb.log(final_log_dict)
        wandb.finish()

    print(
        f"Training completed! Final test accuracy: {detailed_results['test_accuracy']:.4f}"
    )
    print(f"Results saved to: {cfg.output_dir}/test_results")

    return annotation_model, detailed_results


def train_epoch(
    model, train_loader, optimizer, device, scheduler, num_classes, class_weights
):
    """训练一个epoch，使用per-class损失"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    all_per_class_losses = torch.zeros(num_classes, device=device)
    num_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        # 将数据移到设备上
        gene_ids = batch["gene_ids"].to(device)
        expression_bin = batch["expression_bin"].to(device)
        normalized_expr = batch["normalized_expr"].to(device)
        cell_types = batch["cell_types"].to(device)

        # 前向传播
        optimizer.zero_grad()
        logits = model(gene_ids, expression_bin, normalized_expr)

        # 计算per-class损失
        per_class_losses, weighted_loss = calculate_per_class_ce_loss(
            logits, cell_types, num_classes, class_weights
        )

        # 反向传播
        weighted_loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step()

        # 统计
        total_loss += weighted_loss.item()
        all_per_class_losses += per_class_losses
        num_batches += 1

        _, predicted = torch.max(logits.data, 1)
        total += cell_types.size(0)
        correct += (predicted == cell_types).sum().item()

        if batch_idx % 50 == 0:
            print(
                f"  Batch {batch_idx}/{len(train_loader)}, Loss: {weighted_loss.item():.4f}, Acc: {100*correct/total:.2f}%"
            )

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    avg_per_class_losses = all_per_class_losses / num_batches

    return avg_loss, accuracy, avg_per_class_losses


def evaluate(model, data_loader, device, num_classes, class_weights):
    """评估模型，使用per-class损失"""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    all_per_class_losses = torch.zeros(num_classes, device=device)
    num_batches = 0

    with torch.no_grad():
        for batch in data_loader:
            # 将数据移到设备上
            gene_ids = batch["gene_ids"].to(device)
            expression_bin = batch["expression_bin"].to(device)
            normalized_expr = batch["normalized_expr"].to(device)
            cell_types = batch["cell_types"].to(device)

            # 前向传播
            logits = model(gene_ids, expression_bin, normalized_expr)

            # 计算per-class损失
            per_class_losses, weighted_loss = calculate_per_class_ce_loss(
                logits, cell_types, num_classes, class_weights
            )

            total_loss += weighted_loss.item()
            all_per_class_losses += per_class_losses
            num_batches += 1

            # 收集预测和标签
            _, predicted = torch.max(logits.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(cell_types.cpu().numpy())

    # 计算指标
    avg_loss = total_loss / len(data_loader)
    avg_per_class_losses = all_per_class_losses / num_batches
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average="macro")
    precision = precision_score(all_labels, all_predictions, average="macro")
    recall = recall_score(all_labels, all_predictions, average="macro")

    metrics = {
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }

    return avg_loss, accuracy, metrics, avg_per_class_losses


if __name__ == "__main__":
    finetune()
