#!/usr/bin/env python3
"""
细胞类型注释微调任务的数据加载器
从h5ad文件加载数据，进行基因映射和数据预处理
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class CellTypeDataset(Dataset):
    """
    细胞类型注释数据集
    """

    def __init__(
        self,
        h5ad_path: str,
        gene_map_path: str,
        max_length: int = 1024,
        num_bins: int = 5,
        gene_from_zero: bool = True,
        filter_low_expressed: bool = True,
        min_cells: int = 10,
        min_genes: int = 200,
    ):
        """
        Args:
            h5ad_path: h5ad文件路径
            gene_map_path: 基因映射CSV文件路径
            max_length: 最大序列长度
            num_bins: 表达值离散化的bin数量
            gene_from_zero: 是否让基因ID从0开始（为pad token留位置）
            filter_low_expressed: 是否过滤低表达基因
            min_cells: 基因至少在多少个细胞中表达才保留
            min_genes: 细胞至少表达多少个基因才保留
        """
        self.max_length = max_length
        self.num_bins = num_bins
        self.gene_from_zero = gene_from_zero

        logger.info(f"Loading data from {h5ad_path}")

        # 1. 加载h5ad数据
        self.adata = sc.read_h5ad(h5ad_path)
        logger.info(f"Original data shape: {self.adata.shape}")

        # 2. 加载基因映射表
        logger.info(f"Loading gene mapping from {gene_map_path}")
        self.gene_map = pd.read_csv(gene_map_path)
        logger.info(f"Gene mapping table shape: {self.gene_map.shape}")

        # 3. 数据质控和预处理
        if filter_low_expressed:
            self._filter_low_expressed_genes_cells(min_cells, min_genes)

        # 4. 基因映射和过滤
        self._map_genes()

        # 5. 处理细胞类型标签
        self._process_cell_types()

        # 6. 预处理表达数据
        self._preprocess_expression_data()

        logger.info(f"Final dataset: {len(self)} cells, {len(self.gene_ids)} genes")
        logger.info(f"Cell types: {self.n_classes}")

    def _filter_low_expressed_genes_cells(self, min_cells: int, min_genes: int):
        """过滤低表达基因和细胞"""
        logger.info("Filtering low expressed genes and cells...")

        # 过滤基因：在少于min_cells个细胞中表达的基因
        sc.pp.filter_genes(self.adata, min_cells=min_cells)

        # 过滤细胞：表达少于min_genes个基因的细胞
        sc.pp.filter_cells(self.adata, min_genes=min_genes)

        logger.info(f"After filtering: {self.adata.shape}")

    def _map_genes(self):
        """将h5ad中的基因标识符映射到基因ID"""
        logger.info("Mapping genes from h5ad to gene_map...")

        # 从h5ad获取基因标识符（优先级：gene_name > var_names）
        if "gene_name" in self.adata.var.columns:
            h5ad_genes = self.adata.var["gene_name"].values
            logger.info("Using 'gene_name' column from h5ad")
        else:
            # 如果没有gene_name列，使用var_names
            h5ad_genes = self.adata.var_names.values
            logger.info("Using var_names from h5ad as gene identifiers")

        logger.info(f"H5AD genes sample: {h5ad_genes[:5]}")
        logger.info(
            f"Gene map feature_name sample: {self.gene_map['feature_name'].head()}"
        )
        logger.info(f"Gene map Ensembl ID sample: {self.gene_map['Ensembl id'].head()}")

        # 首先尝试通过feature_name（基因符号）映射
        feature_to_id = dict(zip(self.gene_map["feature_name"], self.gene_map["id"]))
        valid_gene_mask_feature = np.isin(h5ad_genes, list(feature_to_id.keys()))

        # 如果feature_name映射失败，尝试Ensembl ID映射
        if valid_gene_mask_feature.sum() == 0:
            logger.info(
                "No genes found using feature_name, trying Ensembl id mapping..."
            )
            ensembl_to_id = dict(zip(self.gene_map["Ensembl id"], self.gene_map["id"]))
            valid_gene_mask = np.isin(h5ad_genes, list(ensembl_to_id.keys()))
            mapping_dict = ensembl_to_id
            mapping_type = "Ensembl id"
        else:
            logger.info("Using feature_name (gene symbol) mapping")
            valid_gene_mask = valid_gene_mask_feature
            mapping_dict = feature_to_id
            mapping_type = "feature_name"

        valid_genes = h5ad_genes[valid_gene_mask]

        logger.info(
            f"Found {len(valid_genes)} genes in mapping table out of {len(h5ad_genes)} total genes using {mapping_type}"
        )
        logger.info(f"Mapping coverage: {len(valid_genes)/len(h5ad_genes)*100:.2f}%")

        if len(valid_genes) == 0:
            logger.error("No genes found in mapping table!")
            logger.error(f"Sample h5ad genes: {h5ad_genes[:10]}")
            logger.error(
                f"Sample gene map feature_names: {list(feature_to_id.keys())[:10]}"
            )
            logger.error(
                f"Sample gene map Ensembl IDs: {list(dict(zip(self.gene_map['Ensembl id'], self.gene_map['id'])).keys())[:10]}"
            )
            raise ValueError(
                "No genes found in mapping table! Please check if h5ad gene_name matches gene_map feature_name or Ensembl id format."
            )

        # 过滤数据，只保留映射表中的基因
        self.adata = self.adata[:, valid_gene_mask].copy()

        # 获取对应的基因ID
        self.gene_identifiers = valid_genes
        self.gene_ids = np.array([mapping_dict[gene] for gene in valid_genes])

        logger.info(f"Gene IDs range: [{self.gene_ids.min()}, {self.gene_ids.max()}]")

    def _process_cell_types(self):
        """处理细胞类型标签"""
        logger.info("Processing cell type labels...")

        if "celltype" not in self.adata.obs.columns:
            raise ValueError("'celltype' column not found in adata.obs")

        # 获取细胞类型标签
        cell_types = self.adata.obs["celltype"].values

        # 编码细胞类型为数值标签
        self.label_encoder = LabelEncoder()
        self.cell_type_labels = self.label_encoder.fit_transform(cell_types)

        self.cell_type_names = self.label_encoder.classes_
        self.n_classes = len(self.cell_type_names)

        logger.info(f"Cell types: {self.cell_type_names}")
        logger.info(f"Number of classes: {self.n_classes}")

        # 统计每个类别的细胞数
        unique, counts = np.unique(self.cell_type_labels, return_counts=True)
        for i, (label, count) in enumerate(zip(unique, counts)):
            logger.info(f"  {self.cell_type_names[label]}: {count} cells")

    def _preprocess_expression_data(self):
        """预处理表达数据"""
        logger.info("Preprocessing expression data...")

        # 获取表达矩阵
        if hasattr(self.adata.X, "toarray"):
            # 如果是稀疏矩阵，转为密集矩阵
            self.expression_matrix = self.adata.X.toarray()
        else:
            self.expression_matrix = self.adata.X

        # 确保数据类型正确
        self.expression_matrix = self.expression_matrix.astype(np.float32)

        logger.info(f"Expression matrix shape: {self.expression_matrix.shape}")
        logger.info(
            f"Expression range: [{self.expression_matrix.min():.3f}, {self.expression_matrix.max():.3f}]"
        )

        # 计算一些统计信息
        mean_expr = np.mean(self.expression_matrix, axis=0)
        std_expr = np.std(self.expression_matrix, axis=0)

        logger.info(
            f"Mean expression range: [{mean_expr.min():.3f}, {mean_expr.max():.3f}]"
        )
        logger.info(
            f"Std expression range: [{std_expr.min():.3f}, {std_expr.max():.3f}]"
        )

    def __len__(self):
        return self.adata.n_obs

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本

        Returns:
            Dict containing:
                - genes: 基因ID tensor
                - expressions: 表达值tensor
                - cell_type: 细胞类型标签
        """
        # 获取基因表达值
        expression_values = self.expression_matrix[idx]

        # 为了效率，我们可以选择表达值最高的前max_length个基因
        # 或者随机采样（这里选择表达值最高的策略）
        if len(self.gene_ids) > self.max_length:
            # 按表达值降序排序，选择前max_length个
            top_indices = np.argsort(expression_values)[::-1][: self.max_length]
            selected_gene_ids = self.gene_ids[top_indices]
            selected_expressions = expression_values[top_indices]
        else:
            selected_gene_ids = self.gene_ids
            selected_expressions = expression_values

        # 转换为torch tensor
        genes = torch.tensor(selected_gene_ids, dtype=torch.long)
        expressions = torch.tensor(selected_expressions, dtype=torch.float)
        cell_type = torch.tensor(self.cell_type_labels[idx], dtype=torch.long)

        return {
            "genes": genes,
            "expressions": expressions,
            "cell_type": cell_type,
        }


class CellTypeDataCollator:
    """
    专门用于细胞类型分类的Data Collator
    不做掩码，保留所有基因信息用于分类
    """

    def __init__(
        self,
        num_bins: int = 5,
        pad_token_id: int = 0,
        pad_value: int = 0,
        max_length: int = 1024,
        gene_from_zero: bool = True,
    ):
        self.num_bins = num_bins
        self.pad_token_id = pad_token_id
        self.pad_value = pad_value
        self.max_length = max_length
        self.gene_from_zero = gene_from_zero

        # 计算特殊token的值
        # 假设基因数量从checkpoint配置中获取
        self.num_genes = 34682  # 从checkpoint配置中获取
        self.cls_token_id = self.num_genes + 1
        self.cls_value = self.num_bins + 1

    def __call__(
        self, examples: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """
        批处理数据

        Args:
            examples: 批次中的样本列表

        Returns:
            批处理后的数据字典，包含：
            - gene_ids: 基因ID
            - expression_bin: 离散化表达值
            - normalized_expr: 归一化表达值
            - cell_types: 细胞类型标签
        """
        batch_size = len(examples)

        # 收集批次数据
        batch_genes = []
        batch_expressions = []
        batch_cell_types = []

        for example in examples:
            genes = example["genes"]
            expressions = example["expressions"]
            cell_type = example["cell_type"]

            # 如果gene_from_zero为True，将基因ID加1为pad token留位置
            if self.gene_from_zero:
                genes = genes + 1

            # 添加CLS token
            genes = torch.cat(
                [torch.tensor([self.cls_token_id], dtype=genes.dtype), genes]
            )

            expressions = torch.cat(
                [torch.tensor([self.cls_value], dtype=expressions.dtype), expressions]
            )

            # 截断或填充到固定长度
            genes, expressions = self._truncate_or_pad(
                genes, expressions, self.max_length
            )

            batch_genes.append(genes)
            batch_expressions.append(expressions)
            batch_cell_types.append(cell_type)

        # 堆叠成批次
        batch_genes = torch.stack(batch_genes)
        batch_expressions = torch.stack(batch_expressions)
        batch_cell_types = torch.stack(batch_cell_types)

        # 离散化表达值
        batch_expression_bins = self._discretize_expressions(batch_expressions)

        return {
            "gene_ids": batch_genes,
            "expression_bin": batch_expression_bins,
            "normalized_expr": batch_expressions,
            "cell_types": batch_cell_types,
        }

    def _truncate_or_pad(
        self, genes: torch.Tensor, expressions: torch.Tensor, max_length: int
    ):
        """截断或填充序列"""
        current_length = len(genes)

        if current_length == max_length:
            return genes, expressions
        elif current_length > max_length:
            # 截断
            return genes[:max_length], expressions[:max_length]
        else:
            # 填充
            pad_length = max_length - current_length
            genes = torch.cat(
                [genes, torch.full((pad_length,), self.pad_token_id, dtype=genes.dtype)]
            )
            expressions = torch.cat(
                [
                    expressions,
                    torch.full((pad_length,), self.pad_value, dtype=expressions.dtype),
                ]
            )
            return genes, expressions

    def _discretize_expressions(self, expressions: torch.Tensor) -> torch.Tensor:
        """
        表达值离散化

        Args:
            expressions: 归一化表达值, shape: (batch_size, seq_len)

        Returns:
            离散化的bin索引, shape: (batch_size, seq_len)
        """
        # 为每个样本独立进行离散化
        batch_size, seq_len = expressions.shape
        discretized = torch.zeros_like(expressions, dtype=torch.long)

        for i in range(batch_size):
            expr = expressions[i]
            # 跳过填充位置
            non_pad_mask = expr != self.pad_value
            if non_pad_mask.any():
                non_pad_expr = expr[non_pad_mask]

                # 离散化非填充位置的表达值
                min_val = non_pad_expr.min()
                max_val = non_pad_expr.max()

                if max_val > min_val:
                    normalized_range = (non_pad_expr - min_val) / (
                        max_val - min_val + 1e-8
                    )
                    bin_indices = torch.floor(
                        normalized_range * (self.num_bins - 1)
                    ).long()
                    bin_indices = (
                        torch.clamp(bin_indices, 0, self.num_bins - 1) + 1
                    )  # bin从1开始
                else:
                    # 所有值相同，分配到bin 1
                    bin_indices = torch.ones_like(non_pad_expr, dtype=torch.long)

                discretized[i, non_pad_mask] = bin_indices
                # 填充位置保持为0

        return discretized


def create_data_loaders(
    h5ad_path: str,
    gene_map_path: str,
    batch_size: int = 16,
    max_length: int = 1024,
    num_bins: int = 5,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    创建训练、验证和测试数据加载器

    Args:
        h5ad_path: h5ad文件路径
        gene_map_path: 基因映射文件路径
        batch_size: 批次大小
        max_length: 最大序列长度
        num_bins: 离散化bin数量
        test_size: 测试集比例
        val_size: 验证集比例
        random_state: 随机种子
        num_workers: 数据加载工作进程数

    Returns:
        train_loader, val_loader, test_loader, num_classes
    """
    from sklearn.model_selection import train_test_split

    # 创建完整数据集
    dataset = CellTypeDataset(
        h5ad_path=h5ad_path,
        gene_map_path=gene_map_path,
        max_length=max_length,
        num_bins=num_bins,
    )

    # 分割数据集
    n_samples = len(dataset)
    indices = np.arange(n_samples)

    # 分层采样以保持类别平衡
    labels = dataset.cell_type_labels

    # 先分出测试集
    train_val_indices, test_indices = train_test_split(
        indices, test_size=test_size, stratify=labels, random_state=random_state
    )

    # 再从训练集中分出验证集
    train_labels = labels[train_val_indices]
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=val_size / (1 - test_size),
        stratify=train_labels,
        random_state=random_state,
    )

    logger.info(
        f"Dataset split: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}"
    )

    # 创建子数据集
    from torch.utils.data import Subset

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    # 创建data collator
    collator = CellTypeDataCollator(
        num_bins=num_bins,
        max_length=max_length,
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, dataset.n_classes


# 使用示例
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)

    # 示例使用
    h5ad_path = "path/to/your/data.h5ad"
    gene_map_path = "/home/angli/DeepSC/scripts/preprocessing/gene_map.csv"

    try:
        train_loader, val_loader, test_loader, num_classes = create_data_loaders(
            h5ad_path=h5ad_path,
            gene_map_path=gene_map_path,
            batch_size=16,
            max_length=1024,
            num_bins=5,
        )

        print(f"Successfully created data loaders with {num_classes} classes")

        # 测试第一个批次
        for batch in train_loader:
            print("Batch shapes:")
            for key, value in batch.items():
                print(f"  {key}: {value.shape}")
            break

    except Exception as e:
        print(f"Error: {e}")
