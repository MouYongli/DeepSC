# Zheng68K 细胞类型注释训练指南

## 数据集信息

- **位置**: `/home/angli/baseline/DeepSC/data/cell_type_annotation/Zheng68K/`
- **文件**:
  - `zheng_train.h5ad` - 训练集 (52,754 cells)
  - `zheng_valid.h5ad` - 验证集 (6,594 cells)  
  - `zheng_test.h5ad` - 测试集 (6,595 cells)
  - `zheng_merged.h5ad` - 合并数据集 (65,943 cells) ✓ 已生成
- **基因数**: 19,264 (已对齐到scFoundation词汇表)
- **细胞类型**: 11类
  - 0: CD14+ Monocyte
  - 1: CD19+ B
  - 2: CD34+
  - 3: CD4+ T Helper2
  - 4: CD4+/CD25 T Reg
  - 5: CD4+/CD45RA+/CD25- Naive T
  - 6: CD4+/CD45RO+ Memory
  - 7: CD56+ NK
  - 8: CD8+ Cytotoxic T
  - 9: CD8+/CD45RA+ Naive Cytotoxic
  - 10: Dendritic

## 环境准备

### 1. 激活conda环境
```bash
source /home/angli/anaconda3/etc/profile.d/conda.sh
conda activate scfoundation
```

### 2. 进入工作目录
```bash
cd /home/angli/scFoundation/model
```

## 启动训练

### 基础训练命令

```bash
python finetune_celltype.py \
    --data /home/angli/baseline/DeepSC/data/cell_type_annotation/Zheng68K/zheng_merged.h5ad \
    --label_key cell_type_label \
    --split_key split \
    --checkpoint ./models/models.ckpt \
    --epochs 30 \
    --batch_size 32 \
    --lr 1e-3 \
    --output_dir ./checkpoints_zheng68k
```

### 后台运行（推荐）

```bash
nohup python finetune_celltype.py \
    --data /home/angli/baseline/DeepSC/data/cell_type_annotation/Zheng68K/zheng_merged.h5ad \
    --label_key cell_type_label \
    --split_key split \
    --checkpoint ./models/models.ckpt \
    --epochs 30 \
    --batch_size 32 \
    --lr 1e-3 \
    --output_dir ./checkpoints_zheng68k \
    > training_zheng68k.log 2>&1 &
```

查看训练日志：
```bash
tail -f training_zheng68k.log
```

## 参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `--data` | zheng_merged.h5ad | 合并后的数据集 |
| `--label_key` | cell_type_label | 标签列名 |
| `--split_key` | split | 数据集划分列（train/val/test） |
| `--checkpoint` | ./models/models.ckpt | 预训练模型路径 |
| `--epochs` | 30 | 训练轮数 |
| `--batch_size` | 32 | 批次大小 |
| `--lr` | 1e-3 | 学习率 |
| `--output_dir` | ./checkpoints_zheng68k | 输出目录 |

### 可选参数调整

```bash
# 大数据集配置（更快收敛）
python finetune_celltype.py \
    --data /home/angli/baseline/DeepSC/data/cell_type_annotation/Zheng68K/zheng_merged.h5ad \
    --label_key cell_type_label \
    --split_key split \
    --checkpoint ./models/models.ckpt \
    --epochs 20 \
    --batch_size 64 \
    --lr 1e-3 \
    --finetune_layers 3 \
    --dropout 0.05 \
    --output_dir ./checkpoints_zheng68k

# 小内存GPU配置
python finetune_celltype.py \
    --data /home/angli/baseline/DeepSC/data/cell_type_annotation/Zheng68K/zheng_merged.h5ad \
    --label_key cell_type_label \
    --split_key split \
    --checkpoint ./models/models.ckpt \
    --epochs 30 \
    --batch_size 16 \
    --lr 5e-4 \
    --finetune_layers 1 \
    --output_dir ./checkpoints_zheng68k
```

## 训练输出

训练完成后，会在输出目录生成以下文件：

```
checkpoints_zheng68k/celltype_finetune_YYYYMMDD_HHMMSS/
├── args.json              # 训练参数
├── label_mapping.json     # 标签映射
├── best_model.pt          # 最佳模型
├── final_model.pt         # 最终模型
├── test_results.json      # 测试集结果
├── history.json           # 训练历史
└── confusion_matrix.npy   # 混淆矩阵
```

## 监控训练

### 方法1：实时查看日志
```bash
tail -f training_zheng68k.log
```

### 方法2：查看进程
```bash
ps aux | grep finetune_celltype.py
```

### 方法3：查看GPU使用
```bash
watch -n 1 nvidia-smi
```

### 方法4：检查输出文件
```bash
ls -lh checkpoints_zheng68k/celltype_finetune_*/
```

## 停止训练

```bash
# 查找进程ID
ps aux | grep finetune_celltype.py

# 终止进程（替换<PID>为实际进程号）
kill <PID>

# 或强制终止
kill -9 <PID>
```

## 预期结果

根据文档和数据集情况，预期性能：
- **准确率 (Accuracy)**: > 90%
- **F1分数 (Macro)**: > 88%
- **训练时间**: 约30-60分钟/epoch（取决于GPU）

## 故障排查

### 问题1：CUDA out of memory
**解决方案**: 减少batch_size
```bash
--batch_size 16  # 或更小
```

### 问题2：训练太慢
**解决方案**: 
- 增加batch_size（如果GPU内存足够）
- 减少finetune_layers
- 增加num_workers

### 问题3：验证集准确率不提升
**解决方案**:
- 调整学习率（减小lr）
- 增加dropout防止过拟合
- 检查数据质量

## 使用训练好的模型

训练完成后，可以用于预测新数据：

```python
import torch
import scanpy as sc
from finetune_celltype import CellTypeClassifier

# 1. 加载模型
checkpoint = torch.load('checkpoints_zheng68k/celltype_finetune_XXXXXX/best_model.pt')
model = CellTypeClassifier(
    ckpt_path='./models/models.ckpt',
    num_classes=11,
    finetune_layers=2
)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.cuda()
model.eval()

# 2. 加载新数据并预测
# （参考 FINETUNE_README.md 中的推理部分）
```

## 相关文档

- 详细微调教程: `/home/angli/scFoundation/model/FINETUNE_README.md`
- 数据集脚本: `/home/angli/scFoundation/model/celltype_dataset.py`
- 训练脚本: `/home/angli/scFoundation/model/finetune_celltype.py`

## 注意事项

1. ✅ 数据集已经预处理并对齐到scFoundation的19,264基因词汇表
2. ✅ 训练/验证/测试集已经划分好（train/val/test split）
3. ✅ 基因匹配率: 100%（完美对齐）
4. ⚠️ 确保有足够的GPU内存（推荐16GB+）
5. ⚠️ 训练时间较长，建议使用后台运行模式

---

**创建时间**: 2025-11-06  
**最后更新**: 2025-11-06
