# Quick Start Guide - Perturbation Prediction

## 快速开始

### 方法1: 使用Bash脚本 (推荐)

```bash
cd /home/angli/DeepSC/src/deepsc/finetune
./run_finetune_perturbation.sh
```

### 方法2: 手动运行

```bash
cd /home/angli/DeepSC
conda activate deepsc
export PYTHONPATH="/home/angli/DeepSC/src:$PYTHONPATH"
python src/deepsc/finetune/finetune_perturbation.py
```

### 方法3: 在Python中使用

```python
import sys
sys.path.insert(0, '/home/angli/DeepSC/src')

from deepsc.models.scgpt_pert import (
    TransformerGenerator,
    masked_mse_loss,
    GeneVocab,
    set_seed,
)

# 创建词汇表
genes = ["GENE1", "GENE2", "GENE3"]
vocab = GeneVocab(genes, specials=["<pad>", "<cls>"])

# 创建模型
model = TransformerGenerator(
    ntoken=len(vocab),
    d_model=512,
    nhead=8,
    d_hid=512,
    nlayers=12,
    nlayers_cls=3,
    n_cls=1,
    vocab=vocab,
)
```

## 文件位置

- **模型**: `/home/angli/DeepSC/src/deepsc/models/scgpt_pert/`
- **训练脚本**: `/home/angli/DeepSC/src/deepsc/finetune/finetune_perturbation.py`
- **Bash启动**: `/home/angli/DeepSC/src/deepsc/finetune/run_finetune_perturbation.sh`
- **文档**: `/home/angli/DeepSC/src/deepsc/finetune/README_finetune_pert.md`

## 常用配置

在 `finetune_perturbation.py` 中修改:

```python
data_name = "norman"     # 数据集: "norman" 或 "adamson"
batch_size = 64          # 批大小
epochs = 10              # 训练轮数
lr = 1e-4               # 学习率
load_model = "../save/scGPT_human"  # 预训练模型路径 (或 None)
```

## 验证安装

```bash
conda activate deepsc
cd /home/angli/DeepSC
python -c "
import sys
sys.path.insert(0, 'src')
from deepsc.models.scgpt_pert import *
print('✓ Installation successful!')
"
```

## 输出位置

结果保存在: `./save/dev_perturb_{dataset}-{timestamp}/`

包含:
- `best_model.pt` - 模型权重
- `run.log` - 训练日志
- `test_metrics.json` - 测试指标

## 常见问题

### Q: 导入错误?
A: 确保 PYTHONPATH 包含 `/home/angli/DeepSC/src`

### Q: CUDA内存不足?
A: 减小 `batch_size` 或 `max_seq_len`

### Q: 找不到数据?
A: 数据会自动下载到 `./data` 目录

## 更多信息

查看完整文档:
- `README_finetune_pert.md` - 详细使用说明
- `MIGRATION_SUMMARY.md` - 迁移总结
