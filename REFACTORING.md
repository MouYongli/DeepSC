# DeepSC 项目重构方案

## 概述

本文档基于对 DeepSC 项目结构的深入分析，提出了系统性的重构方案，旨在提高代码质量、可维护性和扩展性。

## 当前项目分析

### 项目优势
- ✅ 模块化设计良好，功能分离清晰
- ✅ 使用 Hydra 进行配置管理，支持灵活的参数配置
- ✅ 完整的数据处理流水线（下载、预处理、训练、评估）
- ✅ 支持多种先进模型（DeepSC、scBERT、scGPT等）
- ✅ 文档相对完善，每个模块都有说明

### 存在问题
- ❌ **代码重复**：`models/deepsc/` 和 `models/deepsc_new/` 存在冗余
- ❌ **目录混乱**：根目录文件过多，调试脚本分散
- ❌ **依赖冲突**：`pyproject.toml` 与 `requirements.txt` 版本不一致
- ❌ **接口不统一**：模型间缺乏统一的基类和接口规范
- ❌ **缺少测试**：测试覆盖率低，只有基础测试文件

## 重构目标

### 主要目标
1. **提高代码复用性**：提取公共组件，减少重复代码
2. **优化项目结构**：清晰的目录结构，职责分离
3. **统一接口规范**：标准化的模型和数据处理接口
4. **完善依赖管理**：解决版本冲突，规范化依赖
5. **增强可维护性**：更好的文档、测试和代码组织

### 性能目标
- 减少 30% 的代码重复
- 提升项目结构清晰度
- 实现 80% 以上的测试覆盖率
- 统一所有模块的接口规范

## 新目录结构设计

```
DeepSC/
├── configs/                    # 配置文件目录（保持现有结构）
│   ├── finetune/              # 微调配置
│   └── pretrain/              # 预训练配置
├── data/                      # 数据目录（保持现有）
├── docs/                      # 📁 新增：文档目录
│   ├── api/                   # API 文档
│   ├── tutorials/             # 教程文档
│   ├── architecture.md        # 架构设计文档
│   └── contributing.md        # 贡献指南
├── notebooks/                 # Jupyter notebooks（保持现有）
├── scripts/                   # 脚本目录（保持现有）
├── src/deepsc/               # 源代码主目录
│   ├── core/                 # 📁 新增：核心组件库
│   │   ├── __init__.py
│   │   ├── attention.py      # 通用注意力机制
│   │   ├── embeddings.py     # 通用嵌入层
│   │   ├── layers.py         # 通用神经网络层
│   │   └── base_model.py     # 基础模型抽象类
│   ├── models/               # 🔄 重构：模型目录
│   │   ├── __init__.py
│   │   ├── base/             # 基础模型类
│   │   │   ├── __init__.py
│   │   │   ├── model_base.py # 模型基类
│   │   │   └── registry.py   # 模型注册器
│   │   ├── deepsc/           # 🔄 合并：DeepSC模型（合并old和new）
│   │   │   ├── __init__.py
│   │   │   ├── model.py      # 主模型文件
│   │   │   └── components.py # 模型组件
│   │   ├── scbert/           # scBERT模型（保持现有）
│   │   └── scgpt/            # scGPT模型（保持现有）
│   ├── data/                 # 数据处理模块（保持现有）
│   ├── training/             # 🔄 重命名：train -> training
│   │   ├── __init__.py
│   │   ├── trainer.py        # 训练器
│   │   └── strategies.py     # 训练策略
│   ├── evaluation/           # 评估模块（保持现有）
│   ├── utils/                # 工具模块（保持现有）
│   ├── finetune.py          # 顶层接口文件（保持现有）
│   ├── pretrain.py          # 顶层接口文件（保持现有）
│   └── eval.py              # 顶层接口文件（保持现有）
├── tests/                    # 🔄 扩展：测试目录
│   ├── __init__.py
│   ├── unit/                 # 单元测试
│   │   ├── test_models.py
│   │   ├── test_data.py
│   │   └── test_utils.py
│   ├── integration/          # 集成测试
│   │   ├── test_pipeline.py
│   │   └── test_training.py
│   ├── conftest.py          # pytest配置
│   └── fixtures/            # 测试数据
├── tools/                    # 📁 新增：开发工具目录
│   ├── debug/                # 调试脚本
│   │   ├── debug_attention_issue.py  # 从根目录移动
│   │   └── gradient_flow_analysis.py # 从根目录移动
│   ├── analysis/             # 分析工具
│   └── maintenance/          # 维护脚本
├── .github/                  # 📁 新增：GitHub配置
│   ├── workflows/            # CI/CD 工作流
│   └── ISSUE_TEMPLATE.md     # Issue模板
├── pyproject.toml           # 项目配置（统一依赖管理）
├── requirements.txt         # 🔄 简化：仅用于开发依赖
├── README.md                # 主文档（保持现有）
├── README_zh_CN.md          # 中文文档（保持现有）
├── LICENSE                  # 许可证（保持现有）
└── CHANGELOG.md             # 📁 新增：变更日志
```

## 重构实施计划

### Phase 1: 基础设施重构（第1周）

#### 1.1 目录结构调整
- [ ] 创建新的目录结构
- [ ] 移动调试脚本到 `tools/debug/`
- [ ] 创建 `docs/` 目录并整理文档
- [ ] 设置 `.github/` 配置

#### 1.2 依赖管理统一
- [ ] 统一 `pyproject.toml` 和 `requirements.txt` 版本
- [ ] 修正 Python 版本要求冲突（统一为3.10+）
- [ ] 添加缺失的依赖声明
- [ ] 创建开发依赖分离

```toml
# pyproject.toml 修改示例
[project]
requires-python = ">=3.10"
dependencies = [
    "torch==2.6.0",
    "torchvision==0.21.0", 
    "torchaudio==2.6.0",
    # ... 其他核心依赖
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "black>=22.0.0",
    "isort>=5.0.0",
    "flake8>=5.0.0",
]
```

### Phase 2: 核心组件重构（第2周）

#### 2.1 创建核心组件库
```python
# src/deepsc/core/base_model.py
from abc import ABC, abstractmethod
import torch.nn as nn

class BaseModel(nn.Module, ABC):
    """所有模型的基础抽象类"""
    
    @abstractmethod
    def forward(self, x):
        pass
        
    @abstractmethod
    def get_embeddings(self, x):
        """获取特征嵌入"""
        pass
```

#### 2.2 提取公共组件
- [ ] 提取通用注意力机制到 `core/attention.py`
- [ ] 统一嵌入层实现到 `core/embeddings.py`
- [ ] 创建通用层到 `core/layers.py`
- [ ] 建立模型注册系统

#### 2.3 重构DeepSC模型
- [ ] 分析 `deepsc/` 和 `deepsc_new/` 差异
- [ ] 合并两个版本的优点
- [ ] 统一接口和配置
- [ ] 更新相关配置文件

### Phase 3: 接口标准化（第3周）

#### 3.1 统一模型接口
```python
# 标准模型接口示例
class ModelInterface:
    def __init__(self, config):
        pass
    
    def forward(self, batch):
        pass
        
    def get_loss(self, outputs, targets):
        pass
        
    def predict(self, batch):
        pass
```

#### 3.2 数据处理接口优化
- [ ] 标准化数据加载器接口
- [ ] 统一预处理流程
- [ ] 优化批处理逻辑

#### 3.3 训练接口重构
- [ ] 将 `train/` 重命名为 `training/`
- [ ] 统一训练策略接口
- [ ] 改进日志和监控系统

### Phase 4: 测试和文档完善（第4周）

#### 4.1 建立测试框架
```python
# tests/conftest.py
import pytest
import torch

@pytest.fixture
def sample_data():
    return torch.randn(32, 100, 512)

@pytest.fixture
def model_config():
    return {
        'dim': 512,
        'num_tokens': 7,
        'max_seq_len': 60664,
    }
```

#### 4.2 编写单元测试
- [ ] 核心组件测试（attention、embeddings）
- [ ] 模型组件测试
- [ ] 数据处理测试
- [ ] 工具函数测试

#### 4.3 集成测试
- [ ] 端到端训练流程测试
- [ ] 模型推理测试
- [ ] 配置加载测试

#### 4.4 文档完善
- [ ] API文档自动生成
- [ ] 架构设计文档
- [ ] 使用教程和示例
- [ ] 贡献指南

## 代码重构细节

### 1. 模型基类设计

```python
# src/deepsc/core/base_model.py
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Any, Optional

class BaseFoundationModel(nn.Module, ABC):
    """单细胞基础模型的抽象基类"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.model_name = self.__class__.__name__
        
    @abstractmethod
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """前向传播"""
        pass
        
    @abstractmethod
    def get_embeddings(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """获取细胞嵌入表示"""
        pass
        
    @abstractmethod
    def compute_loss(self, outputs: Dict[str, torch.Tensor], 
                    targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算损失"""
        pass
        
    def save_pretrained(self, save_path: str):
        """保存预训练模型"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'model_name': self.model_name
        }, save_path)
        
    @classmethod
    def from_pretrained(cls, model_path: str):
        """加载预训练模型"""
        checkpoint = torch.load(model_path)
        model = cls(checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
```

### 2. 注意力机制统一

```python
# src/deepsc/core/attention.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class UnifiedAttention(nn.Module):
    """统一的注意力机制实现"""
    
    def __init__(self, dim: int, num_heads: int = 8, 
                 dropout: float = 0.1, flash_attention: bool = True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.flash_attention = flash_attention and hasattr(F, 'scaled_dot_product_attention')
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, D = x.shape
        
        # 生成Q, K, V
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        if self.flash_attention:
            # 使用Flash Attention
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=self.dropout.p if self.training else 0.0
            )
        else:
            # 传统注意力计算
            attn = (q @ k.transpose(-2, -1)) * self.scale
            if mask is not None:
                attn.masked_fill_(mask == 0, float('-inf'))
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            out = attn @ v
            
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.proj(out)
```

### 3. 模型注册系统

```python
# src/deepsc/models/base/registry.py
from typing import Dict, Type, Callable
from ..core.base_model import BaseFoundationModel

class ModelRegistry:
    """模型注册器"""
    
    _models: Dict[str, Type[BaseFoundationModel]] = {}
    
    @classmethod
    def register(cls, name: str):
        """注册装饰器"""
        def decorator(model_class: Type[BaseFoundationModel]):
            cls._models[name] = model_class
            return model_class
        return decorator
    
    @classmethod
    def get_model(cls, name: str) -> Type[BaseFoundationModel]:
        """获取模型类"""
        if name not in cls._models:
            raise ValueError(f"Model '{name}' not found. Available: {list(cls._models.keys())}")
        return cls._models[name]
    
    @classmethod
    def list_models(cls) -> list:
        """列出所有注册的模型"""
        return list(cls._models.keys())

# 使用示例
@ModelRegistry.register('deepsc')
class DeepSCModel(BaseFoundationModel):
    pass
```

## 迁移策略

### 向后兼容性保证
1. **渐进式迁移**：保留原有接口，逐步引入新接口
2. **别名支持**：为重命名的模块提供别名
3. **版本标记**：明确标注哪些接口将被废弃
4. **迁移指南**：提供详细的代码迁移说明

### 测试验证
1. **基准测试**：重构前后性能对比
2. **功能测试**：确保所有功能正常工作
3. **回归测试**：验证模型输出一致性

## 风险评估与应对

### 潜在风险
1. **配置兼容性**：Hydra配置文件可能需要调整
2. **模型权重**：预训练模型加载可能受影响
3. **依赖冲突**：新的依赖版本可能引起冲突

### 应对措施
1. **分支开发**：在特性分支进行重构，主分支保持稳定
2. **自动化测试**：建立CI/CD流程，自动化测试所有更改
3. **回滚计划**：准备快速回滚到重构前状态的方案

## 成功指标

### 定量指标
- [ ] 代码重复率减少 30%
- [ ] 测试覆盖率达到 80%+
- [ ] 构建时间减少 20%
- [ ] 文档覆盖率达到 90%+

### 定性指标
- [ ] 新开发者上手时间缩短
- [ ] 代码审查效率提升
- [ ] 模块间耦合度降低
- [ ] 整体架构清晰度提升

## 时间规划

| 阶段 | 时间 | 主要任务 | 里程碑 |
|------|------|----------|---------|
| Phase 1 | 第1周 | 基础设施重构 | 新目录结构建立 |
| Phase 2 | 第2周 | 核心组件重构 | 公共组件提取完成 |
| Phase 3 | 第3周 | 接口标准化 | 统一接口规范 |
| Phase 4 | 第4周 | 测试文档完善 | 测试和文档达标 |

## 后续维护

### 代码质量工具
```yaml
# .github/workflows/quality.yml
name: Code Quality
on: [push, pull_request]
jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
      - name: Run black
        run: black --check src/ tests/
      - name: Run isort
        run: isort --check-only src/ tests/
      - name: Run flake8
        run: flake8 src/ tests/
      - name: Run tests
        run: pytest tests/ --cov=src/deepsc --cov-report=xml
```

### 持续改进
1. **代码审查制度**：所有变更必须经过代码审查
2. **性能监控**：建立性能基准和监控系统
3. **文档维护**：保持文档与代码同步更新
4. **依赖更新**：定期更新依赖版本，确保安全性

## 总结

本重构方案旨在通过系统性的目录重组、代码重构和接口统一，显著提升 DeepSC 项目的代码质量、可维护性和扩展性。通过分阶段实施、严格测试和风险控制，确保重构过程平稳进行，最终交付一个结构清晰、易于维护和扩展的高质量项目。

---

*此文档为项目重构的指导性文件，将根据实际实施情况进行动态更新。*