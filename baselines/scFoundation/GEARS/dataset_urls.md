# GEARS Perturbation Prediction Datasets

## 三个主要数据集（自动下载）

- **Norman数据集**: https://dataverse.harvard.edu/api/access/datafile/6154020
- **Adamson数据集**: https://dataverse.harvard.edu/api/access/datafile/6154417
- **Dixit数据集**: https://dataverse.harvard.edu/api/access/datafile/6154416

## GO基因注释文件

- **gene2go文件**: https://dataverse.harvard.edu/api/access/datafile/6153417

## 使用方法

```python
from gears import PertData

# 初始化
pert_data = PertData('./data')

# 加载数据集（会自动从Harvard Dataverse下载）
pert_data.load(data_name='norman')  # 或 'adamson', 'dixit'
```
