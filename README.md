# Deep Transcriptomic Foundation Models for Single-Cell RNA-Sequencing Data

This is official repo for "Deep Foundation Models for Single Cell RNA Sequencing"  by DBIS at RWTH Aachen University 
[Yongli Mou*](mou@dbis.rwth-aachen.de), Ang Li, Sikander Hayat, Stefan Decker

## Python Environment Setup

1. conda environment
```
conda create --name deepsc python=3.11
conda activate deepsc
```

2. jupyter lab and kernel
```
conda install -c conda-forge jupyterlab
conda install ipykernel
```

```
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu121.html
pip install -r requirements.txt
pip install -e .
```


## Datasets

Cellxgene ...

