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
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 # torch==2.5.1, torchvision==0.20.1, torchaudio==2.5.1
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
pip install -r requirements.txt
pip install -e .
```


## Datasets

Cellxgene ...

