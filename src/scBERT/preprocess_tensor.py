import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
import torch 

pth_data_path = "/home/angli/DeepSC/data/3ac/mapped_batch_data/1d84333c-0327-4ad6-be02-94fee81154ff_sparse.pth"
coo_tensor = torch.load(pth_data_path)
if not coo_tensor.is_coalesced():
    raise ValueError("输入的稀疏张量不是 coalesced 格式，请先调用 coalesce() 进行合并")

values = coo_tensor.values().cpu().numpy()
indices = coo_tensor.indices().cpu().numpy()
shape = coo_tensor.shape
scipy_sparse = sparse.coo_matrix((values, (indices[0], indices[1])), shape=shape).tocsr()
adata = ad.AnnData(X=scipy_sparse)
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata, base=2)
processed_coo = adata.X.tocoo()

torch_sparse = torch.sparse_coo_tensor(
    indices=np.vstack((processed_coo.row, processed_coo.col)),
    values=processed_coo.data,
    size=processed_coo.shape
)

torch.save(torch_sparse, "/home/angli/DeepSC/data/3ac/mapped_batch_data/1d84333c-0327-4ad6-be02-94fee81154ff_sparse_preprocessed.pth")

""" 
panglao = sc.read_h5ad("./data/panglao_10000.h5ad")
data = sc.read_h5ad("./data/your_raw_data.h5ad")
counts = sparse.lil_matrix((data.X.shape[0], panglao.X.shape[1]), dtype=np.float32)
ref = panglao.var_names.tolist()
obj = data.var_names.tolist()

for i in range(len(ref)):
    if ref[i] in obj:
        loc = obj.index(ref[i])
        counts[:, i] = data.X[:, loc]

counts = counts.tocsr()
new = ad.AnnData(X=counts)
new.var_names = ref
new.obs_names = data.obs_names
new.obs = data.obs
new.uns = panglao.uns

sc.pp.filter_cells(new, min_genes=200)
sc.pp.normalize_total(new, target_sum=1e4)
sc.pp.log1p(new, base=2)
new.write("./data/preprocessed_data.h5ad")
 """