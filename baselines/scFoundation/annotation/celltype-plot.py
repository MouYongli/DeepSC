"""
Converted from celltype-plot.ipynb
Cell type annotation and visualization script
"""

# Cell 0
# %pylab inline  # Note: %pylab is deprecated, use %matplotlib inline instead
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving figures
import scanpy as sc
import pandas as pd
import pickle

# Cell 1
seg_label = np.load('./data/Segerstolpe-test-label.npy')
seg_name = np.load('./data/Segerstolpe-str_label.npy')

# Cell 2
emb_path = "./data/seg-emb.pkl"
emb = None
label = None
f = open(emb_path, 'rb')
while 1:
    try:
        sub_pkl = pickle.load(f)
        tmp_emb = sub_pkl["emb"]
        tmp_label = sub_pkl["label"]
        if emb is None:
            emb = tmp_emb
            label = tmp_label
        else:
            emb = np.vstack([emb, tmp_emb])
            label = np.concatenate([label, tmp_label])
    except:
        break

emb = emb[-seg_label.shape[0]:, :]
label = label[-seg_label.shape[0]:]

# Cell 3
emb_path = "./data/seg-cellemb.pkl"
cellemb = None
label = None
f = open(emb_path, 'rb')
while 1:
    try:
        sub_pkl = pickle.load(f)
        tmp_emb = sub_pkl["emb"]
        tmp_label = sub_pkl["label"]
        if cellemb is None:
            cellemb = tmp_emb
            label = tmp_label
        else:
            cellemb = np.vstack([cellemb, tmp_emb])
            label = np.concatenate([label, tmp_label])
    except:
        break

cellemb = cellemb[-seg_label.shape[0]:, :]
label = label[-seg_label.shape[0]:]

# Cell 4
for i in range(seg_label.shape[0]):
    assert seg_label[i] == label[i]

# Cell 5
y_pred = np.argmax(emb, 1)

# Cell 6
segadata = sc.AnnData(cellemb)
sc.pp.neighbors(segadata, use_rep='X')

sc.tl.umap(segadata)

segadata.obs['true_label'] = seg_name[seg_label]
segadata.obs['pred_label'] = seg_name[y_pred]

# Cell 7
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
sc.pl.umap(segadata, color=['true_label', 'pred_label'], wspace=0.3, size=50, save='scf_seg')

# Cell 8
ctpadata = sc.read_h5ad('./data/celltypist_0806_seg.h5ad')

ctpadata.obs['predict_strlabels'] = seg_name[ctpadata.obs['predicted_labels'].values]
ctpadata.obs['predict_strlabels'] = ctpadata.obs['predict_strlabels'].astype('category')

sc.tl.umap(ctpadata)

# Cell 9
pal = {}
for i in range(len(seg_name)):
    pal[segadata.obs.true_label.cat.categories[i]] = segadata.uns['true_label_colors'][i]

# Cell 10
sc.pl.umap(ctpadata, color=['true_strlabels', 'predict_strlabels'], wspace=0.3, size=50, palette=pal, save='celltypist_seg')

# Cell 11
# Empty cell

# Cell 12
# Empty cell

# Cell 13
ori_zheng_label = np.load('./data/zheng-test-label.npy')
zheng_name = np.load('./data/zheng-str_label.npy')

# Cell 14
zhengemb_path = "./data/zheng-emb-2mlp.pkl"
zheng_emb = None
zheng_label = None
f = open(zhengemb_path, 'rb')
while 1:
    try:
        sub_pkl = pickle.load(f)
        tmp_emb = sub_pkl["emb"]
        tmp_label = sub_pkl["label"]
        if zheng_emb is None:
            zheng_emb = tmp_emb
            zheng_label = tmp_label
        else:
            zheng_emb = np.vstack([zheng_emb, tmp_emb])
            zheng_label = np.concatenate([zheng_label, tmp_label])
    except:
        break

# Cell 15
for i in range(zheng_label.shape[0]):
    assert zheng_label[i] == ori_zheng_label[i]

# Cell 16
y_pred = np.argmax(zheng_emb, 1)

# Cell 17
from sklearn.metrics import classification_report
print(classification_report(zheng_label, y_pred, target_names=zheng_name))

# Cell 18
# Empty cell

# Cell 19
zhengemb_path = "./data/zheng-cellemb-2mlp.pkl"
zheng_emb = None
zheng_label = None
f = open(zhengemb_path, 'rb')
while 1:
    try:
        sub_pkl = pickle.load(f)
        tmp_emb = sub_pkl["emb"]
        tmp_label = sub_pkl["label"]
        if zheng_emb is None:
            zheng_emb = tmp_emb
            zheng_label = tmp_label
        else:
            zheng_emb = np.vstack([zheng_emb, tmp_emb])
            zheng_label = np.concatenate([zheng_label, tmp_label])
    except:
        break

# Cell 20
print(zheng_emb.shape)

# Cell 21
zhengadata = sc.AnnData(zheng_emb)
sc.pp.neighbors(zhengadata, use_rep='X')
sc.tl.umap(zhengadata)
zhengadata.obs['true_label'] = zheng_name[zheng_label]
zhengadata.obs['pred_label'] = zheng_name[y_pred]

# Cell 22
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
sc.pl.umap(zhengadata, color=['true_label', 'pred_label'], wspace=0.5, size=10, save='scf_zheng')

# Cell 23
ctpzhengadata = sc.read_h5ad('./data/celltypist_0806_zheng68k.h5ad')

ctpzhengadata.obs['predict_strlabels'] = zheng_name[ctpzhengadata.obs['predicted_labels'].values]
ctpzhengadata.obs['predict_strlabels'] = ctpzhengadata.obs['predict_strlabels'].astype('category')

sc.tl.umap(ctpzhengadata)

# Cell 24
from sklearn.metrics import classification_report
print(classification_report(ctpzhengadata.obs['true_strlabels'], ctpzhengadata.obs['predict_strlabels'], target_names=zheng_name))

# Cell 25
pal = {}
for i in range(len(zheng_name)):
    pal[zhengadata.obs.true_label.cat.categories[i]] = zhengadata.uns['true_label_colors'][i]

# Cell 26
sc.pl.umap(ctpzhengadata, color=['true_strlabels', 'predict_strlabels'], wspace=0.5, size=10, palette=pal, save='celltypist_zheng')

# Cell 27
# Empty cell

