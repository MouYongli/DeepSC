# -*- coding: utf-8 -*-
import os
import gc
import argparse
import json
import random
import math
import random
from functools import reduce
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split, ShuffleSplit, StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support, classification_report
import torch
from torch import nn
from torch.optim import Adam, SGD, AdamW
from torch.nn import functional as F
from performer_pytorch import PerformerLM
import scanpy as sc
import anndata as ad
from utils import *
import pickle as pkl

parser = argparse.ArgumentParser()
parser.add_argument("--bin_num", type=int, default=5, help='Number of bins.')
parser.add_argument("--gene_num", type=int, default=16906, help='Number of genes.')
parser.add_argument("--epoch", type=int, default=100, help='Number of epochs.')
parser.add_argument("--seed", type=int, default=2021, help='Random seed.')
parser.add_argument("--novel_type", type=bool, default=False, help='Novel cell tpye exists or not.')
parser.add_argument("--unassign_thres", type=float, default=0.5, help='The confidence score threshold for novel cell type annotation.')
parser.add_argument("--pos_embed", type=bool, default=True, help='Using Gene2vec encoding or not.')
parser.add_argument("--data_path", type=str, default='./data/zheng_train_subsample.h5ad', help='Path of data for predicting.')
parser.add_argument("--model_path", type=str, default='./ckpts/finetuned.pth', help='Path of finetuned model.')
parser.add_argument("--label_prefix", type=str, default='', help='Prefix for label_dict and label files (e.g., "finetune_zheng68k_")')

args = parser.parse_args()

SEED = args.seed
EPOCHS = args.epoch
SEQ_LEN = args.gene_num + 1
UNASSIGN = args.novel_type
UNASSIGN_THRES = args.unassign_thres if UNASSIGN == True else 0

CLASS = args.bin_num + 2
POS_EMBED_USING = args.pos_embed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Identity(torch.nn.Module):
    def __init__(self, dropout = 0., h_dim = 100, out_dim = 10):
        super(Identity, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, (1, 200))
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(in_features=SEQ_LEN, out_features=512, bias=True)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(in_features=512, out_features=h_dim, bias=True)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(in_features=h_dim, out_features=out_dim, bias=True)

    def forward(self, x):
        x = x[:,None,:,:]
        x = self.conv1(x)
        x = self.act(x)
        x = x.view(x.shape[0],-1)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

data = sc.read_h5ad(args.data_path)
# Get true labels
true_labels = data.obs['celltype'].values

#load the label stored during the fine-tune stage
label_dict_file = f'{args.label_prefix}label_dict' if args.label_prefix else 'label_dict'
label_file = f'{args.label_prefix}label' if args.label_prefix else 'label'
print(f"Loading label files: {label_dict_file}, {label_file}")
with open(label_dict_file, 'rb') as fp:
    label_dict = pkl.load(fp)
with open(label_file, 'rb') as fp:
    label = pkl.load(fp)
print(f"Loaded {len(label_dict)} classes: {label_dict}")

class_num = np.unique(label, return_counts=True)[1].tolist()
class_weight = torch.tensor([(1 - (x / sum(class_num))) ** 2 for x in class_num])
label = torch.from_numpy(label)
data_X = data.X

model = PerformerLM(
    num_tokens = CLASS,
    dim = 200,
    depth = 6,
    max_seq_len = SEQ_LEN,
    heads = 10,
    local_attn_heads = 0,
    g2v_position_emb = True
)
model.to_out = Identity(dropout=0., h_dim=128, out_dim=label_dict.shape[0])
print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters.")
path = args.model_path
ckpt = torch.load(path)
model.load_state_dict(ckpt['model_state_dict'])
for param in model.parameters():
    param.requires_grad = False
model = model.to(device)

batch_size = data_X.shape[0]
model.eval()
pred_finals = []
pred_probs_list = []
novel_indices = []
with torch.no_grad():
    for index in range(batch_size):
        full_seq = data_X[index].toarray()[0]
        full_seq[full_seq > (CLASS - 2)] = CLASS - 2
        full_seq = torch.from_numpy(full_seq).long()
        full_seq = torch.cat((full_seq, torch.tensor([0]))).to(device)
        full_seq = full_seq.unsqueeze(0)
        pred_logits = model(full_seq)
        softmax = nn.Softmax(dim=-1)
        pred_prob = softmax(pred_logits)
        pred_final = pred_prob.argmax(dim=-1).item()
        max_prob = np.amax(np.array(pred_prob.cpu()), axis=-1)
        pred_probs_list.append(max_prob)
        if max_prob < UNASSIGN_THRES:
            novel_indices.append(index)
        pred_finals.append(pred_final)
        print(f"Processed {index+1}/{batch_size} cells.", end='\r')

pred_list = label_dict[pred_finals].tolist()
for index in novel_indices:
    pred_list[index] = 'Unassigned'

print("="*60)
print("PREDICTION RESULTS WITH METRICS")
print("="*60)

# Compute metrics
accuracy = accuracy_score(true_labels, pred_list)
f1_macro = f1_score(true_labels, pred_list, average='macro', zero_division=0)
f1_weighted = f1_score(true_labels, pred_list, average='weighted', zero_division=0)

print(f"\nOverall Metrics:")
print(f"  Accuracy:        {accuracy:.4f}")
print(f"  F1-Score (macro):    {f1_macro:.4f}")
print(f"  F1-Score (weighted): {f1_weighted:.4f}")
print(f"  Unassigned cells:    {len(novel_indices)}/{len(pred_list)}")

print(f"\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
print(classification_report(true_labels, pred_list, zero_division=0))

print("="*60)
print("CONFUSION MATRIX")
print("="*60)
# Get unique labels from both true and predicted
all_labels = sorted(list(set(true_labels) | set(pred_list)))
cm = confusion_matrix(true_labels, pred_list, labels=all_labels)
cm_df = pd.DataFrame(cm, index=all_labels, columns=all_labels)
print(cm_df)

print("\n" + "="*60)
print("PREDICTION CONFIDENCE STATISTICS")
print("="*60)
print(f"  Mean confidence: {np.mean(pred_probs_list):.4f}")
print(f"  Std confidence:  {np.std(pred_probs_list):.4f}")
print(f"  Min confidence:  {np.min(pred_probs_list):.4f}")
print(f"  Max confidence:  {np.max(pred_probs_list):.4f}")

# Per-class accuracy
print("\n" + "="*60)
print("PER-CLASS ACCURACY")
print("="*60)
for label_name in all_labels:
    if label_name == 'Unassigned':
        continue
    mask = true_labels == label_name
    if mask.sum() > 0:
        class_acc = accuracy_score(true_labels[mask], np.array(pred_list)[mask])
        class_count = mask.sum()
        print(f"  {label_name:35s}: {class_acc:.4f} ({class_count} samples)")

print("="*60)
