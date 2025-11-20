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
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, CyclicLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from performer_pytorch import PerformerLM
import scanpy as sc
import anndata as ad
from utils import *
import pickle as pkl

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=-1, help='Local process rank.')
parser.add_argument("--bin_num", type=int, default=5, help='Number of bins.')
parser.add_argument("--gene_num", type=int, default=16906, help='Number of genes.')
parser.add_argument("--epoch", type=int, default=10, help='Number of epochs.')
parser.add_argument("--seed", type=int, default=2021, help='Random seed.')
parser.add_argument("--batch_size", type=int, default=3, help='Number of batch size.')
parser.add_argument("--learning_rate", type=float, default=1e-4, help='Learning rate.')
parser.add_argument("--grad_acc", type=int, default=20, help='Number of gradient accumulation.')
parser.add_argument("--valid_every", type=int, default=1, help='Number of training epochs between twice validation.')
parser.add_argument("--pos_embed", type=bool, default=True, help='Using Gene2vec encoding or not.')
parser.add_argument("--data_path", type=str, default='./data/Zheng68K_train.h5ad', help='Path of data for finetune.')
parser.add_argument("--val_data_path", type=str, default=None, help='Path of validation data. If provided, will use data_path as train and this as validation without splitting.')
parser.add_argument("--model_path", type=str, default='./panglao_pretrain.pth', help='Path of pretrained model.')
parser.add_argument("--ckpt_dir", type=str, default='./ckpts/', help='Directory of checkpoint to save.')
parser.add_argument("--model_name", type=str, default='finetune_zheng_train_', help='Finetuned model name.')

args = parser.parse_args()

# Check if running in distributed mode
USE_DISTRIBUTED = "RANK" in os.environ and "WORLD_SIZE" in os.environ

if USE_DISTRIBUTED:
    # Get local_rank from environment variable (for torchrun) or command line argument
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    if local_rank == -1:
        local_rank = 0

    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    world_size = torch.distributed.get_world_size()

    # If only 1 GPU, use rank 0. If multiple GPUs, use rank 1 for printing
    if world_size == 1:
        is_master = local_rank == 0
    else:
        is_master = local_rank == 1
else:
    # Single GPU mode
    local_rank = 0
    is_master = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    world_size = 1

SEED = args.seed
EPOCHS = args.epoch
BATCH_SIZE = args.batch_size
GRADIENT_ACCUMULATION = args.grad_acc
LEARNING_RATE = args.learning_rate
SEQ_LEN = args.gene_num + 1
VALIDATE_EVERY = args.valid_every

PATIENCE = 10
UNASSIGN_THRES = 0.0

CLASS = args.bin_num + 2
POS_EMBED_USING = args.pos_embed

model_name = args.model_name
ckpt_dir = args.ckpt_dir

seed_all(SEED + (torch.distributed.get_rank() if USE_DISTRIBUTED else 0))


class SCDataset(Dataset):
    def __init__(self, data, label):
        super().__init__()
        self.data = data
        self.label = label

    def __getitem__(self, index):
        rand_start = random.randint(0, self.data.shape[0]-1)
        full_seq = self.data[rand_start].toarray()[0]
        full_seq[full_seq > (CLASS - 2)] = CLASS - 2
        full_seq = torch.from_numpy(full_seq).long()
        full_seq = torch.cat((full_seq, torch.tensor([0], dtype=torch.long)))
        seq_label = self.label[rand_start]
        return full_seq, seq_label

    def __len__(self):
        return self.data.shape[0]


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

# Check if separate validation dataset is provided
if args.val_data_path is not None:
    # Use provided train and validation datasets without splitting
    if is_master:
        print(f"Loading training data from: {args.data_path}")
        print(f"Loading validation data from: {args.val_data_path}")

    # Load training data
    data_train_full = sc.read_h5ad(args.data_path)
    label_dict, label_train = np.unique(np.array(data_train_full.obs['celltype']), return_inverse=True)

    # Load validation data
    data_val_full = sc.read_h5ad(args.val_data_path)
    # Map validation labels using the same label_dict from training data
    label_val_raw = np.array(data_val_full.obs['celltype'])
    label_val = np.zeros(len(label_val_raw), dtype=int)
    for i, cell_type in enumerate(label_val_raw):
        if cell_type in label_dict:
            label_val[i] = np.where(label_dict == cell_type)[0][0]
        else:
            raise ValueError(f"Cell type '{cell_type}' in validation data not found in training data label dictionary")

    # Store the label dict and label for prediction
    label_dict_file = f'{args.model_name}_label_dict'
    label_file = f'{args.model_name}_label'
    with open(label_dict_file, 'wb') as fp:
        pkl.dump(label_dict, fp)
    with open(label_file, 'wb') as fp:
        pkl.dump(np.concatenate([label_train, label_val]), fp)
    if is_master:
        print(f"Saved label files: {label_dict_file}, {label_file}")

    class_num = np.unique(label_train, return_counts=True)[1].tolist()
    class_weight = torch.tensor([(1 - (x / sum(class_num))) ** 2 for x in class_num])

    label_train = torch.from_numpy(label_train)
    label_val = torch.from_numpy(label_val)
    data_train = data_train_full.X
    data_val = data_val_full.X

    train_dataset = SCDataset(data_train, label_train)
    val_dataset = SCDataset(data_val, label_val)

    if is_master:
        print(f"Using separate datasets - Training: {data_train.shape[0]} samples, Validation: {data_val.shape[0]} samples")
else:
    # Original behavior: load single dataset and split internally
    if is_master:
        print(f"Loading data from: {args.data_path}")
        print("Will split into train/validation internally")

    data = sc.read_h5ad(args.data_path)
    label_dict, label = np.unique(np.array(data.obs['celltype']), return_inverse=True)

    # Store the label dict and label for prediction
    label_dict_file = f'{args.model_name}_label_dict'
    label_file = f'{args.model_name}_label'
    with open(label_dict_file, 'wb') as fp:
        pkl.dump(label_dict, fp)
    with open(label_file, 'wb') as fp:
        pkl.dump(label, fp)
    if is_master:
        print(f"Saved label files: {label_dict_file}, {label_file}")

    class_num = np.unique(label, return_counts=True)[1].tolist()
    class_weight = torch.tensor([(1 - (x / sum(class_num))) ** 2 for x in class_num])
    label = torch.from_numpy(label)
    data = data.X

    acc = []
    f1 = []
    f1w = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    pred_list = pd.Series(['un'] * data.shape[0])

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    for index_train, index_val in sss.split(data, label):
        data_train, label_train = data[index_train], label[index_train]
        data_val, label_val = data[index_val], label[index_val]
        train_dataset = SCDataset(data_train, label_train)
        val_dataset = SCDataset(data_val, label_val)

    if is_master:
        print(f"Split dataset - Training: {data_train.shape[0]} samples, Validation: {data_val.shape[0]} samples")

if USE_DISTRIBUTED:
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)
else:
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
print(f'Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}')
model = PerformerLM(
    num_tokens = CLASS,
    dim = 200,
    depth = 6,
    max_seq_len = SEQ_LEN,
    heads = 10,
    local_attn_heads = 0,
    g2v_position_emb = POS_EMBED_USING
)
print(model)
path = args.model_path
ckpt = torch.load(path)
model.load_state_dict(ckpt['model_state_dict'])
for param in model.parameters():
    param.requires_grad = False
for param in model.norm.parameters():
    param.requires_grad = True
for param in model.performer.net.layers[-2].parameters():
    param.requires_grad = True
model.to_out = Identity(dropout=0., h_dim=128, out_dim=label_dict.shape[0])
model = model.to(device)
if USE_DISTRIBUTED:
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
print(f'Model parameters to learn:')
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f'\t{name}')  
# optimizer
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = CosineAnnealingWarmupRestarts(
    optimizer,
    first_cycle_steps=15,
    cycle_mult=2,
    max_lr=LEARNING_RATE,
    min_lr=1e-6,
    warmup_steps=5,
    gamma=0.9
)
loss_fn = nn.CrossEntropyLoss(weight=None).to(device)
print('Start training...')
if USE_DISTRIBUTED:
    dist.barrier()
trigger_times = 0
max_acc = 0.0
for i in range(1, EPOCHS+1):
    if USE_DISTRIBUTED:
        train_loader.sampler.set_epoch(i)
    model.train()
    if USE_DISTRIBUTED:
        dist.barrier()
    running_loss = 0.0
    cum_acc = 0.0
    for index, (data, labels) in enumerate(train_loader):
        index += 1
        data, labels = data.to(device), labels.to(device)
        if index % GRADIENT_ACCUMULATION != 0:
            if USE_DISTRIBUTED:
                with model.no_sync():
                    logits = model(data)
                    loss = loss_fn(logits, labels)
                    loss.backward()
            else:
                logits = model(data)
                loss = loss_fn(logits, labels)
                loss.backward()
        if index % GRADIENT_ACCUMULATION == 0:
            logits = model(data)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), int(1e6))
            optimizer.step()
            optimizer.zero_grad()
        running_loss += loss.item()
        softmax = nn.Softmax(dim=-1)
        final = softmax(logits)
        final = final.argmax(dim=-1)
        pred_num = labels.size(0)
        correct_num = torch.eq(final, labels).sum(dim=-1)
        cum_acc += torch.true_divide(correct_num, pred_num).mean().item()
        print(f'    ==  Epoch: {i} | Step: {index}/{len(train_loader)} | Training Loss: {loss.item():.6f}  ==', end='\r')
    epoch_loss = running_loss / index
    epoch_acc = 100 * cum_acc / index
    print(f'    ==  Epoch: {i} | Training Loss: {epoch_loss:.6f} | Accuracy: {epoch_acc:6.4f}%  ==')
    if USE_DISTRIBUTED:
        epoch_loss = get_reduced(epoch_loss, local_rank, 0, world_size)
        epoch_acc = get_reduced(epoch_acc, local_rank, 0, world_size)
    if is_master:
        print("is master")
        print(f'    ==  Epoch: {i} | Training Loss: {epoch_loss:.6f} | Accuracy: {epoch_acc:6.4f}%  ==')
    if USE_DISTRIBUTED:
        dist.barrier()
    scheduler.step()

    # Evaluate at the end of every epoch
    model.eval()
    if USE_DISTRIBUTED:
        dist.barrier()
    running_loss = 0.0
    predictions = []
    truths = []
    with torch.no_grad():
        for index, (data_v, labels_v) in enumerate(val_loader):
            index += 1
            data_v, labels_v = data_v.to(device), labels_v.to(device)
            logits = model(data_v)
            loss = loss_fn(logits, labels_v)
            running_loss += loss.item()
            softmax = nn.Softmax(dim=-1)
            final_prob = softmax(logits)
            final = final_prob.argmax(dim=-1)
            final[np.amax(np.array(final_prob.cpu()), axis=-1) < UNASSIGN_THRES] = -1
            predictions.append(final)
            truths.append(labels_v)
            print(f'    ==  Epoch: {i} | Step: {index}/{len(val_loader)} | Validation Loss: {loss.item():.6f}  ==', end='\r')
        del data_v, labels_v, logits, final_prob, final
        # gather
        val_loss = running_loss / index
        if USE_DISTRIBUTED:
            predictions = distributed_concat(torch.cat(predictions, dim=0), len(val_sampler.dataset), world_size)
            truths = distributed_concat(torch.cat(truths, dim=0), len(val_sampler.dataset), world_size)
            val_loss = get_reduced(val_loss, local_rank, 0, world_size)
        else:
            predictions = torch.cat(predictions, dim=0)
            truths = torch.cat(truths, dim=0)
        
        # Only compute metrics on master to avoid potential issues on non-master ranks
        if is_master:
            no_drop = predictions != -1
            predictions_np = np.array((predictions[no_drop]).cpu())
            truths_np = np.array((truths[no_drop]).cpu())
            cur_acc = accuracy_score(truths_np, predictions_np)
            f1 = f1_score(truths_np, predictions_np, average='macro')
            
            print(f'    ==  Epoch: {i} | Validation Loss: {val_loss:.6f} | Accuracy: {cur_acc:.6f} | F1 Score: {f1:.6f}  ==')
            print(confusion_matrix(truths_np, predictions_np))
            print(classification_report(truths_np, predictions_np, target_names=[str(x) for x in label_dict.tolist()], labels=range(len(label_dict)), digits=4, zero_division=0))

            if cur_acc > max_acc:
                max_acc = cur_acc
                trigger_times = 0
                save_best_ckpt(i, model, optimizer, scheduler, val_loss, model_name, ckpt_dir)
            else:
                trigger_times += 1
                if trigger_times > PATIENCE:
                    print(f'Early stopping at epoch {i}')
                    break
        del predictions, truths