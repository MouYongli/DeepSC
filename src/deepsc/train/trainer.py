# -*- coding: utf-8 -*-
import os
import gc
import argparse
import json
import random
import logging
from datetime import datetime
import os.path as osp
import math
import random
from functools import reduce
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import scanpy as sc
import anndata as ad
from utils import *
from lightning.fabric import Fabric
from src.data.dataset import SCDataset, extract_rows_from_sparse_tensor
from src.data.preprocess import data_mask
from src.models import select_model

def setup_logging(type, log_path):
    os.makedirs(log_path, exist_ok=True)  # 确保日志目录存在

    # 生成带时间戳的日志文件名
    timestamp = 'today'
    log_file = osp.join(log_path, f"pretrain_{type}_{timestamp}.log")

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filemode="w",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)

    logging.info(f"日志文件: {log_file}")

    return log_file

class Trainer:
    def __init__(self, args):
        self.args = args
        seed_all(args.seed + dist.get_rank())
        self.world_size = dist.get_world_size()
        self.device = torch.device("cuda", args.local_rank)
        self.is_master = int(os.environ["RANK"]) == 0
        setup_logging(dist.get_rank(), './')

        #导入并配置fabric
        self.fabric = Fabric(accelerator="cuda", devices=1, strategy="ddp")
        self.fabric.launch()
        #载入数据并设置模型
        self.load_data()
        self.build_model()



    def load_data(self):
        coo_tensor = torch.load(self.args.data_path)
        #TODO: 需支持文件夹为路径传入
        row_indices = np.arange(coo_tensor.shape[0])
        train_idx, val_idx = train_test_split(row_indices, test_size=0.05, random_state=self.args.seed)

        data_train = extract_rows_from_sparse_tensor(coo_tensor, train_idx)
        data_val = extract_rows_from_sparse_tensor(coo_tensor, val_idx)

        self.train_dataset = SCDataset(data_train, self.args)
        self.val_dataset = SCDataset(data_val, self.args)

        train_sampler = DistributedSampler(self.train_dataset)
        val_sampler = SequentialDistributedSampler(self.val_dataset, batch_size=self.args.batch_size, world_size=self.world_size)

        train_loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, sampler=train_sampler)
        val_loader = DataLoader(self.val_dataset, batch_size=self.args.batch_size, sampler=val_sampler)

        self.train_loader, self.val_loader = self.fabric.setup_dataloaders(train_loader, val_loader)

    def build_model(self):
        args = self.args
        self.model = select_model(args)  
        #TODO:optimizer scheduler loss_fn 是否需要custimizable
        #TODO: 为optimizer和scheduler写一个单独的函数
        self.optimizer = Adam(self.model.parameters(), lr=args.learning_rate)
        self.scheduler = CosineAnnealingWarmupRestarts(
            self.optimizer,
            first_cycle_steps=15,
            cycle_mult=2,
            max_lr=args.learning_rate,
            min_lr=1e-6,
            warmup_steps=5,
            gamma=0.9
        )
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=args.bin_num + 1, reduction='mean')
        self.softmax = nn.Softmax(dim=-1)
        self.model, self.optimizer = self.fabric.setup(self.model, self.optimizer)


    def validate(self, epoch):
        self.model.eval()
        running_loss, predictions, truths = 0.0, [], []
        with torch.no_grad():
            for index, data in enumerate(self.val_loader):
                data = data.to(self.device)
                data, labels = data_mask(data, args=self.args)
                logits = self.model(data)
                loss = self.loss_fn(logits.transpose(1, 2), labels)
                running_loss += loss.item()
                final = self.softmax(logits)[..., 1:-1].argmax(dim=-1) + 1
                predictions.append(final)
                truths.append(labels)
            predictions = distributed_concat(torch.cat(predictions, dim=0), len(self.val_sampler.dataset), self.world_size)
            truths = distributed_concat(torch.cat(truths, dim=0), len(self.val_sampler.dataset), self.world_size)
            correct_num = ((truths != self.args.bin_num + 1) * (predictions == truths)).sum().item()
            val_num = (truths != self.args.bin_num + 1).sum().item()
            val_loss = get_reduced_with_fabric(running_loss / len(self.val_loader), self.fabric)
            val_acc = 100 * correct_num / val_num
            print(f'Validation Epoch {epoch} | Loss: {val_loss:.6f} | Acc: {val_acc:.4f}%')

    def train(self):
        #尽量不要在for循环内to device
        for epoch in range(1, self.args.epoch + 1):
            self.train_loader.sampler.set_epoch(epoch)
            self.model.train()
            running_loss, cum_acc = 0.0, 0.0
            for index, data in enumerate(self.train_loader, start=1):
                index += 1
                logging.info(f"Accumulated gradient on rank: {self.rank}, local_rank: {self.local_rank}")
                data, labels = data_mask(data)
                if index % self.args.grad_acc != 0:
                    with self.fabric.no_backward_sync(self.model):
                        logits = self.model(data)
                        loss = self.loss_fn(logits.transpose(1, 2), labels) / self.args.grad_acc
                        self.fabric.backward(loss)
                else:
                    logits = self.model(data)
                    loss = self.loss_fn(logits.transpose(1, 2), labels) / self.args.grad_acc
                    self.fabric.backward(loss)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1e2)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                running_loss += loss.item()
                final = self.softmax(logits)[..., 1:-1].argmax(dim=-1) + 1
                pred_num = (labels != self.args.bin_num + 1).sum(dim=-1)
                correct_num = ((labels != self.args.bin_num + 1) * (final == labels)).sum(dim=-1)
                cum_acc += torch.true_divide(correct_num, pred_num).mean().item()
            epoch_loss = get_reduced_with_fabric(running_loss / index, self.fabric)
            epoch_acc = get_reduced_with_fabric(100 * cum_acc / index, self.fabric)
            if self.is_master:
                print(f'Epoch {epoch} | Loss: {epoch_loss:.6f} | Acc: {epoch_acc:.4f}%')
            self.scheduler.step()

