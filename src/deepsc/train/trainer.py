import logging

import hydra
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from deepsc.data.dataset import data_mask, extract_rows_from_sparse_tensor
from deepsc.utils import *


class Trainer:
    def __init__(self, args, fabric, model):
        self.args = args
        self.fabric = fabric
        self.model = model
        seed_all(args.seed + self.fabric.global_rank)
        self.world_size = self.fabric.world_size
        # self.device = torch.device("cuda", args.local_rank)
        self.is_master = self.fabric.global_rank == 0
        self.load_data()
        self.prepare_model()

    # 氛围
    def load_data(self):
        coo_tensor = torch.load(self.args.data_path)
        # TODO: 需支持文件夹为路径传入
        # TODO: 大数据量zai ru
        row_indices = np.arange(coo_tensor.shape[0])
        train_idx, val_idx = train_test_split(
            row_indices, test_size=0.05, random_state=self.args.seed
        )

        data_train = extract_rows_from_sparse_tensor(coo_tensor, train_idx)
        data_val = extract_rows_from_sparse_tensor(coo_tensor, val_idx)
        # instantiate dataset
        self.train_dataset: Dataset = hydra.utils.instantiate(
            self.args.dataset, coo_tensor=data_train
        )
        self.val_dataset: Dataset = hydra.utils.instantiate(
            self.args.dataset, coo_tensor=data_val
        )

        self.train_sampler = DistributedSampler(self.train_dataset)
        self.val_sampler = SequentialDistributedSampler(
            self.val_dataset,
            batch_size=self.args.batch_size,
            world_size=self.world_size,
        )

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            sampler=self.train_sampler,
        )
        val_loader = DataLoader(
            self.val_dataset, batch_size=self.args.batch_size, sampler=self.val_sampler
        )

        self.train_loader, self.val_loader = self.fabric.setup_dataloaders(
            train_loader, val_loader
        )

    def prepare_model(self):
        args = self.args
        # 是否应该让optimizer, lossfunction, scheduler customizable?
        self.optimizer = Adam(self.model.parameters(), lr=args.learning_rate)
        self.scheduler = CosineAnnealingWarmupRestarts(
            self.optimizer,
            first_cycle_steps=15,
            cycle_mult=2,
            max_lr=args.learning_rate,
            min_lr=1e-6,
            warmup_steps=5,
            gamma=0.9,
        )
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=args.num_bin + 1, reduction="mean"
        )
        self.softmax = nn.Softmax(dim=-1)
        self.model, self.optimizer = self.fabric.setup(self.model, self.optimizer)

    def validate(self, epoch):
        self.model.eval()
        running_loss, predictions, truths = 0.0, [], []
        with torch.no_grad():
            for index, data in enumerate(self.val_loader):
                data, labels = data_mask(data)
                logits = self.model(data)
                loss = self.loss_fn(logits.transpose(1, 2), labels)
                running_loss += loss.item()
                final = self.softmax(logits)[..., 1:-1].argmax(dim=-1) + 1
                predictions.append(final)
                truths.append(labels)
            predictions = distributed_concat(
                torch.cat(predictions, dim=0),
                len(self.val_sampler.dataset),
                self.world_size,
            )
            truths = distributed_concat(
                torch.cat(truths, dim=0), len(self.val_sampler.dataset), self.world_size
            )
            correct_num = (
                ((truths != self.args.num_bin + 1) * (predictions == truths))
                .sum()
                .item()
            )
            val_num = (truths != self.args.num_bin + 1).sum().item()
            val_loss = get_reduced_with_fabric(
                running_loss / len(self.val_loader), self.fabric
            )
            val_acc = 100 * correct_num / val_num
            logging.info(
                f"Validation Epoch {epoch} | Loss: {val_loss:.6f} | Acc: {val_acc:.4f}%"
            )

    def train(self):
        # 尽量不要在for循环内to device
        for epoch in range(1, self.args.epoch + 1):
            self.train_loader.sampler.set_epoch(epoch)
            self.model.train()
            running_loss, cum_acc = 0.0, 0.0
            for index, data in enumerate(self.train_loader, start=1):
                index += 1
                logging.info(
                    f"Accumulated gradient on rank:{self.fabric.global_rank} and in step {index}"
                )
                data, labels = data_mask(data)
                if index % self.args.grad_acc != 0:
                    with self.fabric.no_backward_sync(self.model):
                        logits = self.model(data)
                        loss = (
                            self.loss_fn(logits.transpose(1, 2), labels)
                            / self.args.grad_acc
                        )
                        self.fabric.backward(loss)
                else:
                    logits = self.model(data)
                    loss = (
                        self.loss_fn(logits.transpose(1, 2), labels)
                        / self.args.grad_acc
                    )
                    self.fabric.backward(loss)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1e2)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                running_loss += loss.item()
                final = self.softmax(logits)[..., 1:-1].argmax(dim=-1) + 1
                pred_num = (labels != self.args.num_bin + 1).sum(dim=-1)
                correct_num = (
                    (labels != self.args.num_bin + 1) * (final == labels)
                ).sum(dim=-1)
                cum_acc += torch.true_divide(correct_num, pred_num).mean().item()
            epoch_loss = get_reduced_with_fabric(running_loss / index, self.fabric)
            epoch_acc = get_reduced_with_fabric(100 * cum_acc / index, self.fabric)
            if self.is_master:
                logging.info(
                    f"Epoch {epoch} | Loss: {epoch_loss:.6f} | Acc: {epoch_acc:.4f}%"
                )
            self.scheduler.step()
            if epoch % self.args.valid_every == 0:
                self.validate(epoch)
            if self.is_master:
                save_ckpt(
                    epoch,
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    epoch_loss,
                    self.args.model_name,
                    self.args.ckpt_dir,
                )
