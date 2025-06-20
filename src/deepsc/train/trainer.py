import logging
import os

import hydra
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import time
import wandb
from deepsc.data.dataset import data_mask, extract_rows_from_sparse_tensor
from deepsc.utils import *


# timeit decorator
def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")
        return result

    return wrapper


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

    def load_all_sparse_tensors_from_folder(self, datapath):
        tensors = []
        for file in os.listdir(datapath):
            if file.endswith(".pth"):
                path = os.path.join(datapath, file)
                if self.is_master:
                    wandb.alert(
                        title="Data Loading", text=f"Loading sparse tensor from {path}"
                    )
                tensor = torch.load(path)
                if not tensor.is_coalesced():
                    tensor = tensor.coalesce()
                tensors.append(tensor)
        if self.is_master:
            wandb.alert(
                title="Data Loading",
                text=f"Loaded {len(tensors)} sparse tensors from {datapath}",
            )
        return torch.cat(tensors, dim=0)

    @timeit
    def load_data(self):
        if os.path.isdir(self.args.data_path):
            coo_tensor = self.load_all_sparse_tensors_from_folder(self.args.data_path)
        else:
            coo_tensor = torch.load(self.args.data_path)

        row_indices = np.arange(coo_tensor.shape[0])
        # TODO: 大数据量载入的问题
        row_indices = np.arange(coo_tensor.shape[0])
        train_idx, val_idx = train_test_split(
            row_indices, test_size=0.05, random_state=self.args.seed
        )
        # extract rows from sparse tensor

        coo_tensor = coo_tensor.coalesce()
        data_train = extract_rows_from_sparse_tensor(coo_tensor, train_idx)
        data_val = extract_rows_from_sparse_tensor(coo_tensor, val_idx)
        # instantiate dataset
        self.train_dataset: Dataset = hydra.utils.instantiate(
            self.args.dataset, coo_tensor=data_train
        )
        self.val_dataset: Dataset = hydra.utils.instantiate(
            self.args.dataset, coo_tensor=data_val
        )
        # setup dataloader
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
            num_workers=32,
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            sampler=self.val_sampler,
            num_workers=8,
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
            data_iter = self.val_loader
            if self.is_master:
                data_iter = tqdm(
                    self.val_loader, desc=f"Epoch {epoch} [val]", ncols=100
                )
            for index, data in enumerate(data_iter):
                data, labels = data_mask(data)
                logits = self.model(data)
                loss = self.loss_fn(logits.transpose(1, 2), labels)
                running_loss += loss.item()
                final = self.softmax(logits)[..., 1:-1].argmax(dim=-1) + 1
                predictions.append(final)
                truths.append(labels)
                if self.is_master:
                    data_iter.set_postfix(loss=running_loss / (index + 1))
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
            if self.is_master:
                wandb.log({"val/loss": val_loss, "val/acc": val_acc, "epoch": epoch})
                wandb.alert(
                    title="Validation",
                    text=f"Validation Epoch {epoch} | Loss: {val_loss:.6f} | Acc: {val_acc:.4f}%",
                )

    def checkpoint_reload(self):
        ckpt_file = os.path.join(self.args.ckpt_dir, "latest_checkpoint.ckpt")
        self.last_epoch = 1
        self.iteration = 1
        if os.path.exists(ckpt_file):
            state = {
                "model": self.model,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "iteration": self.iteration,
                "epoch": self.last_epoch,
            }
            self.fabric.load(ckpt_file, state)
            if self.is_master:
                print(
                    f"Reloaded model, optimizer, and scheduler from {ckpt_file}, last_epoch={self.last_epoch}"
                )

    def train(self):
        # 尽量不要在for循环内to device
        self.checkpoint_reload()  # 只在训练开始时reload一次
        start_epoch = self.last_epoch if hasattr(self, "last_epoch") else 1
        if self.args.model_name == "DeepSC":
            self.model = torch.compile(self.model)
        for epoch in range(start_epoch, self.args.epoch + 1):
            self.train_loader.sampler.set_epoch(epoch)
            self.model.train()
            running_loss, cum_acc = 0.0, 0.0
            data_iter = self.train_loader
            if self.is_master:
                data_iter = tqdm(
                    self.train_loader, desc=f"Epoch {epoch} [train]", ncols=100
                )
            for index, data in enumerate(data_iter, start=1):
                data, labels = data_mask(data)
                is_accumulated = index % self.args.grad_acc != 0
                if is_accumulated:
                    with self.fabric.no_backward_sync(
                        self.model, enabled=is_accumulated
                    ):
                        logits = self.model(data)
                        loss = (
                            self.loss_fn(logits.transpose(1, 2), labels)
                            / self.args.grad_acc
                        )
                        self.fabric.backward(loss)
                else:
                    if self.is_master:
                        logging.info(
                            f"[Epoch {epoch}] Iter {index} | Loss: "
                            f"{running_loss / index:.4f} | Acc: {100 * cum_acc / index:.2f}%"
                        )
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
                if self.is_master:
                    data_iter.set_postfix(
                        loss=running_loss / index, acc=100 * cum_acc / index
                    )
                if index % self.args.save_ckpt_every == 0:
                    save_ckpt_fabric(
                        epoch,
                        self.model,
                        self.optimizer,
                        self.scheduler,
                        running_loss / index,
                        self.args.model_name,
                        self.args.ckpt_dir,
                        self.fabric,
                        iteration=index,
                    )
            epoch_loss = get_reduced_with_fabric(running_loss / index, self.fabric)
            epoch_acc = get_reduced_with_fabric(100 * cum_acc / index, self.fabric)
            if self.is_master:
                wandb.log(
                    {"train/loss": epoch_loss, "train/acc": epoch_acc, "epoch": epoch}
                )
                wandb.alert(
                    title="Training",
                    text=f"Epoch {epoch} | Loss: {epoch_loss:.6f} | Acc: {epoch_acc:.4f}%",
                )
            self.scheduler.step()
            if epoch % self.args.valid_every == 0:
                self.validate(epoch)
            save_ckpt_fabric(
                epoch,
                self.model,
                self.optimizer,
                self.scheduler,
                running_loss / index,
                self.args.model_name,
                self.args.ckpt_dir,
                self.fabric,
                iteration=index,
            )
