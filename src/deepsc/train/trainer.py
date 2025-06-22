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
from deepsc.data.dataset import extract_rows_from_sparse_tensor
from deepsc.data_collator import DataCollator
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
            num_workers=8,
            collate_fn=DataCollator(
                do_padding=True,
                pad_token_id=0,
                pad_value=0,
                do_mlm=True,
                do_binning=True,
                max_length=1000,
                num_genes=self.args.model.num_genes,
                num_bins=self.args.model.num_bins,
            ),
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            sampler=self.val_sampler,
            num_workers=4,
            collate_fn=DataCollator(
                do_padding=True,
                pad_token_id=0,
                pad_value=0,
                do_mlm=True,
                do_binning=True,
                max_length=1000,
                num_genes=self.args.model.num_genes,
                num_bins=self.args.model.num_bins,
            ),
        )
        self.train_loader, self.val_loader = self.fabric.setup_dataloaders(
            train_loader, val_loader
        )

    def pad_for_val(self, seq):
        max_len = max(p.size(1) for p in seq)
        padded_preds = []
        for p in seq:
            seq_len = p.size(1)
            if seq_len < max_len:
                # 在第1维末尾填充 (padding_value 可自定义)
                pad_amount = max_len - seq_len
                # F.pad 参数：(左边填充, 右边填充)，这里对 dim=1 右侧填充 pad_amount
                p = F.pad(p, (0, pad_amount), value=0)
            else:
                # 如果超过 max_len，就截断（truncate）
                p = p[:, :max_len]
            padded_preds.append(p)
        return padded_preds

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
        self.loss_fn = nn.CrossEntropyLoss(reduction="mean")
        self.softmax = nn.Softmax(dim=-1)
        self.model, self.optimizer = self.fabric.setup(self.model, self.optimizer)

    def validate(self, epoch, iteration=0):
        self.model.eval()
        running_loss, predictions, truths = 0.0, [], []
        batch_accs = []  # 新增：用于存储每个batch的mean acc
        l0_lambda = self.args.l0_lambda
        ce_loss_weight = self.args.ce_loss_weight
        mse_loss_weight = self.args.mse_loss_weight
        with torch.no_grad():
            data_iter = self.val_loader
            if self.is_master:
                data_iter = tqdm(
                    self.val_loader,
                    desc=f"Epoch {epoch} [val] Iter {iteration}",
                    ncols=100,
                )
            for index, data in enumerate(data_iter):
                gene = data["gene"]
                discrete_expr_label = data["discrete_expr_label"]
                masked_expr = data["masked_discrete_expr"]
                expression_label = data["continuous_expr_label"]
                if self.args.enable_ce and self.args.enable_mse and self.args.enable_l0:
                    logits, regression_output, y = self.model(
                        gene, masked_expr, return_mask_prob=True
                    )
                    cross_entropy_loss = self.loss_fn(
                        logits.transpose(1, 2), discrete_expr_label
                    )
                    l0_loss = (
                        y[..., 0].abs().sum() + y[..., 2].abs().sum()
                    ) / y.numel()
                    mse_loss = masked_mse_loss(
                        regression_output,
                        expression_label,
                        data["mask"],
                        reduction="mean",
                    )
                    loss = (
                        ce_loss_weight * cross_entropy_loss
                        + mse_loss_weight * mse_loss
                        + l0_lambda * l0_loss
                    )
                elif self.args.enable_ce:
                    logits, y = self.model(gene, masked_expr, return_mask_prob=True)
                    cross_entropy_loss = self.loss_fn(
                        logits.transpose(1, 2), discrete_expr_label
                    )
                    l0_loss = (
                        y[..., 0].abs().sum() + y[..., 2].abs().sum()
                    ) / y.numel()
                    loss = ce_loss_weight * cross_entropy_loss + l0_lambda * l0_loss
                running_loss += loss.item()
                final = self.softmax(logits).argmax(dim=-1)
                predictions.append(final)
                truths.append(discrete_expr_label)
                # 统计 batch mean acc
                pred_num = (discrete_expr_label != -100).sum(dim=-1)
                correct_num = (
                    (discrete_expr_label != -100) * (final == discrete_expr_label)
                ).sum(dim=-1)
                batch_acc = torch.true_divide(correct_num, pred_num).mean().item()
                batch_accs.append(batch_acc)
                if self.is_master:
                    data_iter.set_postfix(loss=running_loss / (index + 1))

            predictions = self.pad_for_val(predictions)
            truths = self.pad_for_val(truths)
            predictions = distributed_concat(
                torch.cat(predictions, dim=0),
                len(self.val_sampler.dataset),
                self.world_size,
            )

            truths = distributed_concat(
                torch.cat(truths, dim=0), len(self.val_sampler.dataset), self.world_size
            )
            # 不再用全局统计acc，改为batch mean
            val_loss = get_reduced_with_fabric(
                running_loss / len(self.val_loader), self.fabric
            )
            val_acc = 100 * np.mean(batch_accs)
            logging.info(
                f"Validation Epoch {epoch} Iter {iteration} | Loss: {val_loss:.6f} | Acc: {val_acc:.4f}%"
            )
            if self.is_master:
                wandb.log({"val/loss": val_loss, "val/acc": val_acc, "epoch": epoch})
                wandb.alert(
                    title="Validation",
                    text=f"Validation Epoch {epoch} Iter {iteration} | Loss: {val_loss:.6f} | Acc: {val_acc:.4f}%",
                )

    def checkpoint_reload(self) -> bool:
        ckpt_file = os.path.join(self.args.ckpt_dir, "latest_checkpoint.ckpt")
        if not os.path.exists(ckpt_file):
            if self.is_master:
                print(f"[WARN] 未找到检查点 {ckpt_file}")
            return False

        # 只传可 in-place 恢复的对象
        state = {
            "model": self.model,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }
        remainder = self.fabric.load(ckpt_file, state)  # ← 其余条目会返回

        # 手动恢复不可变计数器
        self.iteration = remainder.get("iteration", 1)
        self.last_epoch = remainder.get("epoch", 1)

        if self.is_master:
            print(
                f"[INFO] 成功恢复到 epoch={self.last_epoch}, iter={self.iteration} "
                f"from {ckpt_file}"
            )
        return True

    def train(self):
        # 尽量不要在for循环内to device
        # self.checkpoint_reload()  # 只在训练开始时reload一次
        start_epoch = self.last_epoch if hasattr(self, "last_epoch") else 1
        if self.args.model_name == "DeepSC":
            self.model = torch.compile(self.model)
        l0_lambda = self.args.l0_lambda
        ce_loss_weight = self.args.ce_loss_weight
        mse_loss_weight = self.args.mse_loss_weight
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

                # 每个data包含以下key：
                # gene: 基因id
                # masked_discrete_expr: 离散表达值掩码 (输入模型)
                # masked_continuous_expr: 连续表达值掩码 (输入模型)
                # discrete_expr_label: 离散表达值label （和模型的输出比较）(添加了-100,表明这些位置不参加cross entropy loss)
                # continuous_expr_label: 连续表达值label （和模型的输出比较）（未添加-100）
                # mask: 掩码的位置 （用于计算MSE的label，continuous_expr_label使用，在masked_mse函数中使用）

                # 跳过已完成的iteration
                if epoch == start_epoch and index < getattr(self, "iteration", 1):
                    continue
                gene = data["gene"]
                discrete_expr_label = data["discrete_expr_label"]
                continuous_expr_label = data["continuous_expr_label"]
                masked_expr = data["masked_discrete_expr"]
                is_accumulated = index % self.args.grad_acc != 0
                if is_accumulated:
                    with self.fabric.no_backward_sync(
                        self.model, enabled=is_accumulated
                    ):
                        if (
                            self.args.enable_ce
                            and self.args.enable_mse
                            and self.args.enable_l0
                        ):
                            logits, regression_output, y = self.model(
                                gene, masked_expr, return_mask_prob=True
                            )
                            cross_entropy_loss = self.loss_fn(
                                logits.transpose(1, 2), discrete_expr_label
                            )
                            l0_loss = (
                                y[..., 0].abs().sum() + y[..., 2].abs().sum()
                            ) / y.numel()
                            mse_loss = masked_mse_loss(
                                regression_output,
                                continuous_expr_label,
                                data["mask"],
                                reduction="mean",
                            )
                            loss = (
                                ce_loss_weight * cross_entropy_loss
                                + mse_loss_weight * mse_loss
                                + l0_lambda * l0_loss
                            )
                            self.fabric.backward(loss / self.args.grad_acc)
                        elif self.args.enable_ce:
                            logits, y = self.model(
                                gene, masked_expr, return_mask_prob=True
                            )
                            cross_entropy_loss = self.loss_fn(
                                logits.transpose(1, 2), discrete_expr_label
                            )
                            l0_loss = (
                                y[..., 0].abs().sum() + y[..., 2].abs().sum()
                            ) / y.numel()
                            loss = (
                                ce_loss_weight * cross_entropy_loss
                                + l0_lambda * l0_loss
                            )
                            self.fabric.backward(loss / self.args.grad_acc)
                else:
                    if self.is_master:
                        logging.info(
                            f"[Epoch {epoch}] Iter {index} | Loss: "
                            f"{running_loss / index:.4f} | Acc: {100 * cum_acc / index:.2f}%"
                        )
                    if (
                        self.args.enable_ce
                        and self.args.enable_mse
                        and self.args.enable_l0
                    ):
                        logits, regression_output, y = self.model(
                            gene, masked_expr, return_mask_prob=True
                        )
                        cross_entropy_loss = self.loss_fn(
                            logits.transpose(1, 2), discrete_expr_label
                        )
                        l0_loss = (
                            y[..., 0].abs().sum() + y[..., 2].abs().sum()
                        ) / y.numel()
                        mse_loss = masked_mse_loss(
                            regression_output,
                            continuous_expr_label,
                            data["mask"],
                            reduction="mean",
                        )
                        loss = (
                            ce_loss_weight * cross_entropy_loss
                            + mse_loss_weight * mse_loss
                            + l0_lambda * l0_loss
                        )
                        self.fabric.backward(loss / self.args.grad_acc)
                    elif self.args.enable_ce:
                        logits, y = self.model(gene, masked_expr, return_mask_prob=True)
                        cross_entropy_loss = self.loss_fn(
                            logits.transpose(1, 2), discrete_expr_label
                        )
                        l0_loss = (
                            y[..., 0].abs().sum() + y[..., 2].abs().sum()
                        ) / y.numel()
                        loss = ce_loss_weight * cross_entropy_loss + l0_lambda * l0_loss
                        self.fabric.backward(loss / self.args.grad_acc)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1e2)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                running_loss += loss.item()
                final = self.softmax(logits).argmax(dim=-1)
                pred_num = (discrete_expr_label != -100).sum(dim=-1)
                correct_num = (
                    (discrete_expr_label != -100) * (final == discrete_expr_label)
                ).sum(dim=-1)
                cum_acc += torch.true_divide(correct_num, pred_num).mean().item()
                if self.is_master:
                    data_iter.set_postfix(
                        loss=running_loss / index, acc=100 * cum_acc / index
                    )
                self._log_stats(
                    final, discrete_expr_label, epoch, index, print_pred_dist=True
                )
                if index % self.args.save_ckpt_every == 0:
                    if self.is_master:
                        wandb.log(
                            {
                                "train/loss": running_loss / index,
                                "train/acc": 100 * cum_acc / index,
                                "epoch": epoch,
                                "iteration": index,
                            }
                        )
                    self._log_stats(
                        final,
                        discrete_expr_label,
                        epoch,
                        index,
                        print_detailed_stats=True,
                    )
                    self.validate(epoch, index)
                    self.model.train()
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
            # 每个epoch结束后重置iteration
            self.iteration = 1
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

    def _log_stats(
        self,
        final,
        labels,
        epoch,
        index,
        print_detailed_stats=False,
        print_pred_dist=False,
    ):
        if not self.is_master:
            return

        non_padded_mask = labels != -100
        valid_labels = labels[non_padded_mask]
        valid_preds = final[non_padded_mask]
        non_padded_count = valid_preds.numel()
        num_classes = self.args.model.num_bins + 1

        if non_padded_count == 0:
            return

        if print_pred_dist:
            pred_counts = torch.bincount(
                valid_preds.to(torch.int), minlength=num_classes
            )
            pred_dist = pred_counts.float() / non_padded_count
            print(f"--- Epoch {epoch} Iteration {index} Prediction Distribution ---")
            for i, p in enumerate(pred_dist):
                if p > 0:
                    print(f"  Bin {i}: {p.item():.2%}")
            print("---------------------------------------------")

        if print_detailed_stats:
            # 1. Label distribution
            label_counts = torch.bincount(
                valid_labels.to(torch.int), minlength=num_classes
            )
            label_dist = label_counts.float() / non_padded_count

            # 2. Prediction distribution
            pred_counts = torch.bincount(
                valid_preds.to(torch.int), minlength=num_classes
            )
            pred_dist = pred_counts.float() / non_padded_count

            # 5. Correct prediction distribution
            correct_mask = valid_preds == valid_labels
            correct_preds = valid_preds[correct_mask]
            num_correct = correct_preds.numel()
            if num_correct > 0:
                correct_pred_counts = torch.bincount(
                    correct_preds.to(torch.int), minlength=num_classes
                )
                correct_pred_dist = correct_pred_counts.float() / num_correct
            else:
                correct_pred_dist = torch.zeros(num_classes, device=labels.device)

            # 6. Incorrect prediction distribution
            incorrect_mask = ~correct_mask
            incorrect_preds = valid_preds[incorrect_mask]
            num_incorrect = incorrect_preds.numel()
            if num_incorrect > 0:
                incorrect_pred_counts = torch.bincount(
                    incorrect_preds.to(torch.int), minlength=num_classes
                )
                incorrect_pred_dist = incorrect_pred_counts.float() / num_incorrect
            else:
                incorrect_pred_dist = torch.zeros(num_classes, device=labels.device)

            # 7. Labels for incorrect predictions
            incorrect_labels = valid_labels[incorrect_mask]
            if num_incorrect > 0:
                incorrect_label_counts = torch.bincount(
                    incorrect_labels.to(torch.int), minlength=num_classes
                )
                incorrect_label_dist = incorrect_label_counts.float() / num_incorrect
            else:
                incorrect_label_dist = torch.zeros(num_classes, device=labels.device)

            def print_dist(dist_tensor, name):
                print(f"--- {name} Distribution ---")
                for i, p in enumerate(dist_tensor):
                    if p > 0:
                        print(f"  Bin {i}: {p:.2%}")

            print(f"\n===== Epoch {epoch} Iteration {index} Stats =====")
            # 3 & 4. Total non-padded count
            print(f"Total non-padded labels (predictions): {non_padded_count}")

            # 1. Label distribution
            print_dist(label_dist, "Label")
            # 2. Prediction distribution
            print_dist(pred_dist, "Prediction")
            # 5. Correct prediction distribution
            print(f"Total correct predictions: {num_correct}")
            print_dist(correct_pred_dist, "Correct Predictions")
            # 6. Incorrect prediction distribution
            print(f"Total incorrect predictions: {num_incorrect}")
            print_dist(incorrect_pred_dist, "Incorrect Predictions")
            # 7. Labels for incorrect predictions
            print_dist(incorrect_label_dist, "Labels of Incorrect Predictions")
            print("================================\n")
