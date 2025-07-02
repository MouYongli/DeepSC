import logging
import math
import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LinearLR,
    SequentialLR,
)
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import time
import wandb
from deepsc.data import DataCollator
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

    def load_all_csr_from_folder(self, datapath):
        """
        加载文件夹内所有.npz文件，并拼接为一个csr_matrix
        """
        import scipy.sparse

        matrices = []
        for file in os.listdir(datapath):
            if file.endswith(".npz"):
                path = os.path.join(datapath, file)
                matrix = scipy.sparse.load_npz(path)
                matrices.append(matrix)
        if not matrices:
            raise ValueError(f"No .npz files found in {datapath}")
        return scipy.sparse.vstack(matrices)

    @timeit
    def load_data(self):
        # 支持单个.npz文件或目录
        import scipy.sparse

        from deepsc.data.dataset import GeneExpressionDatasetNew

        if os.path.isdir(self.args.data_path):
            csr_matrix = self.load_all_csr_from_folder(self.args.data_path)
        else:
            assert self.args.data_path.endswith(
                ".npz"
            ), "data_path必须为.npz文件或包含.npz文件的目录"
            csr_matrix = scipy.sparse.load_npz(self.args.data_path)
        row_indices = np.arange(csr_matrix.shape[0])
        train_idx, val_idx = train_test_split(
            row_indices, test_size=0.05, random_state=self.args.seed
        )
        train_csr = csr_matrix[train_idx]
        val_csr = csr_matrix[val_idx]
        self.train_dataset: Dataset = GeneExpressionDatasetNew(csr_matrix=train_csr)
        self.val_dataset: Dataset = GeneExpressionDatasetNew(csr_matrix=val_csr)
        self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
        self.val_sampler = DistributedSampler(self.val_dataset, shuffle=True)
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
                max_length=self.args.sequence_length,
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
                max_length=self.args.sequence_length,
                num_genes=self.args.model.num_genes,
                num_bins=self.args.model.num_bins,
            ),
        )
        self.train_loader, self.val_loader = self.fabric.setup_dataloaders(
            train_loader, val_loader
        )

    # for validation part, to pad the predictions to the same length. It does not
    # affect the training part. Considering moving to the utils part.
    def pad_for_val(self, seq):
        max_len = max(p.size(1) for p in seq)
        padded_preds = []
        for p in seq:
            seq_len = p.size(1)
            if seq_len < max_len:
                # 在第1维末尾填充 (padding_value 可自定义)
                pad_amount = max_len - seq_len
                # F.pad 参数：(左边填充, 右边填充)，这里对 dim=1 右侧填充 pad_amount
                p = F.pad(p, (0, pad_amount), value=-100)
            else:
                # 如果超过 max_len，就截断（truncate）
                p = p[:, :max_len]
            padded_preds.append(p)
        return padded_preds

    def pad_for_emb(self, embs):
        """
        对expr_emb list做padding，使得所有tensor在dim=1的长度一致。
        embs: List[Tensor], 每个shape为(batch, g, d)
        返回：List[Tensor]，每个shape为(batch, max_g, d)
        """
        max_len = max(e.shape[1] for e in embs)
        padded_embs = []
        for e in embs:
            seq_len = e.shape[1]
            if seq_len < max_len:
                pad_amount = max_len - seq_len
                # pad: (batch, g, d) → pad g 维右侧
                e = torch.nn.functional.pad(e, (0, 0, 0, pad_amount), value=0)
            else:
                e = e[:, :max_len, :]
            padded_embs.append(e)
        return padded_embs

    def prepare_model(self):
        args = self.args
        # 是否应该让optimizer, lossfunction, scheduler customizable?
        self.optimizer = Adam(self.model.parameters(), lr=args.learning_rate)
        self.loss_fn = nn.CrossEntropyLoss(reduction="mean")
        self.softmax = nn.Softmax(dim=-1)
        self.model, self.optimizer = self.fabric.setup(self.model, self.optimizer)
        self.scheduler = self.create_scheduler(
            self.optimizer, self.args, self.train_loader
        )

    def create_scheduler(self, optimizer, args, train_loader):

        total_steps = args.epoch * math.ceil(len(train_loader) / args.grad_acc)
        warmup_ratio = self.args.warmup_ratio
        warmup_steps = math.ceil(total_steps * warmup_ratio)
        main_steps = total_steps - warmup_steps
        if self.args.use_scbert_scheduler:
            scheduler = CosineAnnealingWarmupRestarts(
                optimizer,
                first_cycle_steps=warmup_steps * 1.5,
                cycle_mult=1.5,
                max_lr=1e-4,
                min_lr=5e-6,
                warmup_steps=warmup_steps,
                gamma=0.8,
            )
            return scheduler
        elif self.args.use_mogaide_scheduler:
            linear_warmup = LinearLR(
                optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
            )
            cosine_anneal = CosineAnnealingWarmRestartsWithDecayAndLinearWarmup(
                optimizer,
                T_0=warmup_steps * 3,
                T_mult=1,
                warmup_steps=0,
                decay=0.85,
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[linear_warmup, cosine_anneal],
                milestones=[warmup_steps],
            )
        elif self.args.use_warmup:
            linear_warmup = LinearLR(
                optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
            )
            cosine_anneal = CosineAnnealingWarmRestarts(optimizer, T_0=warmup_steps)
            scheduler = SequentialLR(
                optimizer,
                schedulers=[linear_warmup, cosine_anneal],
                milestones=[warmup_steps],
            )
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, T_mult=1)
        return scheduler

    def _process_batch(self, data):
        """
        data:
        {
            "gene": (batch_size, sequence_length)
            "masked_discrete_expr": (batch_size, sequence_length)
            "masked_continuous_expr": (batch_size, sequence_length)
            "discrete_expr_label": (batch_size, sequence_length)
            "continuous_expr_label": (batch_size, sequence_length)
        }
        In every iteration, get the gene, masked_discrete_expr,
        masked_continuous_expr, discrete_expr_label, continuous_expr_label, mask
        If enable_ce and enable_mse, get the logits, regression_output, y
        If enable_ce, get the logits, y
        If enable_mse, get the regression_output, y
        Calculate the loss
        Return the loss, final, mse_loss
        """
        gene = data["gene"]
        masked_discrete_expr = data["masked_discrete_expr"]
        masked_continuous_expr = data["masked_continuous_expr"]
        discrete_expr_label = data["discrete_expr_label"]
        continuous_expr_label = data["continuous_expr_label"]
        mask = data["mask"]
        logits = regression_output = y = None
        if self.args.enable_ce and self.args.enable_mse:
            logits, regression_output, y = self.model(
                gene,
                masked_discrete_expr,
                masked_continuous_expr,
                return_mask_prob=True,
            )
        elif self.args.enable_ce:
            logits, y = self.model(
                gene,
                masked_discrete_expr,
                masked_continuous_expr,
                return_mask_prob=True,
            )
        elif self.args.enable_mse:
            regression_output, y = self.model(
                gene,
                masked_discrete_expr,
                masked_continuous_expr,
                return_mask_prob=True,
            )
        loss, ce_loss, mse_loss, l0_loss = self.calculate_loss(
            self.args.enable_ce,
            self.args.enable_mse,
            logits=logits,
            discrete_expr_label=discrete_expr_label,
            regression_output=regression_output,
            continuous_expr_label=continuous_expr_label,
            mask=mask,
            y=y,
            ce_loss_weight=self.args.ce_loss_weight,
            mse_loss_weight=self.args.mse_loss_weight,
            l0_lambda=self.args.l0_lambda,
        )
        if logits is not None:
            final = self.softmax(logits).argmax(dim=-1)
        else:
            final = None
        return loss, final, mse_loss, ce_loss, l0_loss

    def _calculate_per_bin_accuracy(self, final, discrete_expr_label, num_bins):
        """
        计算每个bin的accuracy，然后对bin求平均（不包括bin0）
        final: (batch, seq_len)
        discrete_expr_label: (batch, seq_len)
        num_bins: int
        返回: float, 平均每个bin的accuracy（不含bin0）
        """
        accuracies = []
        for bin_idx in range(1, num_bins):  # 跳过bin0
            mask = discrete_expr_label == bin_idx
            total = mask.sum()
            if total == 0:
                continue  # 该bin没有样本
            correct = ((final == bin_idx) & mask).sum()
            acc = correct.float() / total.float()
            accuracies.append(acc)
        if len(accuracies) == 0:
            return 0.0
        return torch.stack(accuracies).mean().item()

    def _calculate_accuracy(self, final, discrete_expr_label):
        pred_num = (discrete_expr_label != -100).sum(dim=-1)
        correct_num = (
            (discrete_expr_label != -100) * (final == discrete_expr_label)
        ).sum(dim=-1)
        batch_acc = torch.true_divide(correct_num, pred_num).mean().item()
        return batch_acc

    def validate(self, epoch, iteration=0):
        self.model.eval()
        predictions, truths = [], []
        all_expr_embs = []
        with torch.no_grad():
            data_iter = self.val_loader
            if self.is_master:
                data_iter = tqdm(
                    self.val_loader,
                    desc=f"Epoch {epoch} [val] Iter {iteration}",
                    ncols=300,
                )
            accm_loss = []
            accm_ce_loss = []
            accm_l0_loss = []
            accm_mse_loss = []
            accm_per_bin_acc = []
            accm_total_acc = []
            for index, data in enumerate(data_iter):
                # --------- 新增：尝试获取expr_emb ---------
                gene = data["gene"]
                masked_discrete_expr = data["masked_discrete_expr"]
                masked_continuous_expr = data["masked_continuous_expr"]
                try:
                    _, expr_emb = self.model(
                        gene,
                        masked_discrete_expr,
                        masked_continuous_expr,
                        return_encodings=True,
                    )  # expr_emb: (batch, g, d)
                    all_expr_embs.append(expr_emb.cpu())
                except Exception as e:
                    # 某些模型不支持return_encodings，直接跳过
                    pass
                # --------- 原有loss/acc计算 ---------
                loss, final, mse_loss, ce_loss, l0_loss = self._process_batch(data)
                discrete_expr_label = data["discrete_expr_label"]
                accm_loss.append(loss.item())
                accm_ce_loss.append(ce_loss.item())
                accm_l0_loss.append(l0_loss.item())
                accm_mse_loss.append(mse_loss.item())
                if final is not None:
                    predictions.append(final)
                    truths.append(discrete_expr_label)
                    per_bin_accuracy = self._calculate_per_bin_accuracy(
                        final, discrete_expr_label, self.args.model.num_bins
                    )
                    # 计算总体accuracy
                    total_acc = self._calculate_accuracy(final, discrete_expr_label)
                    accm_per_bin_acc.append(per_bin_accuracy)
                    accm_total_acc.append(total_acc)
                if self.is_master:
                    data_iter.set_postfix(
                        loss=sum(accm_loss) / len(accm_loss),
                        mse_loss=sum(accm_mse_loss) / len(accm_mse_loss),
                        ce_loss=sum(accm_ce_loss) / len(accm_ce_loss),
                        l0_loss=sum(accm_l0_loss) / len(accm_l0_loss),
                        per_bin_acc=sum(accm_per_bin_acc) / len(accm_per_bin_acc),
                        total_acc=sum(accm_total_acc) / len(accm_total_acc),
                    )
            # 只在有分类任务时做pad和评估
            if len(predictions) > 0:
                predictions = self.pad_for_val(predictions)
                truths = self.pad_for_val(truths)
                predictions = distributed_concat(
                    torch.cat(predictions, dim=0),
                    len(self.val_sampler.dataset),
                    self.world_size,
                )
                truths = distributed_concat(
                    torch.cat(truths, dim=0),
                    len(self.val_sampler.dataset),
                    self.world_size,
                )
                log_stats(
                    self.is_master,
                    self.args.model.num_bins,
                    predictions,
                    truths,
                    epoch,
                    index=0,
                    print_detailed_stats=True,
                )

            # --------- 新增：分析expr_emb秩 ---------
            if self.is_master and len(all_expr_embs) > 0:
                # 对expr_emb做padding，保证g维一致
                all_expr_embs = self.pad_for_emb(all_expr_embs)
                all_expr_embs = torch.cat(all_expr_embs, dim=0)  # (total_batch, g, d)
                E = all_expr_embs.reshape(-1, all_expr_embs.shape[-1])  # (N, d)
                # 采样最多10000个
                max_samples = 10000
                if E.shape[0] > max_samples:
                    idx = torch.randperm(E.shape[0])[:max_samples]
                    E = E[idx]
                rank = torch.linalg.matrix_rank(E)
                print(f"[Embedding Analysis] expr_emb rank: {rank.item()}")
                U, S, Vh = torch.linalg.svd(E)
                print(
                    f"[Embedding Analysis] Top 10 singular values: {S[:10].cpu().numpy()}"
                )
                # t-SNE可视化
                try:
                    import os

                    import matplotlib.pyplot as plt
                    from sklearn.manifold import TSNE

                    tsne = TSNE(
                        n_components=2, random_state=0, perplexity=30, n_iter=1000
                    )
                    E_np = E.cpu().numpy()
                    E_tsne = tsne.fit_transform(E_np)
                    plt.figure(figsize=(6, 6))
                    plt.scatter(E_tsne[:, 0], E_tsne[:, 1], s=2, alpha=0.5)
                    plt.title(f"expr_emb t-SNE (epoch {epoch}, iteration {iteration})")
                    plt.tight_layout()
                    tsne_dir = os.path.join(self.args.ckpt_dir, "tsne_vis")
                    os.makedirs(tsne_dir, exist_ok=True)
                    plt.savefig(
                        os.path.join(
                            tsne_dir,
                            f"expr_emb_tsne_epoch{epoch}_iteration{iteration}.png",
                        )
                    )
                    tsne_path = os.path.join(
                        tsne_dir, f"expr_emb_tsne_epoch{epoch}_iteration{iteration}.png"
                    )
                    print(f"[Embedding Analysis] t-SNE plot saved:\n  {tsne_path}")
                    # 新增：将t-SNE图片上传到wandb
                    wandb.log(
                        {
                            "tsne": wandb.Image(tsne_path),
                            "epoch": epoch,
                            "iteration": iteration,
                        }
                    )
                    plt.close()

                    # 新增：UMAP可视化
                    import umap

                    reducer = umap.UMAP(n_components=2, random_state=0)
                    E_umap = reducer.fit_transform(E_np)
                    plt.figure(figsize=(6, 6))
                    plt.scatter(E_umap[:, 0], E_umap[:, 1], s=2, alpha=0.5)
                    plt.title(f"expr_emb UMAP (epoch {epoch}, iteration {iteration})")
                    plt.tight_layout()
                    umap_path = os.path.join(
                        tsne_dir, f"expr_emb_umap_epoch{epoch}_iteration{iteration}.png"
                    )
                    plt.savefig(umap_path)
                    wandb.log(
                        {
                            "umap": wandb.Image(umap_path),
                            "epoch": epoch,
                            "iteration": iteration,
                        }
                    )
                    plt.close()
                    print("[Embedding Analysis] t-SNE and UMAP plots saved")
                except Exception as e:
                    print(f"[Embedding Analysis] t-SNE failed: {e}")

            val_loss = get_reduced_with_fabric(
                sum(accm_loss) / len(accm_loss), self.fabric
            )
            val_mse_loss = get_reduced_with_fabric(
                sum(accm_mse_loss) / len(accm_mse_loss), self.fabric
            )
            val_per_bin_acc = get_reduced_with_fabric(
                100 * sum(accm_per_bin_acc) / len(accm_per_bin_acc), self.fabric
            )
            val_total_acc = get_reduced_with_fabric(
                100 * sum(accm_total_acc) / len(accm_total_acc),
                self.fabric,
            )
            val_ce_loss = get_reduced_with_fabric(
                sum(accm_ce_loss) / len(accm_ce_loss), self.fabric
            )
            val_l0_loss = get_reduced_with_fabric(
                sum(accm_l0_loss) / len(accm_l0_loss), self.fabric
            )
            if self.is_master:
                logging.info(
                    "Val E%d I%d | L:%.4f | Acc:%.2f%% | BinAcc:%.2f%%",
                    epoch,
                    iteration,
                    val_loss,
                    val_total_acc,
                    val_per_bin_acc,
                )
                logging.info(
                    "MSE:%.4f | CE:%.4f | L0:%.4f",
                    val_mse_loss,
                    val_ce_loss,
                    val_l0_loss,
                )
                wandb.log(
                    {
                        "val/loss": val_loss,
                        "val/mse_loss": val_mse_loss,
                        "val/per_bin_acc": val_per_bin_acc,
                        "val/total_acc": val_total_acc,
                        "val/ce_loss": val_ce_loss,
                        "val/l0_loss": val_l0_loss,
                        "epoch": epoch,
                        "val/expr_emb_rank": rank.item(),
                    }
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
        # 用于恢复训练,cursor可能会删掉这行，不要让它删掉
        # self.checkpoint_reload()
        if self.args.model_name == "DeepSC":
            self.model = torch.compile(self.model)
        start_epoch = self.last_epoch if hasattr(self, "last_epoch") else 1
        for epoch in range(start_epoch, self.args.epoch + 1):
            self.train_loader.sampler.set_epoch(epoch)
            self.model.train()
            data_iter = self.train_loader
            if self.is_master:
                data_iter = tqdm(
                    self.train_loader, desc=f"Epoch {epoch} [train]", ncols=300
                )

            accm_loss = []
            accm_ce_loss = []
            accm_l0_loss = []
            accm_mse_loss = []
            accm_per_bin_acc = []
            accm_total_acc = []

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
                loss, final, mse_loss, ce_loss, l0_loss = self._process_batch(data)
                discrete_expr_label = data["discrete_expr_label"]
                per_bin_accuracy = self._calculate_per_bin_accuracy(
                    final, discrete_expr_label, self.args.model.num_bins
                )
                total_acc = self._calculate_accuracy(final, discrete_expr_label)
                accm_loss.append(loss.item())
                accm_ce_loss.append(ce_loss.item())
                accm_l0_loss.append(l0_loss.item())
                accm_mse_loss.append(mse_loss.item())
                accm_per_bin_acc.append(per_bin_accuracy)
                accm_total_acc.append(total_acc)

                is_accumulating = index % self.args.grad_acc != 0
                if is_accumulating:
                    with self.fabric.no_backward_sync(
                        self.model, enabled=is_accumulating
                    ):
                        self.fabric.backward(loss / self.args.grad_acc)
                else:
                    self.fabric.backward(loss / self.args.grad_acc)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1e2)
                    self.optimizer.step()
                    self.scheduler.step()  # 每次optimizer.step()后更新学习率
                    self.optimizer.zero_grad()

                    average_loss = sum(accm_loss) / len(accm_loss)
                    average_ce_loss = sum(accm_ce_loss) / len(accm_ce_loss)
                    average_l0_loss = sum(accm_l0_loss) / len(accm_l0_loss)
                    average_mse_loss = sum(accm_mse_loss) / len(accm_mse_loss)
                    average_per_bin_acc = sum(accm_per_bin_acc) / len(accm_per_bin_acc)
                    average_total_acc = sum(accm_total_acc) / len(accm_total_acc)

                    if self.is_master:
                        num_bins = self.args.model.num_bins
                        pred_dist_str = self.get_top_bins_distribution_str(
                            final, discrete_expr_label, num_bins, topk=5
                        )
                        data_iter.set_postfix(
                            loss=average_loss,
                            mse_loss=average_mse_loss,
                            total_acc=average_total_acc,
                            per_bin_acc=average_per_bin_acc,
                            ce_loss=average_ce_loss,
                            l0_loss=average_l0_loss,
                            pred_dist=pred_dist_str,
                        )

                    accm_loss.clear()
                    accm_ce_loss.clear()
                    accm_l0_loss.clear()
                    accm_mse_loss.clear()
                    accm_per_bin_acc.clear()
                    accm_total_acc.clear()
                    if self.is_master:
                        wandb.log(
                            {
                                "train/loss": average_loss,
                                "train/mse_loss": average_mse_loss,
                                "train/per_bin_acc": average_per_bin_acc,
                                "train/total_acc": average_total_acc,
                                "train/ce_loss": average_ce_loss,
                                "train/l0_loss": average_l0_loss,
                                "epoch": epoch,
                                "iteration": index,
                                "train/learning_rate": self.optimizer.param_groups[0][
                                    "lr"
                                ],
                            }
                        )
                if index % self.args.valid_every == 0:
                    self.validate(epoch, index)
                    self.model.train()
                if index % self.args.save_ckpt_every == 0:
                    save_ckpt_fabric(
                        epoch,
                        self.model,
                        self.optimizer,
                        self.scheduler,
                        average_loss,
                        self.args.model_name,
                        self.args.ckpt_dir,
                        self.fabric,
                        iteration=index,
                    )
            # at the end of each epoch, reset the iteration
            self.iteration = 1

            # TODO: 在epoch的末尾有没有必要用这玩意？如果在每个iteration都用或者在每次log wandb都用，会不会拖慢速度？
            epoch_loss = get_reduced_with_fabric(average_loss, self.fabric)
            epoch_mse_loss = get_reduced_with_fabric(average_mse_loss, self.fabric)
            epoch_per_bin_acc = get_reduced_with_fabric(
                average_per_bin_acc, self.fabric
            )
            epoch_total_acc = get_reduced_with_fabric(
                average_total_acc,
                self.fabric,
            )
            if self.is_master:
                wandb.log(
                    {
                        "train/loss": epoch_loss,
                        "train/mse_loss": epoch_mse_loss,
                        "train/per_bin_acc": epoch_per_bin_acc,
                        "train/total_acc": epoch_total_acc,
                        "epoch": epoch,
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                    }
                )
            self.validate(epoch)
            save_ckpt_fabric(
                epoch,
                self.model,
                self.optimizer,
                self.scheduler,
                epoch_loss,
                self.args.model_name,
                self.args.ckpt_dir,
                self.fabric,
                iteration=index,
            )

    def calculate_per_bin_ce_loss(self, logits, discrete_expr_label, ignore_index=-100):
        """
        logits: (batch, seq_len, num_bins)
        discrete_expr_label: (batch, seq_len)
        返回: (num_bins,) 每个bin的平均交叉熵损失
        计算平均时不包括bin0
        """
        num_bins = self.args.model.num_bins
        ce_losses = []
        logits_flat = logits.reshape(-1, num_bins + 1)
        labels_flat = discrete_expr_label.reshape(-1)
        for i in range(1, num_bins + 1):  # 跳过bin0
            # 只统计label为i且不是ignore_index的样本
            mask = (labels_flat == i) & (labels_flat != ignore_index)
            if mask.sum() == 0:
                ce_losses.append(torch.tensor(0.0, device=logits.device))
                continue
            logits_i = logits_flat[mask]
            labels_i = labels_flat[mask]
            ce = nn.CrossEntropyLoss(reduction="mean")
            ce_loss = ce(logits_i, labels_i)
            ce_losses.append(ce_loss)
        if len(ce_losses) == 0:
            return torch.tensor(0.0, device=logits.device)
        return torch.stack(ce_losses)  # (num_bins-1,)

    def calculate_loss(
        self,
        enable_ce,
        enable_mse,
        logits=None,
        discrete_expr_label=None,
        regression_output=None,
        continuous_expr_label=None,
        mask=None,
        y=None,
        ce_loss_weight=1.0,
        mse_loss_weight=1.0,
        l0_lambda=0.0,
    ):
        total_loss = 0.0
        ce_loss = 0.0
        mse_loss = torch.tensor(
            0.0, device=logits.device if logits is not None else "cpu"
        )
        l0_loss = 0.0
        per_bin_ce_loss = None
        if enable_ce and logits is not None and discrete_expr_label is not None:
            per_bin_ce_loss = self.calculate_per_bin_ce_loss(
                logits, discrete_expr_label
            )
            ce_loss = (
                per_bin_ce_loss.mean()
                if per_bin_ce_loss is not None
                else torch.tensor(0.0, device=logits.device)
            )
            total_loss += ce_loss_weight * ce_loss
        if (
            enable_mse
            and regression_output is not None
            and continuous_expr_label is not None
            and mask is not None
        ):
            mse_loss = masked_mse_loss(
                regression_output, continuous_expr_label, mask, reduction="mean"
            )
            total_loss += mse_loss_weight * mse_loss
        l0_loss = (y[..., 0].abs().sum() + y[..., 2].abs().sum()) / y.numel()
        total_loss += l0_lambda * l0_loss
        return total_loss, ce_loss, mse_loss, l0_loss

    def get_top_bins_distribution_str(
        self, final, discrete_expr_label, num_bins, topk=5
    ):
        valid_mask = discrete_expr_label != -100
        top_bins = compute_bin_distribution(final, valid_mask, num_bins, topk=topk)
        if top_bins is not None:
            pred_dist_str = ", ".join([f"bin{idx}:{p:.2%}" for idx, p in top_bins])
        else:
            pred_dist_str = "N/A"
        return pred_dist_str
