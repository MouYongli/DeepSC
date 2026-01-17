import logging
import math
import os

import hydra
import numpy as np
import scipy.sparse
import torch
from omegaconf import DictConfig
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

import wandb
from deepsc.data import DataCollator
from deepsc.data.dataset import GeneExpressionDataset
from deepsc.losses.losses import LossCalculator
from deepsc.utils import (
    CosineAnnealingWarmRestartsWithDecayAndLinearWarmup,
    CosineAnnealingWarmupRestarts,
    check_moe_collapse,
    compute_bin_distribution,
    compute_classification_metrics,
    compute_M_from_y,
    distributed_concat,
    draw_continuous_pred_label_scatter,
    draw_expr_emb_analysis,
    get_reduced_with_fabric,
    load_checkpoint,
    log_stats,
    print_m_matrix,
    restore_wandb_session,
    save_ckpt_fabric,
    seed_all,
)


class Trainer:
    def __init__(self, args: DictConfig, fabric, model):
        self.num_files = len(
            [
                f
                for f in os.listdir(args.data.data_path)
                if os.path.isfile(os.path.join(args.data.data_path, f))
            ]
        )
        self.args = args
        self.fabric = fabric
        self.model = model
        self.last_iteration = 0
        self.last_chunk_idx = 0  # Used to record last processed chunk index
        seed_all(args.seed + self.fabric.global_rank)
        self.world_size = self.fabric.world_size
        self.is_master = self.fabric.global_rank == 0
        self.data_is_directory = os.path.isdir(self.args.data.data_path)
        self.all_files = None  # List of all .npz files in directory
        self.file_chunks = None  # List of file subsets split by chunk_size
        self.chunk_size = self.args.data.chunk_size
        self.class_counts = None
        self.dynamic_mask_probabilities = None
        self.prepare_model()
        self.scheduler = self.create_scheduler(self.optimizer, self.args)

        # Initialize loss calculator
        self.loss_calculator = None

    def _load_all_csr_from_files(self, files):
        matrices = []
        for file in files:
            m = scipy.sparse.load_npz(file)
            logging.info(f"Loaded {file} with shape {m.shape}")
            matrices.append(m)
        if not matrices:
            raise ValueError(f"No .npz files found in {files}")
        return scipy.sparse.vstack(matrices)

    def _build_datasets_from_files(self, files_subset):
        if isinstance(files_subset, (str, os.PathLike)):
            files_subset = [str(files_subset)]
        files_subset = list(files_subset)

        if len(files_subset) == 1 and not self.data_is_directory:
            # 单文件路径：兼容旧逻辑
            csr_matrix = scipy.sparse.load_npz(files_subset[0])
        else:
            csr_matrix = self._load_all_csr_from_files(files_subset)
        row_indices = np.arange(csr_matrix.shape[0])
        logging.info(
            f"Loaded CSR matrix with shape {csr_matrix.shape} from {len(files_subset)} files."
        )
        train_idx, val_idx = train_test_split(
            row_indices, test_size=0.05, random_state=self.args.seed
        )
        train_csr = csr_matrix[train_idx]
        val_csr = csr_matrix[val_idx]
        self.train_dataset: Dataset = GeneExpressionDataset(csr_matrix=train_csr)
        self.val_dataset: Dataset = GeneExpressionDataset(csr_matrix=val_csr)
        self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
        self.val_sampler = DistributedSampler(self.val_dataset, shuffle=True)
        # 计算动态掩码概率（使用已缓存的class_counts）

    def _prepare_file_plan(self):
        """When data_path is a directory, prepare the full file list and split by chunk order
        (no shuffling, convenient for checkpoint recovery)."""
        if not self.data_is_directory:
            # Single file: one chunk
            self.all_files = [self.args.data.data_path]
            self.file_chunks = [self.all_files]
            return

        # Collect all .npz files
        all_files = []
        for fn in os.listdir(self.args.data.data_path):
            if fn.endswith(".npz"):
                all_files.append(os.path.join(self.args.data.data_path, fn))
        if not all_files:
            raise ValueError(
                f"No .npz files found in directory: {self.args.data.data_path}"
            )
        import re

        def _nat_key(s: str):
            return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

        all_files = sorted(all_files, key=_nat_key)

        # Split sequentially into consecutive chunks
        chunks = []
        for i in range(0, len(all_files), self.chunk_size):
            chunks.append(all_files[i : i + self.chunk_size])

        self.all_files = all_files
        self.file_chunks = chunks

    def load_data(self):
        self.args.data.DataCollator.dynamic_mask_probabilities = (
            self.dynamic_mask_probabilities
        )
        logging.info("Dynamic mask probabilities in train:")
        logging.info(self.args.data.DataCollator.dynamic_mask_probabilities)
        data_collator: DataCollator = hydra.utils.instantiate(
            self.args.data.DataCollator,
        )
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            sampler=self.train_sampler,
            num_workers=8,
            collate_fn=data_collator,
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            sampler=self.val_sampler,
            num_workers=4,
            collate_fn=data_collator,
        )
        self.train_loader, self.val_loader = self.fabric.setup_dataloaders(
            train_loader, val_loader
        )

    # Generic padding helper: pad/truncate tensors along a given dimension to a common length
    def pad_list(self, tensors, dim=1, pad_value=0):
        if tensors is None or len(tensors) == 0:
            return tensors
        max_len = max(t.shape[dim] for t in tensors)
        padded = []
        for t in tensors:
            target_shape = list(t.shape)
            target_shape[dim] = max_len
            out = t.new_full(target_shape, pad_value)
            copy_len = min(t.shape[dim], max_len)
            index = [slice(None)] * t.ndim
            index[dim] = slice(0, copy_len)
            out[tuple(index)] = (
                t[tuple(index)] if copy_len == t.shape[dim] else t[tuple(index)]
            )
            padded.append(out)
        return padded

    def prepare_model(self):
        args = self.args
        self.optimizer = Adam(self.model.parameters(), lr=args.learning_rate)
        self.softmax = nn.Softmax(dim=-1)
        if self.args.experiment.use_compile:
            self.model = torch.compile(self.model)  # Before setup
        self.model, self.optimizer = self.fabric.setup(self.model, self.optimizer)

    def create_scheduler(self, optimizer, args):

        total_steps = args.epoch * math.ceil(
            (self.num_files * args.data.dataset_size)
            / (args.batch_size * self.world_size * args.grad_acc)
        )
        warmup_ratio = self.args.scheduler.warmup_ratio
        warmup_steps = math.ceil(total_steps * warmup_ratio)
        if self.args.scheduler.use_scbert_scheduler:
            scheduler = CosineAnnealingWarmupRestarts(
                optimizer,
                first_cycle_steps=warmup_steps * 3,
                cycle_mult=1,
                max_lr=self.args.learning_rate,
                min_lr=5e-6,
                warmup_steps=warmup_steps,
                gamma=0.9,
            )
            return scheduler
        elif self.args.scheduler.use_mogaide_scheduler:
            linear_warmup = LinearLR(
                optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
            )
            cosine_anneal = CosineAnnealingWarmRestartsWithDecayAndLinearWarmup(
                optimizer,
                T_0=warmup_steps * 3,
                T_mult=1,
                warmup_steps=0,
                decay=0.9,
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[linear_warmup, cosine_anneal],
                milestones=[warmup_steps],
            )
        elif self.args.scheduler.use_warmup:
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

    def _process_batch(self, data, is_val=False):
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
        if self.args.loss.enable_ce and self.args.loss.enable_mse:
            logits, regression_output, _, expr_emb = self.model(
                gene,
                masked_discrete_expr,
                masked_continuous_expr,
                return_mask_prob=True,
            )
        elif self.args.loss.enable_ce:
            logits, _, expr_emb = self.model(
                gene,
                masked_discrete_expr,
                masked_continuous_expr,
                return_mask_prob=True,
            )
        elif self.args.loss.enable_mse:
            regression_output, _, expr_emb = self.model(
                gene,
                masked_discrete_expr,
                masked_continuous_expr,
                return_mask_prob=True,
            )
        loss, ce_loss, mse_loss, l0_loss = self.loss_calculator.calculate_loss(
            logits=logits,
            discrete_expr_label=discrete_expr_label,
            regression_output=regression_output,
            continuous_expr_label=continuous_expr_label,
            mask=mask,
            is_val=is_val,
        )
        if logits is not None:
            probs = self.softmax(logits)  # (B, T, C)
            final = probs.argmax(dim=-1)  # Classification prediction result
            if is_val:
                # Accumulate current batch's softmax probabilities and token count
                self.softmax_prob_sum += probs.sum(dim=(0, 1))  # (num_bins+1,)
                self.softmax_total_count += probs.shape[0] * probs.shape[1]

        else:
            final = None
        masked_preds = torch.tensor(
            []
        )  # Can add device and dtype to ensure compatibility
        masked_labels = torch.tensor([])
        if is_val:
            if self.args.loss.enable_mse:
                masked_preds = regression_output[mask].detach().cpu()
                masked_labels = continuous_expr_label[mask].detach().cpu()
            return (
                loss,
                final,
                mse_loss,
                ce_loss,
                l0_loss,
                masked_preds,
                masked_labels,
                expr_emb,
                y,
            )
        return loss, final, mse_loss, ce_loss, l0_loss, y

    def validate(self, epoch, iteration=0):
        self.softmax_prob_sum = torch.zeros(
            self.args.model.num_bins + 1, device=self.fabric.device
        )
        self.softmax_total_count = 0
        self.model.eval()
        predictions, truths = [], []
        all_expr_embs = []
        all_masked_preds = []  # New: collect all masked predictions
        all_masked_labels = []  # New: collect all masked labels
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
            accm_per_bin_recall = []
            accm_total_acc = []
            for index, data in enumerate(data_iter):
                # --------- Original loss/acc calculation ---------
                (
                    loss,
                    final,
                    mse_loss,
                    ce_loss,
                    l0_loss,
                    masked_preds,
                    masked_labels,
                    expr_emb,
                    y,
                ) = self._process_batch(data, is_val=True)
                all_expr_embs.append(expr_emb.cpu())
                discrete_expr_label = data["discrete_expr_label"]
                accm_loss.append(loss.item())
                accm_ce_loss.append(ce_loss.item())
                accm_l0_loss.append(l0_loss.item())
                accm_mse_loss.append(mse_loss.item())
                if final is not None:
                    predictions.append(final)
                    truths.append(discrete_expr_label)
                    per_bin_recall = self._calculate_per_bin_recall(
                        final, discrete_expr_label, self.args.model.num_bins
                    )
                    # Calculate overall accuracy
                    total_acc = self._calculate_accuracy(final, discrete_expr_label)
                    accm_per_bin_recall.append(per_bin_recall)
                    accm_total_acc.append(total_acc)
                else:
                    predictions.append(torch.tensor([]))
                    truths.append(torch.tensor([]))
                    accm_per_bin_recall.append(0.0)
                    accm_total_acc.append(0.0)
                if self.is_master:
                    data_iter.set_postfix(
                        loss=sum(accm_loss) / len(accm_loss),
                        mse_loss=sum(accm_mse_loss) / len(accm_mse_loss),
                        ce_loss=sum(accm_ce_loss) / len(accm_ce_loss),
                        l0_loss=sum(accm_l0_loss) / len(accm_l0_loss),
                        per_bin_recall=sum(accm_per_bin_recall)
                        / len(accm_per_bin_recall),
                        total_acc=sum(accm_total_acc) / len(accm_total_acc),
                    )
                if self.args.loss.enable_mse and mse_loss is not None:
                    all_masked_preds.append(masked_preds)
                    all_masked_labels.append(masked_labels)
            # Only perform pad and evaluation when there are classification tasks
            if len(predictions) > 0:
                # Filter out any empty tensors produced when final is None
                predictions = [
                    t
                    for t in predictions
                    if isinstance(t, torch.Tensor) and t.numel() > 0
                ]
                truths = [
                    t for t in truths if isinstance(t, torch.Tensor) and t.numel() > 0
                ]
                if len(predictions) > 0 and len(truths) > 0:
                    predictions = self.pad_list(predictions, dim=1, pad_value=-100)
                    truths = self.pad_list(truths, dim=1, pad_value=-100)
                else:
                    predictions, truths = [], []
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
            all_expr_embs = self.pad_list(all_expr_embs, dim=1, pad_value=0)
            all_expr_embs = torch.cat(all_expr_embs, dim=0)  # (total_batch, g, d)
            E = all_expr_embs.reshape(-1, all_expr_embs.shape[-1])  # (N, d)
            # Sample at most 10000
            max_samples = 10000
            if E.shape[0] > max_samples:
                idx = torch.randperm(E.shape[0])[:max_samples]
                E = E[idx]
            rank = torch.linalg.matrix_rank(E)
            logging.info(f"[Embedding Analysis] expr_emb rank: {rank.item()}")
            U, S, Vh = torch.linalg.svd(E)
            logging.info(
                f"[Embedding Analysis] Top 10 singular values: {S[:10].cpu().numpy()}"
            )
            if self.args.logging.plot_tsne_and_umap and self.is_master:
                draw_expr_emb_analysis(
                    E,
                    epoch,
                    self.args.system.ckpt_dir,
                    iteration,
                )
            if (
                self.args.loss.enable_mse
                and self.args.logging.draw_continuous_pred_label_scatter
                and self.is_master
            ):
                draw_continuous_pred_label_scatter(
                    all_masked_preds,
                    all_masked_labels,
                    epoch,
                    self.args.system.ckpt_dir,
                    iteration,
                )

            val_loss = get_reduced_with_fabric(
                sum(accm_loss) / len(accm_loss), self.fabric
            )
            val_mse_loss = get_reduced_with_fabric(
                sum(accm_mse_loss) / len(accm_mse_loss), self.fabric
            )
            if self.args.loss.enable_ce:
                val_per_bin_recall = get_reduced_with_fabric(
                    100 * sum(accm_per_bin_recall) / len(accm_per_bin_recall),
                    self.fabric,
                )
                val_total_acc = get_reduced_with_fabric(
                    100 * sum(accm_total_acc) / len(accm_total_acc),
                    self.fabric,
                )
                val_ce_loss = get_reduced_with_fabric(
                    sum(accm_ce_loss) / len(accm_ce_loss), self.fabric
                )
            else:
                val_per_bin_recall = 0.0
                val_total_acc = 0.0
                val_ce_loss = 0.0
            val_l0_loss = get_reduced_with_fabric(
                sum(accm_l0_loss) / len(accm_l0_loss), self.fabric
            )
            if self.is_master:
                logging.info(
                    "Val E%d I%d | L:%.4f | Acc:%.2f%% | BinRecall:%.2f%%",
                    epoch,
                    iteration,
                    val_loss,
                    val_total_acc,
                    val_per_bin_recall,
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
                        "val/huber_loss": val_mse_loss,
                        "val/per_bin_recall": val_per_bin_recall,
                        "val/total_acc": val_total_acc,
                        "val/ce_loss": val_ce_loss,
                        "val/l0_loss": val_l0_loss,
                        "epoch": epoch,
                        "val/expr_emb_rank": rank.item(),
                    }
                )
        avg_probs = self.softmax_prob_sum / self.softmax_total_count
        logging.info("\n[VAL] Average Softmax probabilities per bin:")
        for i, p in enumerate(avg_probs):
            logging.info(f"  Bin {i}: {p.item():.4f}")
        del self.softmax_prob_sum
        del self.softmax_total_count

        # Clear memory
        if all_expr_embs is not None:
            del all_expr_embs
        if all_masked_preds is not None and len(all_masked_preds) > 0:
            del all_masked_preds
        if all_masked_labels is not None and len(all_masked_labels) > 0:
            del all_masked_labels
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def train(self):
        # Handle wandb initialization first - decide whether to restore or create new based on checkpoint situation
        if self.args.resume_last_training:
            if self.is_master:
                checkpoint_loaded = self.checkpoint_reload()
                if not checkpoint_loaded:
                    # No checkpoint or loading failed, create new wandb run
                    logging.info("No checkpoint found, initializing new wandb run...")
                    self.init_wandb()
            else:
                # 非master进程只需要尝试加载checkpoint
                self.checkpoint_reload()
        else:
            # resume_last_training = False，直接新建wandb run
            if self.is_master:
                logging.info(
                    "resume_last_training=False, initializing new wandb run..."
                )
                self.init_wandb()
        # if self.args.model_name == "DeepSC":
        # self.model = torch.compile(self.model)
        start_epoch = self.last_epoch if hasattr(self, "last_epoch") else 1
        for epoch in range(start_epoch, self.args.epoch + 1):
            self._prepare_file_plan()
            # 确定本epoch从哪个chunk开始（仅当从checkpoint恢复且仍在同一epoch时跳过已完成的chunk）
            start_chunk_idx = (
                self.last_chunk_idx if epoch == getattr(self, "last_epoch", 1) else 0
            )
            # Mark: whether scheduler has been created; whether class_count has been calculated
            did_compute_class_counts = self.class_counts is not None
            chunk_total = len(self.file_chunks)
            print(f"chunk_total: {chunk_total}")
            print(f"self.file_chunks: {self.file_chunks}")
            chunk_bar = tqdm(
                total=chunk_total,
                initial=start_chunk_idx,  # Display immediately and set progress to recovery point
                desc="Chunks",
                position=0,
                leave=True,
                dynamic_ncols=True,
                disable=not self.is_master,  # Only display on master
            )
            if start_chunk_idx > 0:
                chunk_bar.update(start_chunk_idx)
            for chunk_idx, files_subset in enumerate(self.file_chunks):
                if chunk_idx < start_chunk_idx:
                    # Skip completed chunks
                    logging.info("Skipping chunk %d (already processed)", chunk_idx)
                    continue
                self.current_chunk_idx = (
                    chunk_idx  # Update current processing chunk index
                )
                self._build_datasets_from_files(files_subset)
                if not did_compute_class_counts and (
                    self.args.data.enable_data_augmentation
                    or self.args.loss.use_ldam_loss
                    or self.args.loss.enable_alternating_ldam_mean_ce_loss
                ):
                    self.class_counts = self.calculate_class_counts()
                    self.init_loss_fn()
                    self.dynamic_mask_probabilities = (
                        self.set_dynamic_mask_probabilities()
                        if self.args.data.enable_data_augmentation
                        else None
                    )
                    did_compute_class_counts = True
                elif not did_compute_class_counts:
                    self.init_loss_fn()
                    did_compute_class_counts = True
                self.load_data()
                self.train_loader.sampler.set_epoch(epoch)
                self.model.train()
                data_iter = self.train_loader
                if self.is_master:
                    data_iter = tqdm(
                        self.train_loader,
                        desc=f"Epoch {epoch} [train] {self.current_chunk_idx+1}/{chunk_total} Chunks",
                        ncols=300,
                        position=1,
                    )

                accm_loss, accm_ce_loss, accm_l0_loss, accm_mse_loss = [], [], [], []
                accm_per_bin_recall, accm_total_acc = [], []
                average_loss = 0.0

                for index, data in enumerate(data_iter):
                    if index < self.last_iteration:
                        continue
                    # Update loss calculator progress
                    if self.loss_calculator is not None:
                        self.loss_calculator.epoch = epoch

                    loss, final, mse_loss, ce_loss, l0_loss, y = self._process_batch(
                        data
                    )

                    # Print M matrix every 10 iterations
                    if (
                        self.is_master
                        and index % self.args.logging.log_m_matrix_every == 0
                        and y is not None
                    ):
                        M = compute_M_from_y(y)
                        print_m_matrix(epoch, index, M)

                    discrete_expr_label = data["discrete_expr_label"]
                    per_bin_recall = self._calculate_per_bin_recall(
                        final, discrete_expr_label, self.args.model.num_bins
                    )
                    self.accumulate_or_log_classification_metrics(
                        final, discrete_expr_label, index
                    )
                    total_acc = self._calculate_accuracy(final, discrete_expr_label)
                    accm_loss.append(loss.item())
                    accm_ce_loss.append(ce_loss.item())
                    accm_l0_loss.append(l0_loss.item())
                    accm_mse_loss.append(mse_loss.item())
                    accm_per_bin_recall.append(per_bin_recall)
                    accm_total_acc.append(total_acc)
                    is_accumulating = (index + 1) % self.args.grad_acc != 0
                    if is_accumulating:
                        with self.fabric.no_backward_sync(
                            self.model, enabled=is_accumulating
                        ):
                            self.fabric.backward(loss / self.args.grad_acc)
                    else:
                        self.fabric.backward(loss / self.args.grad_acc)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1e2)
                        self.optimizer.step()
                        self.scheduler.step()  # Update learning rate after each optimizer.step()
                        self.optimizer.zero_grad()

                        average_loss = sum(accm_loss) / len(accm_loss)
                        average_ce_loss = sum(accm_ce_loss) / len(accm_ce_loss)
                        average_l0_loss = sum(accm_l0_loss) / len(accm_l0_loss)
                        average_mse_loss = sum(accm_mse_loss) / len(accm_mse_loss)
                        average_per_bin_recall = sum(accm_per_bin_recall) / len(
                            accm_per_bin_recall
                        )
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
                                per_bin_recall=average_per_bin_recall,
                                ce_loss=average_ce_loss,
                                l0_loss=average_l0_loss,
                                pred_dist=pred_dist_str,
                            )
                            wandb.log(
                                {
                                    "train/loss": average_loss,
                                    "train/mse_loss": average_mse_loss,
                                    "train/per_bin_recall": average_per_bin_recall,
                                    "train/total_acc": average_total_acc,
                                    "train/ce_loss": average_ce_loss,
                                    "train/l0_loss": average_l0_loss,
                                    "epoch": epoch,
                                    "iteration": index,
                                    "learning_rate": self.optimizer.param_groups[0][
                                        "lr"
                                    ],
                                }
                            )

                        accm_loss.clear()
                        accm_ce_loss.clear()
                        accm_l0_loss.clear()
                        accm_mse_loss.clear()
                        accm_per_bin_recall.clear()
                        accm_total_acc.clear()
                    if index != 0 and index % self.args.valid_every == 0:
                        self.validate(epoch, index)
                        self.model.train()

                    # MoE collapse detection
                    if (
                        index != 0
                        and self.args.logging.enable_moe_collapse_detection
                        and index % self.args.logging.log_moe_collapse_every == 0
                        and self.is_master
                    ):
                        check_moe_collapse(self.model, epoch, index)

                    if index != 0 and index % self.args.save_ckpt_every == 0:
                        save_ckpt_fabric(
                            epoch,
                            self.model,
                            self.optimizer,
                            self.scheduler,
                            self.args.model_name,
                            self.args.system.ckpt_dir,
                            self.fabric,
                            iteration=index + 1,
                            chunk_idx=self.current_chunk_idx,
                        )
                    self.last_iteration = 0
                chunk_bar.update(1)
                self.validate(epoch, index)
                self.model.train()
                save_ckpt_fabric(
                    epoch,
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    self.args.model_name,
                    self.args.system.ckpt_dir,
                    self.fabric,
                    iteration=0,
                    chunk_idx=self.current_chunk_idx + 1,
                )
            chunk_bar.close()
            self.last_chunk_idx = 0
            self.validate(epoch)
            save_ckpt_fabric(
                epoch + 1,
                self.model,
                self.optimizer,
                self.scheduler,
                self.args.model_name,
                self.args.system.ckpt_dir,
                self.fabric,
                iteration=0,  # Reset iteration counter
                chunk_idx=0,  # 重置chunk索引
            )

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

    def checkpoint_reload(self) -> bool:
        ckpt_file = os.path.join(self.args.system.ckpt_dir, "latest_checkpoint.ckpt")

        # Load checkpoint using the new utils function
        checkpoint_info = load_checkpoint(
            ckpt_file_path=ckpt_file,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            fabric=self.fabric,
            is_master=self.is_master,
            resume_training=self.args.resume_last_training,
        )

        if not checkpoint_info["loaded"]:
            print("No checkpoint found, initializing new wandb run...")
            return False

        # Set training state based on checkpoint
        self.last_iteration = checkpoint_info["iteration"]
        self.last_epoch = checkpoint_info["epoch"]
        self.last_chunk_idx = checkpoint_info["chunk_idx"]

        # Restore wandb session if this is the master process
        if self.is_master and checkpoint_info["wandb_run_id"]:
            wandb_restored = restore_wandb_session(
                checkpoint_info["wandb_run_id"],
                checkpoint_info["wandb_config"],
                self.args,
                self.is_master,
            )
            if not wandb_restored:
                logging.info(
                    "[INFO] wandb run_id not found or restore failed, will create new wandb run"
                )
                self.init_wandb()
        elif self.is_master:
            logging.info(
                "[INFO] wandb run_id not found in checkpoint, will create new wandb run"
            )
            self.init_wandb()

        return True

    def _calculate_per_bin_recall(self, final, discrete_expr_label, num_bins):
        """
        Calculate recall for each bin, then average over bins (excluding bin0)
        final: (batch, seq_len)
        discrete_expr_label: (batch, seq_len)
        num_bins: int
        Returns: float, average recall per bin (excluding bin0)
        """
        recalls = []
        # Exclude -100 cases
        valid_mask = discrete_expr_label != -100
        for bin_idx in range(1, num_bins + 1):  # Skip bin0
            # Only consider samples that are non -100 and label is bin_idx
            mask = (discrete_expr_label == bin_idx) & valid_mask
            total = mask.sum()
            if total == 0:
                continue  # This bin has no samples
            true_positive = ((final == bin_idx) & mask).sum()
            rec = true_positive.float() / total.float()
            recalls.append(rec)
        if len(recalls) == 0:
            return 0.0
        return torch.stack(recalls).mean().item()

    def _calculate_accuracy(self, final, discrete_expr_label):
        pred_num = (discrete_expr_label != -100).sum(dim=-1)
        correct_num = (
            (discrete_expr_label != -100) * (final == discrete_expr_label)
        ).sum(dim=-1)
        batch_acc = torch.true_divide(correct_num, pred_num).mean().item()
        return batch_acc

    def init_wandb(self):
        """
        Initialize wandb run with consistent configuration
        """
        wandb.init(
            entity=self.args.logging.wandb_team,
            project=self.args.logging.wandb_project,
            name=f"{self.args.logging.run_name}, lr: {self.args.learning_rate}",
            tags=self.args.logging.tags,
            config=dict(self.args),
        )
        logging.info(
            f"✅ Wandb initialized! Project: {wandb.run.project}, Entity: {wandb.run.entity}"
        )
        logging.info(f"🔗 Wandb URL: {wandb.run.url}")

    def accumulate_or_log_classification_metrics(
        self, final, discrete_expr_label, index
    ):
        non_padded_mask = discrete_expr_label != -100
        valid_labels = discrete_expr_label[non_padded_mask]
        valid_preds = final[non_padded_mask]
        num_classes = self.args.model.num_bins + 1
        _, _, _, macro_f1, average_recall, average_precision = (
            compute_classification_metrics(
                valid_preds, valid_labels, num_classes, discrete_expr_label.device
            )
        )
        if not hasattr(self, "acc_recall"):
            self.acc_recall = []
        if not hasattr(self, "acc_precision"):
            self.acc_precision = []
        if not hasattr(self, "acc_macro_f1"):
            self.acc_macro_f1 = []
        self.acc_recall.append(average_recall)
        self.acc_precision.append(average_precision)
        self.acc_macro_f1.append(macro_f1)
        if index % self.args.logging.log_on_wandb_every == 0:
            if self.is_master:
                average_acc_recall = sum(self.acc_recall) / len(self.acc_recall)
                average_acc_precision = sum(self.acc_precision) / len(
                    self.acc_precision
                )
                average_acc_macro_f1 = sum(self.acc_macro_f1) / len(self.acc_macro_f1)
                wandb.log(
                    {
                        "TrainClassificationMetrics/average_recall": average_acc_recall,
                        "TrainClassificationMetrics/average_precision": average_acc_precision,
                        "TrainClassificationMetrics/average_macro_f1": average_acc_macro_f1,
                    }
                )
                self.acc_recall = []
                self.acc_precision = []
                self.acc_macro_f1 = []

    def calculate_class_counts(self):
        """
        Calculate sample count for each bin, used for LDAM loss
        Returns: torch.Tensor, shape is (num_bins+1,), contains sample count for each bin
        """
        logging.info("Calculating class counts")
        data_collator: DataCollator = hydra.utils.instantiate(
            self.args.data.DataCollator,
        )
        # Initialize counter
        class_counts = torch.zeros(self.args.model.num_bins + 1, dtype=torch.long)

        # Traverse training dataset to calculate sample count for each bin
        temp_loader = DataLoader(
            self.train_dataset,
            batch_size=32,  # Use smaller batch size to save memory
            shuffle=False,
            num_workers=4,
            collate_fn=data_collator,
        )

        total_samples = 0
        for batch in tqdm(temp_loader, desc="Counting class samples"):
            # Get original data, without masking
            discrete_expr = batch[
                "masked_discrete_expr"
            ]  # This is actually original data because do_mlm=False
            # Count sample number for each bin (excluding pad_value positions)
            valid_mask = discrete_expr != 0
            for bin_idx in range(self.args.model.num_bins + 1):
                count = ((discrete_expr == bin_idx) & valid_mask).sum().item()
                class_counts[bin_idx] += count
                if bin_idx > 0:  # Only accumulate total samples for non-padding classes
                    total_samples += count

        # Print statistical information
        logging.info(f"Total valid samples (excluding padding): {total_samples}")
        logging.info("Class distribution:")
        for i, count in enumerate(class_counts):
            if i == 0:
                logging.info(
                    f"  Bin {i} (padding): {count} samples (excluded from LDAM)"
                )
            else:
                percentage = (count / total_samples * 100) if total_samples > 0 else 0
                logging.info(f"  Bin {i}: {count} samples ({percentage:.2f}%)")

        return class_counts

    def set_dynamic_mask_probabilities(self):
        """
        Dynamically set mask probabilities based on bin distribution ratio
        Returns: dict, containing mask probability for each bin
        """
        logging.info(
            "Calculating dynamic mask probabilities based on bin distribution..."
        )

        # 使用缓存的class_counts
        class_counts = self.class_counts

        # Calculate total valid samples (excluding padding)
        total_samples = class_counts[1:].sum().item()

        # Calculate ratio for each bin
        bin_ratios = {}
        for bin_idx in range(1, self.args.model.num_bins + 1):
            ratio = (
                (class_counts[bin_idx].item() / total_samples * 100)
                if total_samples > 0
                else 0
            )
            bin_ratios[bin_idx] = ratio

        # Set mask probability based on ratio
        mask_probabilities = {}
        for bin_idx in range(1, self.args.model.num_bins + 1):
            ratio = bin_ratios[bin_idx]
            if ratio < 1.0:
                # Bins with ratio less than 1% set mask probability to 0.7
                mask_probabilities[bin_idx] = 0.7
            elif ratio < 5.0:
                # Bins with ratio less than 5% set mask probability to 0.5
                mask_probabilities[bin_idx] = 0.5
            elif ratio < 12.5:
                # Bins with ratio less than 10% set mask probability to 0.3
                mask_probabilities[bin_idx] = 0.3
            elif ratio < 20.0:
                # Bins with ratio less than 20% set mask probability to 0.15
                mask_probabilities[bin_idx] = 0.15
            else:
                # Others have mask probability 0.1
                mask_probabilities[bin_idx] = 0.1

        # Print mask probability settings
        logging.info("Dynamic mask probabilities:")
        for bin_idx in range(1, self.args.model.num_bins + 1):
            ratio = bin_ratios[bin_idx]
            prob = mask_probabilities[bin_idx]
            logging.info(f"  Bin {bin_idx}: {ratio:.2f}% -> mask_prob={prob}")

        return mask_probabilities

    def init_loss_fn(self):
        """Initialize the loss calculator with current class counts and configuration"""
        self.loss_calculator: LossCalculator = hydra.utils.instantiate(
            self.args.loss,
            num_bins=self.args.model.num_bins,
            ignore_index=-100,
            class_counts=self.class_counts,
            show_mse_loss_details=self.args.logging.show_mse_loss_details,
        )
        self.loss_calculator.init_loss_fn()
