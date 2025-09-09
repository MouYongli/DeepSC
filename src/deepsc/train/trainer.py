import logging
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
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
from src.deepsc.data import DataCollator
from src.deepsc.utils import (
    CosineAnnealingWarmRestartsWithDecayAndLinearWarmup,
    CosineAnnealingWarmupRestarts,
    LDAMLoss,
    check_grad_flow,
    check_moe_collapse,
    compute_bin_distribution,
    compute_classification_metrics,
    compute_M_from_y,
    distributed_concat,
    draw_continuous_pred_label_scatter,
    draw_expr_emb_analysis,
    get_reduced_with_fabric,
    interval_masked_mse_loss,
    log_stats,
    masked_mse_loss,
    print_m_matrix,
    save_ckpt_fabric,
    seed_all,
    weighted_masked_mse_loss,
    weighted_masked_mse_loss_v2,
)


# timeit decorator
def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"Time taken: {end_time - start_time} seconds")
        return result

    return wrapper


class Trainer:
    def __init__(self, args, fabric, model):
        self.number_of_files = len(
            [
                f
                for f in os.listdir(args.data_path)
                if os.path.isfile(os.path.join(args.data_path, f))
            ]
        )
        self.log_each = False
        self.args = args
        self.fabric = fabric
        self.model = model
        self.epoch = 1  # Add epoch class variable
        self.epoch_length = 0
        self.iteration = 0
        self.last_iteration = 0
        self.last_chunk_idx = 0  # Used to record last processed chunk index
        seed_all(args.seed + self.fabric.global_rank)
        self.world_size = self.fabric.world_size
        # self.device = torch.device("cuda", args.local_rank)
        self.is_master = self.fabric.global_rank == 0
        self.data_is_directory = os.path.isdir(self.args.data_path)
        self.all_files = None  # List of all .npz files in directory
        self.file_chunks = None  # List of file subsets split by chunk_size
        self.chunk_size = getattr(
            self.args, "chunk_size", 4
        )  # Number of files processed per training iteration
        self.shuffle_files_each_epoch = getattr(
            self.args, "shuffle_files_each_epoch", True
        )
        self.class_counts = None
        self.dynamic_mask_probabilities = None
        if self.args.adaptive_mse_weight:
            self.mse_loss_weight = 0.0
        else:
            self.mse_loss_weight = self.args.target_mse_loss_weight
        self.prepare_model()
        self.scheduler = self.create_scheduler(self.optimizer, self.args)

    # def load_all_csr_from_folder(self, datapath):
    #     """
    #     加载文件夹内所有.npz文件，并拼接为一个csr_matrix
    #     """
    #     import scipy.sparse

    #     matrices = []
    #     for file in os.listdir(datapath):
    #         if file.endswith(".npz"):
    #             path = os.path.join(datapath, file)
    #             matrix = scipy.sparse.load_npz(path)
    #             matrices.append(matrix)
    #     if not matrices:
    #         raise ValueError(f"No .npz files found in {datapath}")
    #     return scipy.sparse.vstack(matrices)
    def _load_all_csr_from_files(self, files):
        import scipy.sparse

        matrices = []
        for file in files:
            m = scipy.sparse.load_npz(file)
            logging.info(f"Loaded {file} with shape {m.shape}")
            matrices.append(m)
        if not matrices:
            raise ValueError(f"No .npz files found in {files}")
        return scipy.sparse.vstack(matrices)

    def _build_datasets_from_files(self, files_subset):
        import scipy.sparse

        from deepsc.data.dataset import GeneExpressionDatasetNew

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
        self.train_dataset: Dataset = GeneExpressionDatasetNew(csr_matrix=train_csr)
        self.val_dataset: Dataset = GeneExpressionDatasetNew(csr_matrix=val_csr)
        self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
        self.val_sampler = DistributedSampler(self.val_dataset, shuffle=True)
        # 计算动态掩码概率（使用已缓存的class_counts）

    def _prepare_file_plan(self):
        """When data_path is a directory, prepare the full file list and split by chunk order
        (no shuffling, convenient for checkpoint recovery)."""
        if not self.data_is_directory:
            # Single file: one chunk
            self.all_files = [self.args.data_path]
            self.file_chunks = [self.all_files]
            return

        # Collect all .npz files
        all_files = []
        for fn in os.listdir(self.args.data_path):
            if fn.endswith(".npz"):
                all_files.append(os.path.join(self.args.data_path, fn))
        if not all_files:
            raise ValueError(f"No .npz files found in directory: {self.args.data_path}")

        # -- Key: stable deterministic order -- #
        # To avoid order differences between different platforms/file systems,
        # use 'natural sorting' here to ensure file_2 comes before file_10
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

    def init_dataset_and_sampler(self):
        # Support single .npz file or directory
        import scipy.sparse

        from deepsc.data.dataset import GeneExpressionDatasetNew

        if os.path.isdir(self.args.data_path):
            csr_matrix = self.load_all_csr_from_folder(self.args.data_path)
        else:
            assert self.args.data_path.endswith(
                ".npz"
            ), "data_path must be a .npz file or directory containing .npz files"
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
        # 计算动态掩码概率（使用已缓存的class_counts）

    @timeit
    def load_data(self):
        dynamic_mask_probabilities = self.dynamic_mask_probabilities
        logging.info("Dynamic mask probabilities in train:")
        logging.info(dynamic_mask_probabilities)
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
                dynamic_mask_probabilities=dynamic_mask_probabilities,
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
                dynamic_mask_probabilities=dynamic_mask_probabilities,
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
                # Pad at the end of dimension 1 (padding_value can be customized)
                pad_amount = max_len - seq_len
                # F.pad parameters: (left padding, right padding), here pad pad_amount to the right of dim=1
                p = F.pad(p, (0, pad_amount), value=-100)
            else:
                # If exceeds max_len, truncate
                p = p[:, :max_len]
            padded_preds.append(p)
        return padded_preds

    def pad_for_emb(self, embs):
        """
        Apply padding to expr_emb list to make all tensors have consistent length in dim=1.
        embs: List[Tensor], each shape is (batch, g, d)
        Returns: List[Tensor], each shape is (batch, max_g, d)
        """
        max_len = max(e.shape[1] for e in embs)
        padded_embs = []
        for e in embs:
            seq_len = e.shape[1]
            if seq_len < max_len:
                pad_amount = max_len - seq_len
                # pad: (batch, g, d) → pad right side of g dimension
                e = torch.nn.functional.pad(e, (0, 0, 0, pad_amount), value=0)
            else:
                e = e[:, :max_len, :]
            padded_embs.append(e)
        return padded_embs

    def prepare_model(self):
        args = self.args
        self.optimizer = Adam(self.model.parameters(), lr=args.learning_rate)
        self.softmax = nn.Softmax(dim=-1)
        if self.args.use_compile:
            self.model = torch.compile(self.model)  # Before setup
        self.model, self.optimizer = self.fabric.setup(self.model, self.optimizer)

    def create_scheduler(self, optimizer, args):

        total_steps = args.epoch * math.ceil(
            (self.number_of_files * args.data_length)
            / (args.batch_size * self.world_size * args.grad_acc)
        )
        warmup_ratio = self.args.warmup_ratio
        warmup_steps = math.ceil(total_steps * warmup_ratio)
        main_steps = total_steps - warmup_steps
        if self.args.use_scbert_scheduler:
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
        elif self.args.use_mogaide_scheduler:
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
        if self.args.enable_ce and self.args.enable_mse:
            logits, regression_output, y, _, expr_emb = self.model(
                gene,
                masked_discrete_expr,
                masked_continuous_expr,
                return_mask_prob=True,
            )
        elif self.args.enable_ce:
            logits, y, _, expr_emb = self.model(
                gene,
                masked_discrete_expr,
                masked_continuous_expr,
                return_mask_prob=True,
            )
        elif self.args.enable_mse:
            regression_output, y, _, expr_emb = self.model(
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
            mse_loss_weight=self.mse_loss_weight,
            l0_lambda=self.args.l0_lambda,
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
        # New: interval MSE statistics
        interval_mse = None
        if self.args.enable_mse and regression_output is not None:
            interval_mse = interval_masked_mse_loss(
                regression_output, continuous_expr_label, mask
            )
        masked_preds = torch.tensor(
            []
        )  # Can add device and dtype to ensure compatibility
        masked_labels = torch.tensor([])
        if is_val:
            if self.args.enable_mse:
                masked_preds = regression_output[mask].detach().cpu()
                masked_labels = continuous_expr_label[mask].detach().cpu()
            return (
                loss,
                final,
                mse_loss,
                ce_loss,
                l0_loss,
                interval_mse,
                masked_preds,
                masked_labels,
                expr_emb,
                y,
            )
        return loss, final, mse_loss, ce_loss, l0_loss, interval_mse, y

    def validate(self, epoch, iteration=0):
        self.softmax_prob_sum = torch.zeros(
            self.args.model.num_bins + 1, device=self.fabric.device
        )
        self.softmax_total_count = 0
        self.log_each = True
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
            accm_per_bin_acc = []
            accm_total_acc = []
            accm_interval_mse = {k: [] for k in {"lt3", "3to5", "5to7", "ge7"}}
            for index, data in enumerate(data_iter):
                # --------- Original loss/acc calculation ---------
                (
                    loss,
                    final,
                    mse_loss,
                    ce_loss,
                    l0_loss,
                    interval_mse,
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
                    per_bin_accuracy = self._calculate_per_bin_accuracy(
                        final, discrete_expr_label, self.args.model.num_bins
                    )
                    # Calculate overall accuracy
                    total_acc = self._calculate_accuracy(final, discrete_expr_label)
                    accm_per_bin_acc.append(per_bin_accuracy)
                    accm_total_acc.append(total_acc)
                else:
                    predictions.append(torch.tensor([]))
                    truths.append(torch.tensor([]))
                    accm_per_bin_acc.append(0.0)
                    accm_total_acc.append(0.0)
                if self.args.enable_mse:
                    # 新增：累加区间 MSE
                    if index == 0:
                        accm_interval_mse = {
                            k: [] for k in {"lt3", "3to5", "5to7", "ge7"}
                        }
                    for k in {"lt3", "3to5", "5to7", "ge7"}:
                        accm_interval_mse[k].append(float(interval_mse[k]))
                    interval_mse_str = ", ".join(
                        [
                            f"{k}:{sum(accm_interval_mse[k])/len(accm_interval_mse[k]):.4f}"
                            for k in accm_interval_mse
                        ]
                    )
                else:
                    interval_mse_str = "N/A"
                    accm_interval_mse = {
                        "lt3": [0.0],
                        "3to5": [0.0],
                        "5to7": [0.0],
                        "ge7": [0.0],
                    }
                if self.is_master:
                    data_iter.set_postfix(
                        loss=sum(accm_loss) / len(accm_loss),
                        mse_loss=sum(accm_mse_loss) / len(accm_mse_loss),
                        ce_loss=sum(accm_ce_loss) / len(accm_ce_loss),
                        l0_loss=sum(accm_l0_loss) / len(accm_l0_loss),
                        per_bin_acc=sum(accm_per_bin_acc) / len(accm_per_bin_acc),
                        total_acc=sum(accm_total_acc) / len(accm_total_acc),
                        interval_mse=interval_mse_str,
                    )
                if self.args.enable_mse and mse_loss is not None:
                    all_masked_preds.append(masked_preds)
                    all_masked_labels.append(masked_labels)
            # Only perform pad and evaluation when there are classification tasks
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
            all_expr_embs = self.pad_for_emb(all_expr_embs)
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
            if self.args.plot_tsne_and_umap and self.is_master:
                draw_expr_emb_analysis(
                    E,
                    epoch,
                    self.args.ckpt_dir,
                    iteration,
                )
            if (
                self.args.enable_mse
                and self.args.draw_continuous_pred_label_scatter
                and self.is_master
            ):
                draw_continuous_pred_label_scatter(
                    all_masked_preds,
                    all_masked_labels,
                    epoch,
                    self.args.ckpt_dir,
                    iteration,
                )

            val_loss = get_reduced_with_fabric(
                sum(accm_loss) / len(accm_loss), self.fabric
            )
            val_mse_loss = get_reduced_with_fabric(
                sum(accm_mse_loss) / len(accm_mse_loss), self.fabric
            )
            if self.args.enable_ce:
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
            else:
                val_per_bin_acc = 0.0
                val_total_acc = 0.0
                val_ce_loss = 0.0
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
                        "ValRegression/interval_mse_5to7": sum(
                            accm_interval_mse["5to7"]
                        )
                        / len(accm_interval_mse["5to7"]),
                        "ValRegression/interval_mse_ge7": sum(accm_interval_mse["ge7"])
                        / len(accm_interval_mse["ge7"]),
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
                    wandb.init(
                        # entity=self.args.get("wandb_team", "rwth_lfb"),
                        project=self.args.get("wandb_project", "DeepSCNewProj"),
                        name=f"{self.args.run_name}, lr: {self.args.learning_rate}",
                        tags=self.args.tags,
                        config=dict(self.args),
                    )
            else:
                # 非master进程只需要尝试加载checkpoint
                self.checkpoint_reload()
        else:
            # resume_last_training = False，直接新建wandb run
            if self.is_master:
                logging.info(
                    "resume_last_training=False, initializing new wandb run..."
                )
                wandb.init(
                    entity=self.args.get("wandb_team", "rwth_lfb"),
                    project=self.args.get("wandb_project", "DeepSCNewProj"),
                    name=f"{self.args.run_name}, lr: {self.args.learning_rate}",
                    tags=self.args.tags,
                    config=dict(self.args),
                )
        self.log_each = False
        # if self.args.model_name == "DeepSC":
        # self.model = torch.compile(self.model)
        start_epoch = self.last_epoch if hasattr(self, "last_epoch") else 1
        self.epoch_length = 0
        for epoch in range(start_epoch, self.args.epoch + 1):
            self.epoch = epoch  # Update class variable epoch
            self._prepare_file_plan()
            # 确定本epoch从哪个chunk开始（仅当从checkpoint恢复且仍在同一epoch时跳过已完成的chunk）
            start_chunk_idx = (
                self.last_chunk_idx if epoch == getattr(self, "last_epoch", 1) else 0
            )
            # Mark: whether scheduler has been created; whether class_count has been calculated
            did_compute_class_counts = self.class_counts is not None
            chunk_total = len(self.file_chunks)
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
                    self.args.enable_data_augmentation
                    or self.args.use_ldam_loss
                    or self.args.enable_alternating_ldam_mean_ce_loss
                    or self.args.enable_warm_alternating_ldam_mean_ce_loss
                ):
                    self.class_counts = self.calculate_class_counts()
                    self.init_loss_fn()
                    self.dynamic_mask_probabilities = (
                        self.calculate_dynamic_mask_probabilities()
                        if self.args.enable_data_augmentation
                        else None
                    )
                    did_compute_class_counts = True
                elif not did_compute_class_counts:
                    self.init_loss_fn()
                    did_compute_class_counts = True
                self.load_data()
                self.epoch_length = len(
                    self.train_loader
                )  # Length of epoch, I'm not sure if this is correct...
                self.train_loader.sampler.set_epoch(epoch)
                self.model.train()
                data_iter = self.train_loader
                if self.is_master:
                    data_iter = tqdm(
                        self.train_loader,
                        desc=f"Epoch {epoch} [train] {self.current_chunk_idx}/{chunk_total} Chunks",
                        ncols=300,
                        position=1,
                    )

                accm_loss, accm_ce_loss, accm_l0_loss, accm_mse_loss = [], [], [], []
                accm_per_bin_acc, accm_total_acc = [], []
                interval_mse = {"lt3": 0.0, "3to5": 0.0, "5to7": 0.0, "ge7": 0.0}
                accm_interval_mse = {k: [] for k in interval_mse}
                average_loss = 0.0

                for index, data in enumerate(data_iter):
                    if index < self.last_iteration:
                        continue
                    self.iteration = index
                    loss, final, mse_loss, ce_loss, l0_loss, interval_mse, y = (
                        self._process_batch(data)
                    )

                    # Print M matrix every 10 iterations
                    if (
                        self.is_master
                        and index % self.args.log_m_matrix_every == 0
                        and y is not None
                    ):
                        M = compute_M_from_y(y)
                        print_m_matrix(epoch, index, M)

                    discrete_expr_label = data["discrete_expr_label"]
                    if interval_mse is None:
                        interval_mse = {
                            "lt3": 0.0,
                            "3to5": 0.0,
                            "5to7": 0.0,
                            "ge7": 0.0,
                        }
                    per_bin_accuracy = self._calculate_per_bin_accuracy(
                        final, discrete_expr_label, self.args.model.num_bins
                    )
                    self.accumulate_or_log_classification_metrics(
                        final, discrete_expr_label
                    )
                    total_acc = self._calculate_accuracy(final, discrete_expr_label)
                    accm_loss.append(loss.item())
                    accm_ce_loss.append(ce_loss.item())
                    accm_l0_loss.append(l0_loss.item())
                    accm_mse_loss.append(mse_loss.item())
                    accm_per_bin_acc.append(per_bin_accuracy)
                    accm_total_acc.append(total_acc)
                    # 新增：累加区间 MSE
                    if index == 1:
                        accm_interval_mse = {k: [] for k in interval_mse}
                    for k in interval_mse:
                        accm_interval_mse[k].append(float(interval_mse[k]))

                    is_accumulating = (index + 1) % self.args.grad_acc != 0
                    if is_accumulating:
                        with self.fabric.no_backward_sync(
                            self.model, enabled=is_accumulating
                        ):
                            self.fabric.backward(loss / self.args.grad_acc)
                    else:
                        # Gradient check (optional, controlled by configuration)
                        if self.args.adaptive_mse_weight:
                            self.calculate_adaptive_mse_weight()
                        # This thing seems useless, added by cursor randomly, will try again later
                        # # Add temperature annealing
                        # if hasattr(self.model, 'anneal_temperature'):
                        #     self.model.anneal_temperature(epoch, self.args.num_epochs)
                        if (
                            hasattr(self.args, "check_grad_flow")
                            and self.args.check_grad_flow
                            and index % 100 == 0
                        ):
                            logging.info(
                                f"\n[Gradient Check] Epoch {epoch}, Iteration {index}"
                            )
                            try:
                                grad_stats = check_grad_flow(
                                    self.model,
                                    loss / self.args.grad_acc,
                                    verbose=True,
                                    retain_graph=True,
                                    backward_fn=self.fabric.backward,
                                )
                                if self.is_master:
                                    wandb.log(
                                        {
                                            "grad_check/ok_params": len(
                                                grad_stats["ok"]
                                            ),
                                            "grad_check/zero_params": len(
                                                grad_stats["zero"]
                                            ),
                                            "grad_check/none_params": len(
                                                grad_stats["none"]
                                            ),
                                        }
                                    )
                            except Exception as e:
                                logging.warning(f"[WARNING] Gradient check failed: {e}")
                                if self.is_master:
                                    wandb.log(
                                        {
                                            "grad_check/error": 1,
                                        }
                                    )

                        self.fabric.backward(loss / self.args.grad_acc)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1e2)
                        self.optimizer.step()
                        self.scheduler.step()  # Update learning rate after each optimizer.step()
                        self.optimizer.zero_grad()

                        average_loss = sum(accm_loss) / len(accm_loss)
                        average_ce_loss = sum(accm_ce_loss) / len(accm_ce_loss)
                        average_l0_loss = sum(accm_l0_loss) / len(accm_l0_loss)
                        average_mse_loss = sum(accm_mse_loss) / len(accm_mse_loss)
                        average_per_bin_acc = sum(accm_per_bin_acc) / len(
                            accm_per_bin_acc
                        )
                        average_total_acc = sum(accm_total_acc) / len(accm_total_acc)
                        if self.is_master:
                            num_bins = self.args.model.num_bins
                            pred_dist_str = self.get_top_bins_distribution_str(
                                final, discrete_expr_label, num_bins, topk=5
                            )
                            # New: interval MSE display
                            interval_mse_str = ", ".join(
                                [
                                    f"{k}:{sum(accm_interval_mse[k])/len(accm_interval_mse[k]):.4f}"
                                    for k in accm_interval_mse
                                ]
                            )
                            data_iter.set_postfix(
                                loss=average_loss,
                                mse_loss=average_mse_loss,
                                total_acc=average_total_acc,
                                per_bin_acc=average_per_bin_acc,
                                ce_loss=average_ce_loss,
                                l0_loss=average_l0_loss,
                                pred_dist=pred_dist_str,
                                interval_mse=interval_mse_str,
                            )
                            wandb.log(
                                {
                                    "train/loss": average_loss,
                                    "train/mse_loss": average_mse_loss,
                                    "train/per_bin_acc": average_per_bin_acc,
                                    "train/total_acc": average_total_acc,
                                    "train/ce_loss": average_ce_loss,
                                    "train/l0_loss": average_l0_loss,
                                    "TrainRegressionMetrics/interval_mse_5to7": sum(
                                        accm_interval_mse["5to7"]
                                    )
                                    / len(accm_interval_mse["5to7"]),
                                    "TrainRegressionMetrics/interval_mse_ge7": sum(
                                        accm_interval_mse["ge7"]
                                    )
                                    / len(accm_interval_mse["ge7"]),
                                    "TrainLossWeight/ce_loss_weight": self.args.ce_loss_weight,
                                    "TrainLossWeight/mse_loss_weight": self.mse_loss_weight,
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
                        accm_per_bin_acc.clear()
                        accm_total_acc.clear()
                    if index != 0 and index % self.args.valid_every == 0:
                        self.validate(epoch, index)
                        self.model.train()

                    # MoE collapse detection
                    if (
                        index != 0
                        and hasattr(self.args, "log_moe_collapse_every")
                        and index % self.args.log_moe_collapse_every == 0
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
                            self.args.ckpt_dir,
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
                    getattr(self, "scheduler", None),
                    self.args.model_name,
                    self.args.ckpt_dir,
                    self.fabric,
                    iteration=0,
                    chunk_idx=self.current_chunk_idx + 1,
                )
                # at the end of each epoch, reset the iteration
                self.iteration = 0
            chunk_bar.close()
            self.last_chunk_idx = 0
            self.validate(epoch)
            self.log_each = False
            save_ckpt_fabric(
                epoch + 1,
                self.model,
                self.optimizer,
                self.scheduler,
                self.args.model_name,
                self.args.ckpt_dir,
                self.fabric,
                iteration=0,  # Reset iteration counter
                chunk_idx=0,  # 重置chunk索引
            )

    def calculate_per_bin_ce_loss(self, logits, discrete_expr_label, ignore_index=-100):
        """
        logits: (batch, seq_len, num_bins)
        discrete_expr_label: (batch, seq_len)
        返回: (num_bins,) 每个bin的平均交叉熵损失
        Average calculation does not include bin0
        """
        num_bins = self.args.model.num_bins
        ce_losses = []
        logits_flat = logits.reshape(-1, num_bins + 1)
        labels_flat = discrete_expr_label.reshape(-1)
        for i in range(1, num_bins + 1):  # Skip bin0
            # Only count samples where label is i and not ignore_index
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
        is_val=False,
    ):
        total_loss = 0.0
        ce_loss = torch.tensor(
            0.0, device=logits.device if logits is not None else "cpu"
        )
        regression_loss = torch.tensor(
            0.0, device=logits.device if logits is not None else "cpu"
        )
        l0_loss = 0.0
        per_bin_ce_loss = None
        if enable_ce and logits is not None and discrete_expr_label is not None:
            # Ensure label and logits are on the same device
            discrete_expr_label = discrete_expr_label.to(logits.device)
            if self.args.enable_alternating_ldam_mean_ce_loss:
                ce_loss = self.calculate_alternating_ldam_mean_ce_loss(
                    self.epoch, logits, discrete_expr_label
                )
            elif self.args.enable_adaptive_ce_loss:
                ce_loss = self.calculate_mogaide_ce_loss(logits, discrete_expr_label)
            elif self.args.enable_warm_alternating_ldam_mean_ce_loss:
                ce_loss = self.calculate_warm_alternating_ldam_mean_ce_loss(
                    self.epoch,
                    logits,
                    discrete_expr_label,
                    self.epoch_length,
                    self.iteration,
                )
            elif self.args.weighted_ce_loss:
                ce_loss = self.calculate_weighted_ce_loss(logits, discrete_expr_label)
            elif self.args.use_ldam_loss:
                ce_loss = self.calculate_ldam_ce_loss(logits, discrete_expr_label)
            elif self.args.mean_ce_loss:
                ce_loss = self.calculate_mean_ce_loss(logits, discrete_expr_label)
            else:
                ce_loss = self.calculate_ce_loss(logits, discrete_expr_label)
            total_loss += ce_loss_weight * ce_loss
        if (
            enable_mse
            and regression_output is not None
            and continuous_expr_label is not None
            and mask is not None
        ):
            if self.args.use_normal_regression_loss:
                regression_loss = masked_mse_loss(
                    regression_output,
                    continuous_expr_label,
                    mask,
                    loss_fn=self.regression_loss_fn,
                    reduction="mean",
                )
            elif self.args.use_hard_regression_loss:
                regression_loss = weighted_masked_mse_loss(
                    regression_output,
                    continuous_expr_label,
                    mask,
                    loss_fn=self.regression_loss_fn,
                    reduction="mean",
                    log_each=is_val and self.args.show_mse_loss_details,
                )
            elif self.args.use_exp_regression_loss:
                regression_loss = weighted_masked_mse_loss_v2(
                    regression_output,
                    continuous_expr_label,
                    mask,
                    loss_fn=self.regression_loss_fn,
                    reduction="mean",
                    log_each=is_val and self.args.show_mse_loss_details,
                )
            total_loss += mse_loss_weight * regression_loss
        if y is not None:
            l0_loss = (y[..., 0].abs().sum() + y[..., 2].abs().sum()) / (
                y.shape[0] * y.shape[1] * y.shape[2]
            )
            total_loss += 0.1 * l0_loss
        else:
            l0_loss = torch.tensor(0.0)  # Ensure it's a Tensor
        return total_loss, ce_loss, regression_loss, l0_loss

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
        ckpt_file = os.path.join(self.args.ckpt_dir, "latest_checkpoint.ckpt")
        if not os.path.exists(ckpt_file):
            if self.is_master:
                logging.warning(f"[WARN] Checkpoint not found: {ckpt_file}")
            return False

        # Directly read state dict
        remainder = self.fabric.load(ckpt_file)

        # Restore model, optimizer, scheduler parameters
        if "model" in remainder:
            self.model.load_state_dict(remainder["model"])
        if "optimizer" in remainder and self.optimizer is not None:
            self.optimizer.load_state_dict(remainder["optimizer"])
        if (
            "scheduler" in remainder
            and self.scheduler is not None
            and remainder["scheduler"] is not None
        ):
            self.scheduler.load_state_dict(remainder["scheduler"])

        # Restore counters
        if self.args.resume_last_training:
            self.last_iteration = remainder.get("iteration", 0)
            self.last_epoch = remainder.get("epoch", 1)
            self.last_chunk_idx = remainder.get("chunk_idx", 0)
        else:
            self.last_iteration = 0
            self.last_epoch = 1
            self.last_chunk_idx = 0

        # Restore wandb session
        if self.is_master:
            saved_run_id = remainder.get("wandb_run_id", None)
            saved_wandb_config = remainder.get("wandb_config", None)

            if saved_run_id:
                logging.info(f"[INFO] Found saved wandb run_id: {saved_run_id}")
                logging.info("[INFO] Using original run_id to restore wandb session...")
                if saved_wandb_config:
                    wandb.init(
                        id=saved_run_id,
                        resume="allow",
                        project=saved_wandb_config.get(
                            "project", self.args.get("wandb_project", "DeepSC")
                        ),
                        entity=saved_wandb_config.get(
                            "entity", self.args.get("wandb_team", "rwth_lfb")
                        ),
                        name=saved_wandb_config.get(
                            "name",
                            f"{self.args.run_name}, lr: {self.args.learning_rate}",
                        ),
                        tags=saved_wandb_config.get("tags", self.args.tags),
                        config=saved_wandb_config.get("config", dict(self.args)),
                    )
                else:
                    wandb.init(
                        id=saved_run_id,
                        resume="allow",
                        project=self.args.get("wandb_project", "DeepSC"),
                        entity=self.args.get("wandb_team", "rwth_lfb"),
                        name=f"{self.args.run_name}, lr: {self.args.learning_rate}",
                        tags=self.args.tags,
                        config=dict(self.args),
                    )
                logging.info(f"[INFO] Restored wandb session, run_id: {wandb.run.id}")
                logging.info(
                    f"[INFO] Project: {wandb.run.project}, Name: {wandb.run.name}"
                )
            else:
                logging.info(
                    "[INFO] wandb run_id not found in checkpoint, will create new wandb run"
                )

        if self.is_master:
            logging.info(
                f"[INFO] reload epoch={self.last_epoch}, chunk={self.last_chunk_idx}, iter={self.last_iteration}"
            )
        return True

    def _calculate_per_bin_accuracy(self, final, discrete_expr_label, num_bins):
        """
        Calculate accuracy for each bin, then average over bins (excluding bin0)
        final: (batch, seq_len)
        discrete_expr_label: (batch, seq_len)
        num_bins: int
        Returns: float, average accuracy per bin (excluding bin0)
        """
        accuracies = []
        # Exclude -100 cases
        valid_mask = discrete_expr_label != -100
        for bin_idx in range(1, num_bins + 1):  # Skip bin0
            # Only consider samples that are non -100 and label is bin_idx
            mask = (discrete_expr_label == bin_idx) & valid_mask
            total = mask.sum()
            if total == 0:
                continue  # This bin has no samples
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

    def accumulate_or_log_classification_metrics(self, final, discrete_expr_label):
        non_padded_mask = discrete_expr_label != -100
        valid_labels = discrete_expr_label[non_padded_mask]
        valid_preds = final[non_padded_mask]
        num_classes = self.args.model.num_bins + 1
        recall, precision, f1, macro_f1, average_recall, average_precision = (
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
        if self.iteration % self.args.log_on_wandb_every == 0:
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

        # Initialize counter
        class_counts = torch.zeros(self.args.model.num_bins + 1, dtype=torch.long)

        # Traverse training dataset to calculate sample count for each bin
        temp_loader = DataLoader(
            self.train_dataset,
            batch_size=32,  # Use smaller batch size to save memory
            shuffle=False,
            num_workers=4,
            collate_fn=DataCollator(
                do_padding=True,
                pad_token_id=0,
                pad_value=0,
                do_mlm=False,  # Don't use masking, get original data
                do_binning=True,
                max_length=self.args.sequence_length,
                num_genes=self.args.model.num_genes,
                num_bins=self.args.model.num_bins,
                # dynamic_mask_probabilities not needed here, as we're just counting distribution
            ),
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

    def calculate_dynamic_mask_probabilities(self):
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

    # need to predefine the weight and also need to ensure that the length of weight corresponding to the number of bins
    def init_weighted_ce_loss(self):
        # Use original weighted CrossEntropyLoss
        logging.info("Using weighted CrossEntropyLoss...")
        # bin0:0, bin1:1, bin2:6, bin3:36, bin4:100, bin5:300
        ce_weight = torch.tensor([0, 1, 6.9, 118.7])
        self.weighted_ce_loss_fn = nn.CrossEntropyLoss(
            weight=ce_weight, reduction="mean", ignore_index=-100
        )

    def calculate_weighted_ce_loss(self, logits, discrete_expr_label):
        self.weighted_ce_loss_fn.to(logits.device)
        logits_reshaped = logits.view(-1, self.args.model.num_bins + 1)
        labels_reshaped = discrete_expr_label.view(-1)
        return self.weighted_ce_loss_fn(logits_reshaped, labels_reshaped)

    def init_ldam_loss(self):
        logging.info("Using LDAM loss...")
        # 使用缓存的class_counts
        class_counts = self.class_counts
        cls_num_list = class_counts.cpu().numpy()

        # Ensure padding class (class 0) has reasonable weight, avoid division by zero error
        if cls_num_list[0] == 0:
            logging.warning(
                "Warning: Padding class (bin 0) has 0 samples, setting to 1 to avoid division by zero"
            )
            cls_num_list[0] = (
                1  # Set to 1 to avoid division by zero, but doesn't affect actual training
            )

        return LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, s=30, ignore_index=-100)

    # if: enable_alternating_ldam_mean_ce_loss
    # need to ensure that loss_fn is initialized with LDAM loss
    def calculate_alternating_ldam_mean_ce_loss(
        self, epoch, logits, discrete_expr_label
    ):
        if epoch % 2 == 1:
            ce_loss = self.calculate_ldam_ce_loss(logits, discrete_expr_label)
        else:
            ce_loss = self.calculate_mean_ce_loss(logits, discrete_expr_label)
        return ce_loss

    # if: enable_adaptive_ce_loss
    def calculate_mogaide_ce_loss(self, logits, discrete_expr_label):
        if self.epoch == 1 or self.epoch == 2:
            ce_loss = self.cross_entropy_loss_fn(logits, discrete_expr_label)
        elif self.epoch == 3 or self.epoch == 4:
            ce_loss = self.calculate_weighted_ce_loss(logits, discrete_expr_label)
        else:
            ce_loss = self.calculate_mean_ce_loss(logits, discrete_expr_label)
        return ce_loss

    # if: enable_warm_alternating_ldam_mean_ce_loss
    # need to ensure that loss_fn is initialized with LDAM loss
    def calculate_warm_alternating_ldam_mean_ce_loss(
        self, epoch, logits, discrete_expr_label, epoch_length, index
    ):
        ldam_loss = self.calculate_ldam_ce_loss(logits, discrete_expr_label)
        per_bin_ce_loss_mean = self.calculate_mean_ce_loss(logits, discrete_expr_label)
        # Odd rounds: first half biased towards ldam, even rounds: first half biased towards ce
        progress = index / epoch_length if epoch_length > 0 else 0.0
        if epoch % 2 == 1:
            # Odd rounds: first half more ldam, second half more ce
            ldam_weight = 1.0 - progress
            ce_weight = progress
        else:
            # Even rounds: first half more ce, second half more ldam
            ldam_weight = progress
            ce_weight = 1.0 - progress
        if index % 20 == 0:
            if self.is_master:
                wandb.log(
                    {
                        "TrainLossWeight/ldam_weight": ldam_weight,
                        "TrainLossWeight/ce_weight": ce_weight,
                    }
                )
        ce_loss = ldam_weight * ldam_loss + ce_weight * per_bin_ce_loss_mean
        return ce_loss

    def calculate_mean_ce_loss(self, logits, discrete_expr_label):
        per_bin_ce_loss = self.calculate_per_bin_ce_loss(logits, discrete_expr_label)
        mean_ce_loss = (
            per_bin_ce_loss.mean()
            if per_bin_ce_loss is not None
            else torch.tensor(0.0, device=logits.device)
        )
        return mean_ce_loss

    # if: use_ldam_loss
    # need to ensure that loss_fn is initialized with LDAM loss
    def calculate_ldam_ce_loss(self, logits, discrete_expr_label):
        self.ldam_loss_fn.to(logits.device)
        logits_reshaped = logits.view(-1, self.args.model.num_bins + 1)
        labels_reshaped = discrete_expr_label.view(-1)
        return self.ldam_loss_fn(logits_reshaped, labels_reshaped)

    def calculate_ce_loss(self, logits, discrete_expr_label):
        self.cross_entropy_loss_fn.to(logits.device)
        logits_reshaped = logits.view(-1, self.args.model.num_bins + 1)
        labels_reshaped = discrete_expr_label.view(-1)
        return self.cross_entropy_loss_fn(logits_reshaped, labels_reshaped)

    def calculate_adaptive_mse_weight(self):
        weight_grad = (
            self.args.target_mse_loss_weight * self.args.grad_acc / self.epoch_length
        )
        if (self.epoch == 1 and self.iteration > self.epoch_length / 3) or (
            self.epoch == 2 and self.iteration < 2 * self.epoch_length / 3
        ):
            self.mse_loss_weight = weight_grad + self.mse_loss_weight

    def init_loss_fn(self):
        if (
            self.args.enable_alternating_ldam_mean_ce_loss
            or self.args.enable_warm_alternating_ldam_mean_ce_loss
            or self.args.use_ldam_loss
        ):
            self.ldam_loss_fn = self.init_ldam_loss()
        elif self.args.weighted_ce_loss or self.args.enable_adaptive_ce_loss:
            self.init_weighted_ce_loss()
            self.cross_entropy_loss_fn = nn.CrossEntropyLoss(
                reduction="mean", ignore_index=-100
            )
        else:
            self.cross_entropy_loss_fn = nn.CrossEntropyLoss(
                reduction="mean", ignore_index=-100
            )

        if self.args.enable_mse_loss:
            self.regression_loss_fn = nn.MSELoss(reduction="none")
        elif self.args.enable_huber_loss:
            self.regression_loss_fn = nn.HuberLoss(reduction="none")
