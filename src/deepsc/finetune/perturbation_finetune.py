"""
Perturbation Prediction Fine-tuning for DeepSC
Based on scGPT's perturbation prediction logic with DeepSC's gene alignment workflow
"""

import copy
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from gears import PertData
from gears.inference import deeper_analysis, non_dropout_analysis
from gears.utils import create_cell_graph_dataset_for_prediction
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from deepsc.utils import (
    build_gene_ids_for_dataset,
    build_vocab_from_csv,
    extract_state_dict,
    report_loading_result,
    sample_weight_norms,
    compute_perturbation_metrics,
    seed_all,
)


class PerturbationPredictor:
    """
    Perturbation Prediction class for DeepSC model
    Adapts scGPT's perturbation prediction workflow to DeepSC architecture
    """

    def __init__(self, args, fabric, model):
        """
        Initialize the perturbation predictor

        Args:
            args: Configuration arguments (Hydra DictConfig or argparse Namespace)
            fabric: Lightning Fabric for distributed training
            model: DeepSC model instance
        """
        self.args = args
        self.fabric = fabric
        self.model = model
        self.world_size = self.fabric.world_size

        # Set random seed for reproducibility
        seed_all(args.seed + self.fabric.global_rank)
        self.is_master = self.fabric.global_rank == 0

        # Setup directories
        self.setup_output_directory()

        # Load vocabulary BEFORE model setup (needed for pretrained loading)
        self.vocab, self.id2vocab, self.pad_token, self.pad_value = (
            build_vocab_from_csv(self.args.csv_path)
        )

        # Load pretrained model if specified
        if getattr(args, 'pretrained_model', False):
            self.load_pretrained_model()

        # Setup optimizer
        self.optimizer = Adam(
            self.model.parameters(),
            lr=getattr(args, 'learning_rate', 1e-4)
        )

        # Setup model and optimizer with fabric
        self.model, self.optimizer = self.fabric.setup(self.model, self.optimizer)

        # Initialize loss function
        self.init_loss_fn()

        # Setup learning rate scheduler
        self.scheduler = self.create_scheduler()

        # Prepare perturbation data
        self.prepare_data()

        # Build gene ID mappings (align dataset genes with vocab)
        self.gene_ids = build_gene_ids_for_dataset(self.original_genes, self.vocab)
        self.name2col = {g: i for i, g in enumerate(self.original_genes)}
        self.valid_gene_mask = self.gene_ids != 0

        # Perturbations to plot for visualization
        self.perts_to_plot = getattr(args, 'perts_to_plot', None)
        if self.perts_to_plot is None:
            data_name = getattr(self.args, 'data_name', 'norman')
            if data_name == 'norman':
                self.perts_to_plot = ["SAMD1+ZBTB1"]
            elif data_name == 'adamson':
                self.perts_to_plot = ["KCTD16+ctrl"]
            else:
                self.perts_to_plot = []

        # Log gene alignment info
        if self.is_master:
            # Convert to numpy if it's a tensor
            gene_ids_np = self.gene_ids.cpu().numpy() if isinstance(self.gene_ids, torch.Tensor) else self.gene_ids
            matched_genes = np.sum(gene_ids_np != 0)
            total_genes = len(gene_ids_np)
            vocab_size = len(self.vocab)
            print(f"\n{'='*60}")
            print(f"Gene Alignment Summary:")
            print(f"  Dataset genes: {total_genes}")
            print(f"  Matched genes: {matched_genes} ({100*matched_genes/total_genes:.1f}%)")
            print(f"  Unmatched genes: {total_genes - matched_genes}")
            print(f"  Vocabulary size: {vocab_size}")
            print(f"{'='*60}\n")

            # Compare name2col with node_map if available
            if hasattr(self, 'node_map'):
                name2col_equal = self.name2col == self.node_map
                print(f"self.name2col == self.node_map: {name2col_equal}")

    def setup_output_directory(self):
        """
        Setup output directories for checkpoints, logs, and visualizations
        """
        if self.is_master:
            # Use Hydra output directory if available
            try:
                from hydra.core.hydra_config import HydraConfig
                hydra_cfg = HydraConfig.get()
                self.output_dir = hydra_cfg.runtime.output_dir
            except:
                # Fallback to time-stamped directory
                data_name = getattr(self.args, 'data_name', 'default')
                self.output_dir = f"./save/perturb_{data_name}-{time.strftime('%b%d-%H-%M')}"

            # Create subdirectories
            self.output_dir = Path(self.output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

            self.ckpt_dir = self.output_dir / "checkpoints"
            self.log_dir = self.output_dir / "logs"
            self.vis_dir = self.output_dir / "visualizations"

            self.ckpt_dir.mkdir(exist_ok=True)
            self.log_dir.mkdir(exist_ok=True)
            self.vis_dir.mkdir(exist_ok=True)

            print(f"Output directory: {self.output_dir}")
            print(f"  - Checkpoints: {self.ckpt_dir}")
            print(f"  - Logs: {self.log_dir}")
            print(f"  - Visualizations: {self.vis_dir}")
        else:
            self.output_dir = None
            self.ckpt_dir = None
            self.log_dir = None
            self.vis_dir = None

        # Broadcast to all processes
        self.output_dir = self.fabric.broadcast(self.output_dir, src=0)
        self.ckpt_dir = self.fabric.broadcast(self.ckpt_dir, src=0)
        self.log_dir = self.fabric.broadcast(self.log_dir, src=0)
        self.vis_dir = self.fabric.broadcast(self.vis_dir, src=0)

    def init_loss_fn(self):
        """Initialize loss function (MSE for regression)"""
        self.criterion = nn.MSELoss()

    def create_scheduler(self):
        """
        Create learning rate scheduler
        Similar to scGPT's StepLR scheduler
        """
        schedule_interval = getattr(self.args, 'schedule_interval', 1)
        gamma = getattr(self.args, 'scheduler_gamma', 0.9)

        scheduler = StepLR(
            self.optimizer,
            step_size=schedule_interval,
            gamma=gamma
        )
        return scheduler

    def prepare_data(self):
        """
        Prepare perturbation data using GEARS
        Similar to scGPT's data preparation
        """
        data_name = getattr(self.args, 'data_name', 'norman')
        split = getattr(self.args, 'split', 'simulation')
        data_path = getattr(self.args, 'data_path', './data')
        batch_size = getattr(self.args, 'batch_size', 64)
        eval_batch_size = getattr(self.args, 'eval_batch_size', 64)

        if self.is_master:
            print(f"Loading perturbation data: {data_name}")
            print(f"Split: {split}, Data path: {data_path}")

        # Load perturbation data
        pert_data = PertData(data_path)
        pert_data.load(data_name=data_name)
        pert_data.prepare_split(split=split, seed=1)
        pert_data.get_dataloader(
            batch_size=batch_size,
            test_batch_size=eval_batch_size
        )

        self.pert_data = pert_data
        self.original_genes = pert_data.adata.var["gene_name"].tolist()
        self.num_genes = len(self.original_genes)
        self.node_map = pert_data.node_map

        # Setup dataloaders with fabric
        self.train_loader = pert_data.dataloader["train_loader"]
        self.valid_loader = pert_data.dataloader["val_loader"]
        self.test_loader = pert_data.dataloader["test_loader"]

        self.train_loader, self.valid_loader, self.test_loader = self.fabric.setup_dataloaders(
            self.train_loader, self.valid_loader, self.test_loader
        )

        if self.is_master:
            print(f"Loaded {self.num_genes} genes")
            print(f"Train batches: {len(self.train_loader)}")
            print(f"Valid batches: {len(self.valid_loader)}")
            print(f"Test batches: {len(self.test_loader)}")

    def map_raw_id_to_vocab_id(self, raw_ids, gene_ids):
        """
        Map raw gene indices to vocab indices
        Equivalent to scGPT's map_raw_id_to_vocab_id

        Args:
            raw_ids: torch.Tensor of raw gene indices (positions in dataset)
            gene_ids: torch.Tensor or np.ndarray of vocab indices for each gene

        Returns:
            torch.Tensor of mapped vocab indices
        """
        if isinstance(raw_ids, torch.Tensor):
            device = raw_ids.device
            dtype = raw_ids.dtype

            # Ensure gene_ids is on same device
            if isinstance(gene_ids, torch.Tensor):
                gene_ids = gene_ids.to(device)
            else:
                gene_ids = torch.as_tensor(gene_ids, device=device)

            # Map raw indices to vocab indices
            mapped_ids = gene_ids[raw_ids]
            return mapped_ids.to(dtype=dtype)
        else:
            raise ValueError("raw_ids must be torch.Tensor")

    def construct_pert_flags(self, batch_data, batch_size, device):
        """
        Construct perturbation flags from batch_data.pert
        Based on scGPT's construct_pert_flags function

        Args:
            batch_data: Batch data from GEARS dataloader
            batch_size: Batch size
            device: Torch device

        Returns:
            pert_flags: torch.Tensor of shape (batch_size, n_genes) with 1 for perturbed genes
        """
        pert_flags = torch.zeros(
            batch_size, self.num_genes,
            device=device, dtype=torch.long
        )

        for r, p in enumerate(batch_data.pert):
            for g in p.split("+"):
                if g and g != "ctrl":
                    j = self.name2col.get(g, -1)
                    if j != -1:
                        pert_flags[r, j] = 1

        return pert_flags

    def discretize_expression(self, input_values, num_bins=5):
        """
        Discretize expression values into bins
        For compatibility with DeepSC's expression embedding

        Args:
            input_values: Expression values tensor
            num_bins: Number of bins for discretization

        Returns:
            discrete_bins: Discretized expression bins
        """
        batch_size = input_values.shape[0]
        discrete_bins = torch.zeros_like(input_values, dtype=torch.long)

        for i in range(batch_size):
            row_vals = input_values[i]
            valid_mask = row_vals > 0  # Non-zero expression

            if valid_mask.any():
                valid_vals = row_vals[valid_mask]
                min_val = valid_vals.min()
                max_val = valid_vals.max()

                # Normalize to [0, 1]
                if max_val > min_val:
                    norm = (valid_vals - min_val) / (max_val - min_val + 1e-8)
                else:
                    norm = torch.zeros_like(valid_vals)

                # Bin into discrete values [1, num_bins]
                bins = torch.floor(norm * (num_bins - 1)).long()
                bins = torch.clamp(bins, 0, num_bins - 1) + 1  # +1 to reserve 0 for padding
                discrete_bins[i][valid_mask] = bins

        return discrete_bins

    def train_epoch(self, epoch):
        """
        Train the model for one epoch
        Based on scGPT's train function

        Args:
            epoch: Current epoch number
        """
        self.model.train()
        total_loss = 0.0
        total_mse = 0.0
        start_time = time.time()

        # Settings
        include_zero_gene = getattr(self.args, 'include_zero_gene', 'all')
        max_seq_len = getattr(self.args, 'data_length', 1536)
        amp = getattr(self.args, 'amp', True)
        log_interval = getattr(self.args, 'log_interval', 100)
        grad_acc = getattr(self.args, 'grad_acc', 1)

        # Progress bar
        data_iter = self.train_loader
        if self.is_master:
            data_iter = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch}",
                ncols=150,
                position=0
            )

        num_batches = len(self.train_loader)

        for batch_idx, batch_data in enumerate(data_iter):
            batch_size = len(batch_data.y)
            device = batch_data.x.device

            # Extract data from GEARS format
            x = batch_data.x  # (batch_size * n_genes, 2)
            ori_gene_values = x[:, 0].view(batch_size, self.num_genes)
            target_gene_values = batch_data.y  # (batch_size, n_genes)

            # Construct perturbation flags
            pert_flags = self.construct_pert_flags(batch_data, batch_size, device)

            # Select genes to include (all genes or batch-wise non-zero)
            if include_zero_gene in ['all', 'batch-wise']:
                if include_zero_gene == 'all':
                    input_gene_ids = torch.arange(
                        self.num_genes, device=device, dtype=torch.long
                    )
                else:
                    # Get genes with non-zero expression in batch
                    input_gene_ids = (
                        ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
                    )

                # Sample if too many genes
                if len(input_gene_ids) > max_seq_len:
                    input_gene_ids = torch.randperm(
                        len(input_gene_ids), device=device
                    )[:max_seq_len]

                # Select values for chosen genes
                input_values = ori_gene_values[:, input_gene_ids]
                input_pert_flags = pert_flags[:, input_gene_ids]
                target_values = target_gene_values[:, input_gene_ids]

                # Map gene IDs to vocabulary IDs
                mapped_input_gene_ids = self.map_raw_id_to_vocab_id(
                    input_gene_ids, self.gene_ids
                )
                mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

                # Discretize expression for DeepSC's expression embedding
                discrete_input_bins = self.discretize_expression(input_values)

                # Forward pass with AMP
                with torch.cuda.amp.autocast(enabled=amp):
                    # DeepSC model forward
                    regression_output, gene_emb, expr_emb = self.model(
                        gene_ids=mapped_input_gene_ids,
                        expression_bin=discrete_input_bins,
                        normalized_expr=input_values,
                        input_pert_flags=input_pert_flags,
                    )

                    # Compute MSE loss
                    loss = self.criterion(regression_output, target_values)

                # Gradient accumulation
                loss = loss / grad_acc
                self.fabric.backward(loss)

                # Accumulate loss for logging
                total_loss += loss.item() * grad_acc
                total_mse += loss.item() * grad_acc

                # Update parameters every grad_acc steps
                if (batch_idx + 1) % grad_acc == 0 or (batch_idx + 1) == num_batches:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        1.0,
                        error_if_nonfinite=False if amp else True
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                # Logging
                if batch_idx % log_interval == 0 and batch_idx > 0:
                    lr = self.scheduler.get_last_lr()[0]
                    ms_per_batch = (time.time() - start_time) * 1000 / log_interval
                    cur_loss = total_loss / log_interval
                    cur_mse = total_mse / log_interval

                    if self.is_master:
                        print(
                            f"| epoch {epoch:3d} | {batch_idx:3d}/{num_batches:3d} batches | "
                            f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                            f"loss {cur_loss:5.2f} | mse {cur_mse:5.2f} |"
                        )

                    total_loss = 0
                    total_mse = 0
                    start_time = time.time()

                # Update progress bar
                if self.is_master:
                    data_iter.set_postfix(
                        loss=loss.item() * grad_acc,
                        avg_loss=total_loss / (batch_idx + 1) if batch_idx > 0 else loss.item() * grad_acc
                    )

    def evaluate(self, loader):
        """
        Evaluate model on given dataloader
        Based on scGPT's eval_perturb function

        Args:
            loader: DataLoader to evaluate on

        Returns:
            results: Dictionary containing predictions and ground truth
        """
        self.model.eval()

        pert_cat = []
        pred = []
        truth = []
        pred_de = []
        truth_de = []
        results = {}

        include_zero_gene = getattr(self.args, 'include_zero_gene', 'all')
        max_seq_len = getattr(self.args, 'data_length', 1536)

        # Progress bar
        data_iter = loader
        if self.is_master:
            data_iter = tqdm(
                loader,
                desc="Evaluating",
                ncols=150,
                position=1
            )

        with torch.no_grad():
            for batch_data in data_iter:
                batch_size = len(batch_data.y)
                device = batch_data.x.device
                pert_cat.extend(batch_data.pert)

                # Extract data
                x = batch_data.x
                ori_gene_values = x[:, 0].view(batch_size, self.num_genes)
                target_gene_values = batch_data.y

                # Construct perturbation flags
                pert_flags = self.construct_pert_flags(batch_data, batch_size, device)

                # Select genes
                if include_zero_gene in ['all', 'batch-wise']:
                    if include_zero_gene == 'all':
                        input_gene_ids = torch.arange(
                            self.num_genes, device=device, dtype=torch.long
                        )
                    else:
                        input_gene_ids = (
                            ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
                        )

                    if len(input_gene_ids) > max_seq_len:
                        input_gene_ids = torch.randperm(
                            len(input_gene_ids), device=device
                        )[:max_seq_len]

                    input_values = ori_gene_values[:, input_gene_ids]
                    input_pert_flags = pert_flags[:, input_gene_ids]

                    # Map gene IDs
                    mapped_input_gene_ids = self.map_raw_id_to_vocab_id(
                        input_gene_ids, self.gene_ids
                    )
                    mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

                    # Discretize expression
                    discrete_input_bins = self.discretize_expression(input_values)

                    # Forward pass
                    regression_output, _, _ = self.model(
                        gene_ids=mapped_input_gene_ids,
                        expression_bin=discrete_input_bins,
                        normalized_expr=input_values,
                        input_pert_flags=input_pert_flags,
                    )

                    # Reconstruct full prediction
                    pred_gene_values = ori_gene_values.clone()
                    pred_gene_values[:, input_gene_ids] = regression_output

                    pred.extend(pred_gene_values.cpu())
                    truth.extend(target_gene_values.cpu())

                    # Differentially expressed genes
                    for itr, de_idx in enumerate(batch_data.de_idx):
                        pred_de.append(pred_gene_values[itr, de_idx])
                        truth_de.append(target_gene_values[itr, de_idx])

        # Compile results
        results['pert_cat'] = np.array(pert_cat)
        pred = torch.stack(pred)
        truth = torch.stack(truth)
        results['pred'] = pred.detach().cpu().numpy().astype(np.float64)
        results['truth'] = truth.detach().cpu().numpy().astype(np.float64)

        pred_de = torch.stack(pred_de)
        truth_de = torch.stack(truth_de)
        results['pred_de'] = pred_de.detach().cpu().numpy().astype(np.float64)
        results['truth_de'] = truth_de.detach().cpu().numpy().astype(np.float64)

        return results

    def train(self):
        """
        Main training loop
        """
        epochs = getattr(self.args, 'epoch', 10)
        early_stop = getattr(self.args, 'early_stop', 10)

        best_val_corr = 0
        best_model = None
        patience = 0

        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()

            # Train one epoch
            self.train_epoch(epoch)

            # Evaluate on validation set
            val_res = self.evaluate(self.valid_loader)
            val_metrics = compute_perturbation_metrics(
                val_res,
                self.pert_data.adata[self.pert_data.adata.obs["condition"] == "ctrl"]
            )

            if self.is_master:
                print(f"\nValidation metrics at epoch {epoch}:")
                print(val_metrics)

            elapsed = time.time() - epoch_start_time
            if self.is_master:
                print(f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s |")

            # Check for best model
            val_score = val_metrics.get('pearson', 0)
            if val_score > best_val_corr:
                best_val_corr = val_score
                best_model = copy.deepcopy(self.model)

                if self.is_master:
                    print(f"Best model with score {val_score:5.4f}")
                    # Save best model
                    best_ckpt_path = self.ckpt_dir / "best_model.pt"
                    self.fabric.save(
                        best_ckpt_path,
                        {"model": self.model, "optimizer": self.optimizer, "epoch": epoch}
                    )
                    print(f"Best model saved to: {best_ckpt_path}")

                patience = 0
            else:
                patience += 1
                if patience >= early_stop:
                    if self.is_master:
                        print(f"Early stopping at epoch {epoch}")
                    break

            # Save checkpoint
            if self.is_master:
                ckpt_path = self.ckpt_dir / f"epoch_{epoch}.pt"
                self.fabric.save(
                    ckpt_path,
                    {"model": self.model, "optimizer": self.optimizer, "epoch": epoch}
                )
                print(f"Checkpoint saved to: {ckpt_path}")

            # Step scheduler
            self.scheduler.step()

        # Test evaluation with best model
        if self.is_master:
            print("\nRunning test evaluation with best model...")

        if best_model is not None:
            self.model = best_model

        test_res = self.evaluate(self.test_loader)
        test_metrics = compute_perturbation_metrics(
            test_res,
            self.pert_data.adata[self.pert_data.adata.obs["condition"] == "ctrl"]
        )

        if self.is_master:
            print("\nTest metrics:")
            print(test_metrics)

            # Save test metrics
            with open(self.output_dir / "test_metrics.json", "w") as f:
                json.dump(test_metrics, f, indent=2)

            print(f"\nTest metrics saved to: {self.output_dir / 'test_metrics.json'}")

            # Run deeper analysis
            print("\nRunning deeper analysis...")
            deeper_res = deeper_analysis(self.pert_data.adata, test_res)
            print("Running non-dropout analysis...")
            non_dropout_res = non_dropout_analysis(self.pert_data.adata, test_res)

            # Subgroup analysis
            self.run_subgroup_analysis(deeper_res, non_dropout_res)

    def run_subgroup_analysis(self, deeper_res, non_dropout_res):
        """
        Run subgroup analysis on test results
        Based on scGPT's subgroup analysis
        """
        metrics = ["pearson_delta", "pearson_delta_de"]
        metrics_non_dropout = [
            "pearson_delta_top20_de_non_dropout",
            "pearson_top20_de_non_dropout",
        ]

        subgroup_analysis = {}
        for name in self.pert_data.subgroup["test_subgroup"].keys():
            subgroup_analysis[name] = {}
            for m in metrics:
                subgroup_analysis[name][m] = []
            for m in metrics_non_dropout:
                subgroup_analysis[name][m] = []

        for name, pert_list in self.pert_data.subgroup["test_subgroup"].items():
            for pert in pert_list:
                for m in metrics:
                    subgroup_analysis[name][m].append(deeper_res[pert][m])
                for m in metrics_non_dropout:
                    subgroup_analysis[name][m].append(non_dropout_res[pert][m])

        if self.is_master:
            print("\nSubgroup analysis:")
            for name, result in subgroup_analysis.items():
                for m in result.keys():
                    mean_value = np.mean(subgroup_analysis[name][m])
                    print(f"test_{name}_{m}: {mean_value:.4f}")

    def load_pretrained_model(self):
        """
        Load pretrained model weights
        """
        ckpt_path = self.args.pretrained_model_path
        assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"

        if self.fabric.global_rank == 0:
            print(f"Loading checkpoint: {ckpt_path}")
            raw = torch.load(ckpt_path, map_location="cpu")
            state_dict = extract_state_dict(raw)
        else:
            raw = None
            state_dict = None

        # Broadcast to all processes
        state_dict = self.fabric.broadcast(state_dict, src=0)

        # Sample weight norms for comparison
        sample_weight_norms(self.model, state_dict, k=5)

        # Load state dict (strict=False to allow new parameters)
        load_info = self.model.load_state_dict(state_dict, strict=False)
        report_loading_result(load_info)

    def predict(self, pert_list: List[List[str]], pool_size: Optional[int] = None) -> Dict:
        """
        Predict gene expression values for given perturbations
        Based on scGPT's predict function

        Args:
            pert_list: List of perturbations, e.g., [["FEV"], ["FEV", "SAMD11"]]
            pool_size: Number of control cells to use for prediction (None = all)

        Returns:
            results_pred: Dictionary mapping perturbation names to predicted expression
        """
        adata = self.pert_data.adata
        ctrl_adata = adata[adata.obs["condition"] == "ctrl"]

        if pool_size is None:
            pool_size = len(ctrl_adata.obs)

        gene_list = self.pert_data.gene_names.values.tolist()

        # Validate perturbations
        for pert in pert_list:
            for gene in pert:
                if gene not in gene_list:
                    raise ValueError(
                        f"Gene {gene} not in perturbation graph. "
                        f"Please select from available genes."
                    )

        self.model.eval()
        device = next(self.model.parameters()).device

        include_zero_gene = getattr(self.args, 'include_zero_gene', 'all')
        max_seq_len = getattr(self.args, 'data_length', 1536)
        eval_batch_size = getattr(self.args, 'eval_batch_size', 64)

        with torch.no_grad():
            results_pred = {}
            for pert in pert_list:
                # Create cell graphs for this perturbation
                cell_graphs = create_cell_graph_dataset_for_prediction(
                    pert, ctrl_adata, gene_list, device, num_samples=pool_size
                )
                loader = DataLoader(cell_graphs, batch_size=eval_batch_size, shuffle=False)

                preds = []
                for batch_data in loader:
                    batch_size = len(batch_data.y)
                    device = batch_data.x.device

                    # Extract data
                    x = batch_data.x
                    ori_gene_values = x[:, 0].view(batch_size, self.num_genes)

                    # Construct perturbation flags
                    pert_flags = self.construct_pert_flags(batch_data, batch_size, device)

                    # Select genes
                    if include_zero_gene in ['all', 'batch-wise']:
                        if include_zero_gene == 'all':
                            input_gene_ids = torch.arange(
                                self.num_genes, device=device, dtype=torch.long
                            )
                        else:
                            input_gene_ids = (
                                ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
                            )

                        if len(input_gene_ids) > max_seq_len:
                            input_gene_ids = torch.randperm(
                                len(input_gene_ids), device=device
                            )[:max_seq_len]

                        input_values = ori_gene_values[:, input_gene_ids]
                        input_pert_flags = pert_flags[:, input_gene_ids]

                        # Map gene IDs
                        mapped_input_gene_ids = self.map_raw_id_to_vocab_id(
                            input_gene_ids, self.gene_ids
                        )
                        mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

                        # Discretize expression
                        discrete_input_bins = self.discretize_expression(input_values)

                        # Forward pass
                        regression_output, _, _ = self.model(
                            gene_ids=mapped_input_gene_ids,
                            expression_bin=discrete_input_bins,
                            normalized_expr=input_values,
                            input_pert_flags=input_pert_flags,
                        )

                        # Reconstruct full prediction
                        pred_gene_values = ori_gene_values.clone()
                        pred_gene_values[:, input_gene_ids] = regression_output

                        preds.append(pred_gene_values)

                # Average predictions across control cells
                preds = torch.cat(preds, dim=0)
                results_pred["_".join(pert)] = np.mean(preds.detach().cpu().numpy(), axis=0)

        return results_pred

    def plot_perturbation(
        self,
        query: str,
        save_file: Optional[str] = None,
        pool_size: Optional[int] = None
    ) -> matplotlib.figure.Figure:
        """
        Plot perturbation results for a specific query
        Based on scGPT's plot_perturbation function

        Args:
            query: Perturbation query string (e.g., "SAMD1+ZBTB1" or "KCTD16+ctrl")
            save_file: Path to save the plot (optional)
            pool_size: Number of control cells to use for prediction

        Returns:
            fig: Matplotlib figure
        """
        sns.set_theme(style="ticks", rc={"axes.facecolor": (0, 0, 0, 0)}, font_scale=1.5)

        adata = self.pert_data.adata
        gene2idx = self.pert_data.node_map
        cond2name = dict(adata.obs[["condition", "condition_name"]].values)
        gene_raw2id = dict(zip(adata.var.index.values, adata.var.gene_name.values))

        # Get top differentially expressed genes for this perturbation
        de_idx = [
            gene2idx[gene_raw2id[i]]
            for i in adata.uns["top_non_dropout_de_20"][cond2name[query]]
        ]
        genes = [
            gene_raw2id[i] for i in adata.uns["top_non_dropout_de_20"][cond2name[query]]
        ]

        # Get ground truth
        truth = adata[adata.obs.condition == query].X.toarray()[:, de_idx]

        # Get predictions
        if query.split("+")[1] == "ctrl":
            pred = self.predict([[query.split("+")[0]]], pool_size=pool_size)
            pred = pred[query.split("+")[0]][de_idx]
        else:
            pred = self.predict([query.split("+")], pool_size=pool_size)
            pred = pred["_".join(query.split("+"))][de_idx]

        # Get control means
        ctrl_means = adata[adata.obs["condition"] == "ctrl"].to_df().mean()[de_idx].values

        # Compute changes relative to control
        pred = pred - ctrl_means
        truth = truth - ctrl_means

        # Create plot
        fig, ax = plt.subplots(figsize=[16.5, 4.5])
        plt.title(query)
        plt.boxplot(truth, showfliers=False, medianprops=dict(linewidth=0))

        # Plot predictions as red dots
        for i in range(pred.shape[0]):
            _ = plt.scatter(i + 1, pred[i], color="red")

        plt.axhline(0, linestyle="dashed", color="green")

        ax.xaxis.set_ticklabels(genes, rotation=90)

        plt.ylabel("Change in Gene Expression over Control", labelpad=10)
        plt.tick_params(axis="x", which="major", pad=5)
        plt.tick_params(axis="y", which="major", pad=5)
        sns.despine()

        if save_file:
            fig.savefig(save_file, bbox_inches="tight", transparent=False)

        return fig

    def plot_predictions(self):
        """
        Generate plots for configured perturbations
        """
        if not self.perts_to_plot:
            if self.is_master:
                print("No perturbations configured for plotting")
            return

        if self.is_master:
            print(f"\nPlotting perturbations: {self.perts_to_plot}")

        for pert in self.perts_to_plot:
            if self.is_master:
                print(f"Plotting perturbation: {pert}")
                save_path = self.vis_dir / f"{pert}.png"
                self.plot_perturbation(pert, pool_size=300, save_file=str(save_path))
                print(f"Plot saved to: {save_path}")
