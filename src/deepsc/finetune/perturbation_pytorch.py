"""
Perturbation Prediction Fine-tuning for DeepSC - Pure PyTorch Version
Based on scGPT's perturbation logic without Lightning Fabric dependency
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
    compute_perturbation_metrics,
    seed_all,
)


class PerturbationPredictorPyTorch:
    """
    Pure PyTorch Perturbation Predictor (no Fabric dependency)
    """

    def __init__(self, args, model, device='cuda'):
        """
        Initialize predictor

        Args:
            args: Configuration (DictConfig or Namespace)
            model: DeepSC model
            device: 'cuda' or 'cpu'
        """
        self.args = args
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Set seed
        seed_all(args.seed)

        print(f"Using device: {self.device}")

        # Setup output directory
        self.setup_output_directory()

        # Load vocabulary
        self.vocab, self.id2vocab, self.pad_token, self.pad_value = (
            build_vocab_from_csv(self.args.csv_path)
        )

        # Prepare data FIRST (before model loading)
        self.prepare_data()

        # Build gene mappings
        self.gene_ids = build_gene_ids_for_dataset(self.original_genes, self.vocab)
        self.name2col = {g: i for i, g in enumerate(self.original_genes)}
        self.valid_gene_mask = self.gene_ids != 0

        # Log gene alignment
        self.log_gene_alignment()

        # Move model to device BEFORE loading pretrained
        self.model = model.to(self.device)

        # Load pretrained model if specified
        if getattr(args, 'pretrained_model', False):
            self.load_pretrained_model()

        # Setup optimizer
        self.optimizer = Adam(
            self.model.parameters(),
            lr=getattr(args, 'learning_rate', 1e-4)
        )

        # Setup scheduler
        self.scheduler = self.create_scheduler()

        # Loss function
        self.criterion = nn.MSELoss()

        # Perturbations to plot
        self.perts_to_plot = getattr(args, 'perts_to_plot', None)
        if self.perts_to_plot is None:
            data_name = getattr(self.args, 'data_name', 'norman')
            if data_name == 'norman':
                self.perts_to_plot = ["SAMD1+ZBTB1"]
            elif data_name == 'adamson':
                self.perts_to_plot = ["KCTD16+ctrl"]
            else:
                self.perts_to_plot = []

    def setup_output_directory(self):
        """Setup output directories"""
        try:
            from hydra.core.hydra_config import HydraConfig
            hydra_cfg = HydraConfig.get()
            self.output_dir = Path(hydra_cfg.runtime.output_dir)
        except:
            data_name = getattr(self.args, 'data_name', 'default')
            self.output_dir = Path(f"./results/perturb_{data_name}-{time.strftime('%b%d-%H-%M')}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = self.output_dir / "checkpoints"
        self.log_dir = self.output_dir / "logs"
        self.vis_dir = self.output_dir / "visualizations"

        self.ckpt_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        self.vis_dir.mkdir(exist_ok=True)

        print(f"Output directory: {self.output_dir}")

    def log_gene_alignment(self):
        """Log gene alignment statistics"""
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

    def create_scheduler(self):
        """Create learning rate scheduler"""
        schedule_interval = getattr(self.args, 'schedule_interval', 1)
        gamma = getattr(self.args, 'scheduler_gamma', 0.9)
        return StepLR(self.optimizer, step_size=schedule_interval, gamma=gamma)

    def prepare_data(self):
        """Prepare perturbation data using GEARS"""
        data_name = getattr(self.args, 'data_name', 'norman')
        split = getattr(self.args, 'split', 'simulation')
        data_path = getattr(self.args, 'data_path', './data')
        batch_size = getattr(self.args, 'batch_size', 64)
        eval_batch_size = getattr(self.args, 'eval_batch_size', 64)

        print(f"\nLoading perturbation data: {data_name}")

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

        self.train_loader = pert_data.dataloader["train_loader"]
        self.valid_loader = pert_data.dataloader["val_loader"]
        self.test_loader = pert_data.dataloader["test_loader"]

        print(f"Loaded {self.num_genes} genes")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Valid batches: {len(self.valid_loader)}")
        print(f"Test batches: {len(self.test_loader)}")

    def map_raw_id_to_vocab_id(self, raw_ids, gene_ids):
        """Map raw gene indices to vocab indices"""
        if isinstance(raw_ids, torch.Tensor):
            device = raw_ids.device
            dtype = raw_ids.dtype

            if isinstance(gene_ids, torch.Tensor):
                gene_ids = gene_ids.to(device)
            else:
                gene_ids = torch.as_tensor(gene_ids, device=device)

            mapped_ids = gene_ids[raw_ids]
            return mapped_ids.to(dtype=dtype)
        else:
            raise ValueError("raw_ids must be torch.Tensor")

    def construct_pert_flags(self, batch_data, batch_size, device):
        """Construct perturbation flags"""
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
        """Discretize expression values"""
        batch_size = input_values.shape[0]
        discrete_bins = torch.zeros_like(input_values, dtype=torch.long)

        for i in range(batch_size):
            row_vals = input_values[i]
            valid_mask = row_vals > 0

            if valid_mask.any():
                valid_vals = row_vals[valid_mask]
                min_val = valid_vals.min()
                max_val = valid_vals.max()

                if max_val > min_val:
                    norm = (valid_vals - min_val) / (max_val - min_val + 1e-8)
                else:
                    norm = torch.zeros_like(valid_vals)

                bins = torch.floor(norm * (num_bins - 1)).long()
                bins = torch.clamp(bins, 0, num_bins - 1) + 1
                discrete_bins[i][valid_mask] = bins

        return discrete_bins

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0

        include_zero_gene = getattr(self.args, 'include_zero_gene', 'all')
        max_seq_len = getattr(self.args, 'data_length', 1536)
        grad_acc = getattr(self.args, 'grad_acc', 1)

        data_iter = tqdm(self.train_loader, desc=f"Epoch {epoch}", ncols=100)

        for batch_idx, batch_data in enumerate(data_iter):
            batch_size = len(batch_data.y)

            # Move to device
            batch_data = batch_data.to(self.device)

            # Extract data
            x = batch_data.x
            ori_gene_values = x[:, 0].view(batch_size, self.num_genes)
            target_gene_values = batch_data.y

            # Construct pert flags
            pert_flags = self.construct_pert_flags(batch_data, batch_size, self.device)

            # Select genes
            if include_zero_gene == 'all':
                input_gene_ids = torch.arange(
                    self.num_genes, device=self.device, dtype=torch.long
                )
            else:
                input_gene_ids = (
                    ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
                )

            if len(input_gene_ids) > max_seq_len:
                input_gene_ids = torch.randperm(
                    len(input_gene_ids), device=self.device
                )[:max_seq_len]

            input_values = ori_gene_values[:, input_gene_ids]
            input_pert_flags = pert_flags[:, input_gene_ids]
            target_values = target_gene_values[:, input_gene_ids]

            # Map gene IDs
            mapped_input_gene_ids = self.map_raw_id_to_vocab_id(
                input_gene_ids, self.gene_ids
            )
            mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

            # Discretize
            discrete_input_bins = self.discretize_expression(input_values)

            # Forward
            regression_output, _, _ = self.model(
                gene_ids=mapped_input_gene_ids,
                expression_bin=discrete_input_bins,
                normalized_expr=input_values,
                input_pert_flags=input_pert_flags,
            )

            # Loss
            loss = self.criterion(regression_output, target_values)
            loss = loss / grad_acc

            # Backward
            loss.backward()

            total_loss += loss.item() * grad_acc

            # Update
            if (batch_idx + 1) % grad_acc == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Update progress
            data_iter.set_postfix(loss=loss.item() * grad_acc)

        return total_loss / len(self.train_loader)

    def evaluate(self, loader):
        """Evaluate on dataloader"""
        self.model.eval()

        pert_cat = []
        pred = []
        truth = []
        pred_de = []
        truth_de = []

        include_zero_gene = getattr(self.args, 'include_zero_gene', 'all')
        max_seq_len = getattr(self.args, 'data_length', 1536)

        with torch.no_grad():
            for batch_data in tqdm(loader, desc="Evaluating", ncols=100):
                batch_size = len(batch_data.y)
                batch_data = batch_data.to(self.device)
                pert_cat.extend(batch_data.pert)

                x = batch_data.x
                ori_gene_values = x[:, 0].view(batch_size, self.num_genes)
                target_gene_values = batch_data.y

                pert_flags = self.construct_pert_flags(batch_data, batch_size, self.device)

                if include_zero_gene == 'all':
                    input_gene_ids = torch.arange(
                        self.num_genes, device=self.device, dtype=torch.long
                    )
                else:
                    input_gene_ids = (
                        ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
                    )

                if len(input_gene_ids) > max_seq_len:
                    input_gene_ids = torch.randperm(
                        len(input_gene_ids), device=self.device
                    )[:max_seq_len]

                input_values = ori_gene_values[:, input_gene_ids]
                input_pert_flags = pert_flags[:, input_gene_ids]

                mapped_input_gene_ids = self.map_raw_id_to_vocab_id(
                    input_gene_ids, self.gene_ids
                )
                mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

                discrete_input_bins = self.discretize_expression(input_values)

                regression_output, _, _ = self.model(
                    gene_ids=mapped_input_gene_ids,
                    expression_bin=discrete_input_bins,
                    normalized_expr=input_values,
                    input_pert_flags=input_pert_flags,
                )

                pred_gene_values = ori_gene_values.clone()
                pred_gene_values[:, input_gene_ids] = regression_output

                pred.extend(pred_gene_values.cpu())
                truth.extend(target_gene_values.cpu())

                for itr, de_idx in enumerate(batch_data.de_idx):
                    pred_de.append(pred_gene_values[itr, de_idx])
                    truth_de.append(target_gene_values[itr, de_idx])

        results = {}
        results['pert_cat'] = np.array(pert_cat)
        pred = torch.stack(pred)
        truth = torch.stack(truth)
        results['pred'] = pred.detach().cpu().numpy().astype(np.float64)
        results['truth'] = truth.detach().cpu().numpy().astype(np.float64)
        results['pred_de'] = torch.stack(pred_de).detach().cpu().numpy().astype(np.float64)
        results['truth_de'] = torch.stack(truth_de).detach().cpu().numpy().astype(np.float64)

        return results

    def train(self):
        """Main training loop"""
        epochs = getattr(self.args, 'epoch', 10)
        early_stop = getattr(self.args, 'early_stop', 10)

        best_val_corr = 0
        best_model = None
        patience = 0

        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)

        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()

            # Train
            train_loss = self.train_epoch(epoch)

            # Evaluate
            val_res = self.evaluate(self.valid_loader)
            val_metrics = compute_perturbation_metrics(
                val_res,
                self.pert_data.adata[self.pert_data.adata.obs["condition"] == "ctrl"]
            )

            print(f"\nEpoch {epoch} - Train Loss: {train_loss:.4f}")
            print(f"Val Metrics: {val_metrics}")

            elapsed = time.time() - epoch_start_time
            print(f"Time: {elapsed:.2f}s")

            # Check best
            val_score = val_metrics.get('pearson', 0)
            if val_score > best_val_corr:
                best_val_corr = val_score
                best_model = copy.deepcopy(self.model.state_dict())
                print(f"✓ New best model! Pearson: {val_score:.4f}")

                # Save
                torch.save(best_model, self.ckpt_dir / "best_model.pt")
                patience = 0
            else:
                patience += 1
                if patience >= early_stop:
                    print(f"Early stopping at epoch {epoch}")
                    break

            # Save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_metrics': val_metrics,
            }, self.ckpt_dir / f"epoch_{epoch}.pt")

            self.scheduler.step()

        # Load best model for testing
        if best_model is not None:
            self.model.load_state_dict(best_model)

        # Test
        print("\n" + "="*60)
        print("Testing")
        print("="*60)

        test_res = self.evaluate(self.test_loader)
        test_metrics = compute_perturbation_metrics(
            test_res,
            self.pert_data.adata[self.pert_data.adata.obs["condition"] == "ctrl"]
        )

        print(f"\nTest Metrics: {test_metrics}")

        # Save test metrics
        with open(self.output_dir / "test_metrics.json", "w") as f:
            json.dump(test_metrics, f, indent=2)

        # Deeper analysis
        print("\nRunning deeper analysis...")
        deeper_res = deeper_analysis(self.pert_data.adata, test_res)
        non_dropout_res = non_dropout_analysis(self.pert_data.adata, test_res)

        print("\n" + "="*60)
        print("Training Complete!")
        print(f"Results saved to: {self.output_dir}")
        print("="*60)

    def load_pretrained_model(self):
        """Load pretrained model"""
        ckpt_path = self.args.pretrained_model_path
        assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"

        print(f"Loading checkpoint: {ckpt_path}")
        raw = torch.load(ckpt_path, map_location=self.device)
        state_dict = extract_state_dict(raw)

        # Load with strict=False
        load_info = self.model.load_state_dict(state_dict, strict=False)

        print(f"Loaded pretrained model:")
        print(f"  Missing keys: {len(load_info.missing_keys)}")
        print(f"  Unexpected keys: {len(load_info.unexpected_keys)}")
