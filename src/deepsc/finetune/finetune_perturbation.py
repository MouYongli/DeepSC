"""
Fine-tune perturbation prediction model.
Supports both scGPT and DeepSC models.
Migrated from scGPT examples, all dependencies are self-contained.
"""
import json
import os
import sys
import time
import copy
import logging
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Union, Optional
import warnings

import torch
import numpy as np
import matplotlib
from torch import nn
from torch.nn import functional as F
from torch_geometric.loader import DataLoader
from gears import PertData, GEARS
from gears.inference import compute_metrics, deeper_analysis, non_dropout_analysis
from gears.utils import create_cell_graph_dataset_for_prediction

matplotlib.rcParams["savefig.transparent"] = False
warnings.filterwarnings("ignore")

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# MODEL SELECTION - Choose "scgpt" or "deepsc"
# ============================================================================
MODEL_TYPE = "deepsc"  # Options: "scgpt" or "deepsc"

# Import model-specific modules based on selection
if MODEL_TYPE == "scgpt":
    from deepsc.models.scgpt_pert import (
        TransformerGenerator,
        masked_mse_loss,
        criterion_neg_log_bernoulli,
        masked_relative_error,
        GeneVocab,
        tokenize_batch,
        pad_batch,
        tokenize_and_pad_batch,
        set_seed,
        map_raw_id_to_vocab_id,
        compute_perturbation_metrics,
        load_pretrained,
        add_file_handler,
    )
    set_seed(42)
elif MODEL_TYPE == "deepsc":
    from deepsc.models.generation_deepsc.model import DeepSC
    from deepsc.utils import (
        build_vocab_from_csv,
        build_gene_ids_for_dataset,
        seed_all,
        compute_perturbation_metrics,
    )
    seed_all(42)
else:
    raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}. Must be 'scgpt' or 'deepsc'")

# ============================================================================
# COMMON SETTINGS
# ============================================================================
# settings for data processing
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
pad_value = 0  # for padding values
pert_pad_id = 0
include_zero_gene = "all"
max_seq_len = 1536

# settings for training
amp = True
batch_size = 64
eval_batch_size = 64
epochs = 10
schedule_interval = 1
early_stop = 10
log_interval = 100
lr = 1e-4

# plotting settings
plot_scatter = False  # Set to True to plot prediction vs target scatter plots

# dataset and evaluation choices
data_name = "norman"
split = "simulation"
if data_name == "norman":
    perts_to_plot = ["SAMD1+ZBTB1"]
elif data_name == "adamson":
    perts_to_plot = ["KCTD16+ctrl"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# scGPT MODEL CONFIGURATION
# ============================================================================
if MODEL_TYPE == "scgpt":
    # scGPT training objectives
    MLM = True  # whether to use masked language modeling
    CLS = False  # celltype classification objective
    CCE = False  # Contrastive cell embedding objective
    MVC = False  # Masked value prediction for cell embedding
    ECS = False  # Elastic cell similarity objective

    # scGPT pretrained model
    load_model = "/home/angli/DeepSC/results/scGPT_human"
    load_param_prefixs = [
        "encoder",
        "value_encoder",
        "transformer_encoder",
    ]

    # scGPT model architecture
    embsize = 512  # embedding dimension
    d_hid = 512  # dimension of the feedforward network model
    nlayers = 12  # number of transformer layers
    nhead = 8  # number of attention heads
    n_layers_cls = 3
    dropout = 0  # dropout probability
    use_fast_transformer = True  # whether to use fast transformer

# ============================================================================
# DeepSC MODEL CONFIGURATION
# ============================================================================
elif MODEL_TYPE == "deepsc":
    # DeepSC pretrained model and vocabulary
    load_model = "/home/angli/DeepSC/results/pretraining_1201/latest_checkpoint.ckpt"
    vocab_csv_path = "/home/angli/baseline/DeepSC-117-t86/scripts/data/preprocessing/gene_map_tp10k.csv"

    # DeepSC model architecture (must match pretrained model)
    embedding_dim = 256
    num_genes = 35210  # vocabulary size
    num_layers = 10
    num_heads = 8
    num_bins = 5  # number of expression bins (MUST match pretrained model!)
    alpha = 0.5  # balance between discrete and continuous features
    attn_dropout = 0.1
    ffn_dropout = 0.1
    num_layers_ffn = 2
    gene_embedding_participate_til_layer = 3
    attention_stream = 2
    cross_attention_architecture = "A"

    # DeepSC MoE configuration
    use_moe_regressor = True
    use_moe_ffn = False
    n_moe_layers = 4
    moe_dim = 256
    moe_inter_dim = 512
    n_routed_experts = 2
    n_activated_experts = 2
    n_shared_experts = 1
    moe_score_func = "softmax"
    moe_route_scale = 1.0

    # DeepSC loss configuration
    enable_l0 = False
    enable_mse = True
    enable_ce = False

save_dir = Path(f"./save/dev_perturb_{MODEL_TYPE}_{data_name}-{time.strftime('%b%d-%H-%M')}/")
save_dir.mkdir(parents=True, exist_ok=True)
print(f"saving to {save_dir}")

# Add file handler for logging
def add_file_handler(logger, log_file):
    """Add file handler to logger"""
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(funcName)s: %(message)s',
                                  datefmt='%H:%M:%S')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

add_file_handler(logger, save_dir / "run.log")
logger.info(f"Running on {time.strftime('%Y-%m-%d %H:%M:%S')}")
logger.info(f"Model type: {MODEL_TYPE}")

# Load perturbation data
pert_data = PertData("./archive/data")
pert_data.load(data_name=data_name)
pert_data.prepare_split(split=split, seed=1)
pert_data.get_dataloader(batch_size=batch_size, test_batch_size=eval_batch_size)

genes = pert_data.adata.var["gene_name"].tolist()
n_genes = len(genes)

# ============================================================================
# VOCABULARY AND MODEL LOADING - Model-specific
# ============================================================================

if MODEL_TYPE == "scgpt":
    # scGPT vocabulary loading
    if load_model is not None:
        model_dir = Path(load_model)
        model_config_file = model_dir / "args.json"
        model_file = model_dir / "best_model.pt"
        vocab_file = model_dir / "vocab.json"

        vocab = GeneVocab.from_file(vocab_file)
        for s in special_tokens:
            if s not in vocab:
                vocab.append_token(s)

        pert_data.adata.var["id_in_vocab"] = [
            1 if gene in vocab else -1 for gene in pert_data.adata.var["gene_name"]
        ]
        gene_ids_in_vocab = np.array(pert_data.adata.var["id_in_vocab"])
        logger.info(
            f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
            f"in vocabulary of size {len(vocab)}."
        )

        # Load model config
        with open(model_config_file, "r") as f:
            model_configs = json.load(f)
        logger.info(
            f"Resume model from {model_file}, the model args will override the "
            f"config {model_config_file}."
        )
        embsize = model_configs["embsize"]
        nhead = model_configs["nheads"]
        d_hid = model_configs["d_hid"]
        nlayers = model_configs["nlayers"]
        n_layers_cls = model_configs["n_layers_cls"]
        use_fast_transformer = model_configs.get("fast_transformer", use_fast_transformer)
    else:
        vocab = GeneVocab(genes, specials=special_tokens)

    vocab.set_default_index(vocab["<pad>"])
    gene_ids = np.array(
        [vocab[gene] if gene in vocab else vocab["<pad>"] for gene in genes], dtype=int
    )

elif MODEL_TYPE == "deepsc":
    # DeepSC vocabulary loading from CSV
    vocab_dict, id2vocab, pad_token_vocab, pad_value_vocab = build_vocab_from_csv(
        vocab_csv_path, special_tokens=("<pad>", "<cls>", "<mlm>")
    )

    # Build gene IDs for dataset genes using DeepSC's alignment approach
    gene_ids = build_gene_ids_for_dataset(genes, vocab_dict)

    # Create valid gene mask and indices (filter out genes not in vocab)
    if isinstance(gene_ids, torch.Tensor):
        valid_gene_mask = gene_ids != 0
        valid_gene_indices = torch.arange(len(gene_ids))[valid_gene_mask]
        gene_ids_np = gene_ids.cpu().numpy()
        # Convert mask to numpy for counting
        valid_gene_mask_np = valid_gene_mask.cpu().numpy()
    else:
        gene_ids_np = np.array(gene_ids)
        valid_gene_mask_np = gene_ids_np != 0
        valid_gene_mask = valid_gene_mask_np  # Keep numpy version
        valid_gene_indices = torch.from_numpy(np.arange(len(gene_ids_np))[valid_gene_mask_np])

    # Log matching statistics
    matched_genes = np.sum(valid_gene_mask_np)
    logger.info(
        f"match {matched_genes}/{len(gene_ids)} genes "
        f"in vocabulary of size {len(vocab_dict)}."
    )
    logger.info(f"Valid genes (in vocab): {matched_genes}")
    logger.info(f"Invalid genes (not in vocab): {len(gene_ids) - matched_genes}")

    # Store for later use
    gene_ids = gene_ids_np

    # Create a vocab-like object for compatibility
    class VocabWrapper:
        def __init__(self, vocab_dict, pad_value):
            self.vocab_dict = vocab_dict
            self.pad_value = pad_value

        def __len__(self):
            return len(self.vocab_dict)

        def __getitem__(self, key):
            return self.vocab_dict.get(key, self.pad_value)

    vocab = VocabWrapper(vocab_dict, pad_value_vocab)



# ============================================================================
# MODEL INITIALIZATION - Model-specific
# ============================================================================

if MODEL_TYPE == "scgpt":
    # scGPT model initialization
    ntokens = len(vocab)
    model = TransformerGenerator(
        ntokens,
        embsize,
        nhead,
        d_hid,
        nlayers,
        nlayers_cls=n_layers_cls,
        n_cls=1,
        vocab=vocab,
        dropout=dropout,
        pad_token=pad_token,
        pad_value=pad_value,
        pert_pad_id=pert_pad_id,
        use_fast_transformer=use_fast_transformer,
    )

    if load_model is not None:
        # Load pretrained scGPT weights
        pretrained_params = torch.load(model_file, map_location="cpu")
        model = load_pretrained(
            model,
            pretrained_params,
            strict=False,
            prefix=load_param_prefixs,
            verbose=True,
        )
        logger.info("Loaded pretrained scGPT model")

    model.to(device)

elif MODEL_TYPE == "deepsc":
    # DeepSC model initialization
    # Create a namespace object for moe config (supports attribute access)
    from types import SimpleNamespace
    moe_config_obj = SimpleNamespace(
        n_moe_layers=4,
        use_moe_ffn=use_moe_ffn,
        dim=moe_dim,
        moe_inter_dim=moe_inter_dim,
        n_routed_experts=n_routed_experts,
        n_activated_experts=n_activated_experts,
        n_shared_experts=n_shared_experts,
        score_func=moe_score_func,
        route_scale=moe_route_scale,
    )

    model = DeepSC(
        embedding_dim=embedding_dim,
        num_genes=num_genes,
        num_layers=num_layers,
        attn_dropout=attn_dropout,
        ffn_dropout=ffn_dropout,
        num_bins=num_bins,
        alpha=alpha,
        num_heads=num_heads,
        enable_l0=enable_l0,
        enable_mse=enable_mse,
        enable_ce=enable_ce,
        num_layers_ffn=num_layers_ffn,
        use_moe_regressor=use_moe_regressor,
        gene_embedding_participate_til_layer=gene_embedding_participate_til_layer,
        attention_stream=attention_stream,
        cross_attention_architecture=cross_attention_architecture,
        moe=moe_config_obj
    )

    if load_model is not None:
        # Load pretrained DeepSC checkpoint using extract_state_dict utility
        from deepsc.utils import extract_state_dict

        logger.info(f"Loading checkpoint from {load_model}")
        checkpoint = torch.load(load_model, map_location="cpu")
        state_dict = extract_state_dict(checkpoint)

        # Load state dict with strict=False to allow missing keys for new components
        load_info = model.load_state_dict(state_dict, strict=False)

        logger.info(f"Loaded pretrained DeepSC model from {load_model}")
        if load_info.missing_keys:
            logger.info(f"Missing keys ({len(load_info.missing_keys)}): {load_info.missing_keys[:10]}...")
        if load_info.unexpected_keys:
            logger.info(f"Unexpected keys ({len(load_info.unexpected_keys)}): {load_info.unexpected_keys[:10]}...")

    model.to(device)



# ============================================================================
# OPTIMIZER AND LOSS - Common
# ============================================================================
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, schedule_interval, gamma=0.9)
scaler = torch.cuda.amp.GradScaler(enabled=amp)

# Loss functions
if MODEL_TYPE == "scgpt":
    criterion = masked_mse_loss
    criterion_cls = nn.CrossEntropyLoss()
elif MODEL_TYPE == "deepsc":
    criterion = nn.MSELoss()

# Build gene name to column index mapping
name2col = {g: i for i, g in enumerate(genes)}

# Helper function for DeepSC: discretize expression values
def discretize_expression(input_values, num_bins=5):
    """
    Discretize expression values into bins for DeepSC.
    Uses per-sample normalization to adapt to each sample's expression range.
    """
    batch_size = input_values.shape[0]
    discrete_input_bins = torch.zeros_like(input_values, dtype=torch.long)

    for i in range(batch_size):
        row_vals = input_values[i]
        # Skip invalid values (if any)
        valid_mask = row_vals != -1.0
        if valid_mask.any():
            valid_vals = row_vals[valid_mask]
            # Normalize to [0, 1] using this sample's min/max
            min_val = valid_vals.min()
            max_val = valid_vals.max()
            norm = (valid_vals - min_val) / (max_val - min_val + 1e-8)
            # Discretize into bins
            bins = torch.floor(norm * (num_bins - 1)).long()
            # Add 1 to avoid collision with pad token (0)
            bins = torch.clamp(bins, 0, num_bins - 1) + 1
            discrete_input_bins[i][valid_mask] = bins

    return discrete_input_bins


def construct_pert_flags(batch_data, batch_size, n_genes, device):
    """
    Construct perturbation flags from batch_data.pert for new version of gears.
    Uses 3-level flag system:
    - 0: Normal genes
    - 1: Directly perturbed genes (from batch_data.pert)
    - 2: Differentially expressed genes (from batch_data.de_idx)
    """
    pert_flags = torch.zeros(batch_size, n_genes, device=device, dtype=torch.long)

    # Mark directly perturbed genes (pert_flags=1)
    for r, p in enumerate(batch_data.pert):
        for g in p.split("+"):
            if g and g != "ctrl":
                j = name2col.get(g, -1)
                if j != -1:
                    pert_flags[r, j] = 1

    # Mark differentially expressed genes (pert_flags=2)
    for r, de_idx in enumerate(batch_data.de_idx):
        for g_idx in de_idx:
            pert_flags[r, g_idx] = 2

    return pert_flags


def train(model: nn.Module, train_loader: torch.utils.data.DataLoader) -> None:
    """
    Train the model for one epoch.
    """
    model.train()
    total_loss, total_mse = 0.0, 0.0
    start_time = time.time()

    num_batches = len(train_loader)
    for batch, batch_data in enumerate(train_loader):
        batch_size = len(batch_data.y)
        batch_data.to(device)

        # Extract data using new gears format
        x: torch.Tensor = batch_data.x  # (batch_size * n_genes, 2)
        ori_gene_values = x[:, 0].view(batch_size, n_genes)
        target_gene_values = batch_data.y  # (batch_size, n_genes)

        # Construct pert_flags from batch_data.pert instead of x[:, 1]
        pert_flags = construct_pert_flags(batch_data, batch_size, n_genes, device)

        if include_zero_gene in ["all", "batch-wise"]:
            if include_zero_gene == "all":
                if MODEL_TYPE == "scgpt":
                    input_gene_ids = torch.arange(n_genes, device=device, dtype=torch.long)
                elif MODEL_TYPE == "deepsc":
                    # Only use valid genes (genes that exist in vocab)
                    input_gene_ids = valid_gene_indices.to(device)
            else:
                # Get nonzero genes
                nonzero_gene_ids = ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]

                if MODEL_TYPE == "deepsc":
                    # Filter to only include valid genes (in vocab)
                    valid_mask = torch.isin(nonzero_gene_ids, valid_gene_indices.to(device))
                    input_gene_ids = nonzero_gene_ids[valid_mask]
                else:
                    input_gene_ids = nonzero_gene_ids

            # sample input_gene_id
            if len(input_gene_ids) > max_seq_len:
                input_gene_ids = input_gene_ids[torch.randperm(len(input_gene_ids), device=device)[:max_seq_len]]

            input_values = ori_gene_values[:, input_gene_ids]
            input_pert_flags = pert_flags[:, input_gene_ids]
            target_values = target_gene_values[:, input_gene_ids]

            # Map gene IDs to vocabulary IDs
            if MODEL_TYPE == "scgpt":
                mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
            elif MODEL_TYPE == "deepsc":
                # DeepSC: directly use gene_ids array as mapping
                mapped_input_gene_ids = torch.tensor(
                    [gene_ids[i] for i in input_gene_ids.cpu().numpy()],
                    device=device, dtype=torch.long
                )

            mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

            src_key_padding_mask = torch.zeros_like(
                input_values, dtype=torch.bool, device=device
            )

        with torch.cuda.amp.autocast(enabled=amp):
            if MODEL_TYPE == "scgpt":
                # scGPT forward pass
                output_dict = model(
                    mapped_input_gene_ids,
                    input_values,
                    input_pert_flags,
                    src_key_padding_mask=src_key_padding_mask,
                    CLS=CLS,
                    CCE=CCE,
                    MVC=MVC,
                    ECS=ECS,
                )
                output_values = output_dict["mlm_output"]

                masked_positions = torch.ones_like(
                    input_values, dtype=torch.bool
                )  # Use all
                loss = loss_mse = criterion(output_values, target_values, masked_positions)

            elif MODEL_TYPE == "deepsc":
                # DeepSC forward pass
                # Discretize expression values
                discrete_input = discretize_expression(input_values, num_bins=num_bins)

                # Normalize expression values
                normalized_expr = input_values / (input_values.max() + 1e-6)

                # Forward pass
                regression_output, _, _ = model(
                    gene_ids=mapped_input_gene_ids,
                    expression_bin=discrete_input,
                    normalized_expr=normalized_expr,
                    input_pert_flags=input_pert_flags,
                )

                # Compute MSE loss
                loss = loss_mse = criterion(regression_output, target_values)

        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("always")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
                error_if_nonfinite=False if scaler.is_enabled() else True,
            )
            if len(w) > 0:
                logger.warning(
                    f"Found infinite gradient. This may be caused by the gradient "
                    f"scaler. The current scale is {scaler.get_scale()}. This warning "
                    "can be ignored if no longer occurs after autoscaling of the scaler."
                )
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_mse += loss_mse.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            cur_mse = total_mse / log_interval
            logger.info(
                f"| epoch {epoch:3d} | {batch:3d}/{num_batches:3d} batches | "
                f"lr {lr:05.4f} | ms/batch {ms_per_batch:5.2f} | "
                f"loss {cur_loss:5.2f} | mse {cur_mse:5.2f} |"
            )
            total_loss = 0
            total_mse = 0
            start_time = time.time()


def eval_perturb(
    loader: DataLoader, model: nn.Module, device: torch.device
) -> Dict:
    """
    Run model in inference mode using a given data loader
    Supports both scGPT and DeepSC models
    """

    model.eval()
    model.to(device)
    pert_cat = []
    pred = []
    truth = []
    pred_de = []
    truth_de = []
    results = {}

    for itr, batch in enumerate(loader):
        batch.to(device)
        pert_cat.extend(batch.pert)

        with torch.no_grad():
            if MODEL_TYPE == "scgpt":
                # scGPT prediction
                p = model.pred_perturb(
                    batch,
                    include_zero_gene=include_zero_gene,
                    gene_ids=gene_ids,
                    genes=genes,
                )

            elif MODEL_TYPE == "deepsc":
                # DeepSC prediction - manually implement pred_perturb logic
                batch_size = len(batch.y)
                x = batch.x
                ori_gene_values = x[:, 0].view(batch_size, n_genes)
                pert_flags = construct_pert_flags(batch, batch_size, n_genes, device)

                # Use only valid genes (genes that exist in vocab)
                if include_zero_gene == "all":
                    input_gene_ids = valid_gene_indices.to(device)
                else:
                    nonzero_gene_ids = ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
                    # Filter to only include valid genes (in vocab)
                    valid_mask = torch.isin(nonzero_gene_ids, valid_gene_indices.to(device))
                    input_gene_ids = nonzero_gene_ids[valid_mask]

                if len(input_gene_ids) > max_seq_len:
                    input_gene_ids = input_gene_ids[:max_seq_len]

                input_values = ori_gene_values[:, input_gene_ids]
                input_pert_flags = pert_flags[:, input_gene_ids]

                # Map gene IDs
                mapped_input_gene_ids = torch.tensor(
                    [gene_ids[i] for i in input_gene_ids.cpu().numpy()],
                    device=device, dtype=torch.long
                )
                mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

                # Discretize and normalize
                discrete_input = discretize_expression(input_values, num_bins=num_bins)
                normalized_expr = input_values / (input_values.max() + 1e-6)

                # Forward pass
                regression_output, _, _ = model(
                    gene_ids=mapped_input_gene_ids,
                    expression_bin=discrete_input,
                    normalized_expr=normalized_expr,
                    input_pert_flags=input_pert_flags,
                )

                # Reconstruct full prediction: use original values as base, only update predicted genes
                p = ori_gene_values.clone()
                p[:, input_gene_ids] = regression_output

            t = batch.y
            pred.extend(p.cpu())
            truth.extend(t.cpu())

            # Differentially expressed genes
            for itr, de_idx in enumerate(batch.de_idx):
                pred_de.append(p[itr, de_idx])
                truth_de.append(t[itr, de_idx])

    # all genes
    results["pert_cat"] = np.array(pert_cat)
    pred = torch.stack(pred)
    truth = torch.stack(truth)
    results["pred"] = pred.detach().cpu().numpy().astype(np.float64)
    results["truth"] = truth.detach().cpu().numpy().astype(np.float64)

    pred_de = torch.stack(pred_de)
    truth_de = torch.stack(truth_de)
    results["pred_de"] = pred_de.detach().cpu().numpy().astype(np.float64)
    results["truth_de"] = truth_de.detach().cpu().numpy().astype(np.float64)

    return results


best_val_loss = float("inf")
best_val_corr = 0
best_model = None
patience = 0

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train_loader = pert_data.dataloader["train_loader"]
    valid_loader = pert_data.dataloader["val_loader"]

    train(
        model,
        train_loader,
    )

    val_res = eval_perturb(valid_loader, model, device)
    val_metrics = compute_perturbation_metrics(
        val_res, pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"]
    )
    logger.info(f"val_metrics at epoch {epoch}: ")
    logger.info(val_metrics)

    elapsed = time.time() - epoch_start_time
    logger.info(f"| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | ")

    val_score = val_metrics["pearson"]
    if val_score > best_val_corr:
        best_val_corr = val_score
        best_model = copy.deepcopy(model)
        logger.info(f"Best model with score {val_score:5.4f}")
        patience = 0
    else:
        patience += 1
        if patience >= early_stop:
            logger.info(f"Early stop at epoch {epoch}")
            break

    scheduler.step()


torch.save(best_model.state_dict(), save_dir / "best_model.pt")
logger.info("Best model saved")

def predict(
    model: nn.Module, pert_list: List[str], pool_size: Optional[int] = None
) -> Dict:
    """
    Predict the gene expression values for the given perturbations.
    Supports both scGPT and DeepSC models.
    """
    adata = pert_data.adata
    ctrl_adata = adata[adata.obs["condition"] == "ctrl"]
    if pool_size is None:
        pool_size = len(ctrl_adata.obs)
    gene_list = pert_data.gene_names.values.tolist()
    for pert in pert_list:
        for i in pert:
            if i not in gene_list:
                raise ValueError(
                    "The gene is not in the perturbation graph. Please select from GEARS.gene_list!"
                )

    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        results_pred = {}
        for pert in pert_list:
            cell_graphs = create_cell_graph_dataset_for_prediction(
                pert, ctrl_adata, gene_list, device, num_samples=pool_size
            )
            loader = DataLoader(cell_graphs, batch_size=eval_batch_size, shuffle=False)
            preds = []

            for batch_data in loader:
                if MODEL_TYPE == "scgpt":
                    # scGPT prediction
                    pred_gene_values = model.pred_perturb(
                        batch_data, include_zero_gene, gene_ids=gene_ids, amp=amp, genes=genes
                    )

                elif MODEL_TYPE == "deepsc":
                    # DeepSC prediction - use eval_perturb logic
                    batch_size = len(batch_data.y)
                    x = batch_data.x
                    ori_gene_values = x[:, 0].view(batch_size, n_genes)
                    pert_flags = construct_pert_flags(batch_data, batch_size, n_genes, device)

                    # Use only valid genes (genes that exist in vocab)
                    if include_zero_gene == "all":
                        input_gene_ids = valid_gene_indices.to(device)
                    else:
                        nonzero_gene_ids = ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
                        # Filter to only include valid genes (in vocab)
                        valid_mask = torch.isin(nonzero_gene_ids, valid_gene_indices.to(device))
                        input_gene_ids = nonzero_gene_ids[valid_mask]

                    if len(input_gene_ids) > max_seq_len:
                        input_gene_ids = input_gene_ids[:max_seq_len]

                    input_values = ori_gene_values[:, input_gene_ids]
                    input_pert_flags = pert_flags[:, input_gene_ids]

                    mapped_input_gene_ids = torch.tensor(
                        [gene_ids[i] for i in input_gene_ids.cpu().numpy()],
                        device=device, dtype=torch.long
                    )
                    mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)

                    discrete_input = discretize_expression(input_values, num_bins=num_bins)
                    normalized_expr = input_values / (input_values.max() + 1e-6)

                    regression_output, _, _ = model(
                        gene_ids=mapped_input_gene_ids,
                        expression_bin=discrete_input,
                        normalized_expr=normalized_expr,
                        input_pert_flags=input_pert_flags,
                    )

                    # Reconstruct full prediction: use original values as base, only update predicted genes
                    pred_gene_values = ori_gene_values.clone()
                    pred_gene_values[:, input_gene_ids] = regression_output

                preds.append(pred_gene_values)

            preds = torch.cat(preds, dim=0)
            results_pred["_".join(pert)] = np.mean(preds.detach().cpu().numpy(), axis=0)

    return results_pred


def plot_perturbation(
    model: nn.Module, query: str, save_file: str = None, pool_size: int = None
) -> matplotlib.figure.Figure:
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    sns.set_theme(style="ticks", rc={"axes.facecolor": (0, 0, 0, 0)}, font_scale=1.5)

    adata = pert_data.adata
    gene2idx = pert_data.node_map
    cond2name = dict(adata.obs[["condition", "condition_name"]].values)
    gene_raw2id = dict(zip(adata.var.index.values, adata.var.gene_name.values))

    de_idx = [
        gene2idx[gene_raw2id[i]]
        for i in adata.uns["top_non_dropout_de_20"][cond2name[query]]
    ]
    genes = [
        gene_raw2id[i] for i in adata.uns["top_non_dropout_de_20"][cond2name[query]]
    ]
    truth = adata[adata.obs.condition == query].X.toarray()[:, de_idx]
    if query.split("+")[1] == "ctrl":
        pred = predict(model, [[query.split("+")[0]]], pool_size=pool_size)
        pred = pred[query.split("+")[0]][de_idx]
    else:
        pred = predict(model, [query.split("+")], pool_size=pool_size)
        pred = pred["_".join(query.split("+"))][de_idx]
    ctrl_means = adata[adata.obs["condition"] == "ctrl"].to_df().mean()[de_idx].values

    pred = pred - ctrl_means
    truth = truth - ctrl_means

    fig, ax = plt.subplots(figsize=[16.5, 4.5])
    plt.title(query)
    plt.boxplot(truth, showfliers=False, medianprops=dict(linewidth=0))

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

if 'perts_to_plot' in locals() and perts_to_plot:
    logger.info(f"Plotting perturbations: {perts_to_plot}")
    for p in perts_to_plot:
        logger.info(f"Plotting perturbation: {p}")
        plot_perturbation(best_model, p, pool_size=300, save_file=f"{save_dir}/{p}.png")
        logger.info(f"Perturbation plot saved: {p}")
else:
    logger.info("No perturbations to plot (perts_to_plot not defined for this dataset)")

logger.info("Running test evaluation...")
test_loader = pert_data.dataloader["test_loader"]
test_res = eval_perturb(test_loader, best_model, device)
logger.info("Test evaluation completed")

# Optional: Plot scatter plots for prediction vs target values
if plot_scatter:
    logger.info("Plotting scatter plots...")
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Flatten the prediction and target arrays for scatter plot
    pred_values = test_res["pred"].flatten()
    target_values = test_res["truth"].flatten()

    # Create scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(target_values, pred_values, alpha=0.6, s=20)
    plt.plot([target_values.min(), target_values.max()], [target_values.min(), target_values.max()], 'r--', lw=2)
    plt.xlabel('Target Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title('Prediction vs Target Values - All Genes', fontsize=14)
    plt.grid(True, alpha=0.3)

    # Add correlation coefficient to plot
    correlation = np.corrcoef(target_values, pred_values)[0, 1]
    plt.text(0.05, 0.95, f'Correlation: {correlation:.4f}', transform=plt.gca().transAxes,
             fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f"{save_dir}/prediction_vs_target_scatter_all_genes.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Also create scatter plot for differentially expressed genes only
    pred_de_values = test_res["pred_de"].flatten()
    target_de_values = test_res["truth_de"].flatten()

    plt.figure(figsize=(10, 8))
    plt.scatter(target_de_values, pred_de_values, alpha=0.6, s=20, color='orange')
    plt.plot([target_de_values.min(), target_de_values.max()], [target_de_values.min(), target_de_values.max()], 'r--', lw=2)
    plt.xlabel('Target Values (DE genes)', fontsize=12)
    plt.ylabel('Predicted Values (DE genes)', fontsize=12)
    plt.title('Prediction vs Target Values - Differentially Expressed Genes', fontsize=14)
    plt.grid(True, alpha=0.3)

    # Add correlation coefficient to plot
    correlation_de = np.corrcoef(target_de_values, pred_de_values)[0, 1]
    plt.text(0.05, 0.95, f'Correlation: {correlation_de:.4f}', transform=plt.gca().transAxes,
             fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(f"{save_dir}/prediction_vs_target_scatter_de_genes.png", dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Scatter plots saved")
    logger.info(f"Correlation (all genes): {correlation:.4f}")
    logger.info(f"Correlation (DE genes): {correlation_de:.4f}")
else:
    logger.info("Skipping scatter plot generation (plot_scatter=False)")


logger.info("Computing test metrics...")
test_metrics = compute_perturbation_metrics(
    test_res, pert_data.adata[pert_data.adata.obs["condition"] == "ctrl"]
)
logger.info("Test metrics computed successfully")
# save the dicts in json
with open(f"{save_dir}/test_metrics.json", "w") as f:
    json.dump(test_metrics, f)

logger.info("Running deeper_analysis...")
deeper_res = deeper_analysis(pert_data.adata, test_res)
logger.info("deeper_analysis completed")
logger.info("Running non_dropout_analysis...")
non_dropout_res = non_dropout_analysis(pert_data.adata, test_res)
logger.info("non_dropout_analysis completed")

metrics = ["pearson_delta", "pearson_delta_de"]
metrics_non_dropout = [
    "pearson_delta_top20_de_non_dropout",
    "pearson_top20_de_non_dropout",
]
subgroup_analysis = {}
for name in pert_data.subgroup["test_subgroup"].keys():
    subgroup_analysis[name] = {}
    for m in metrics:
        subgroup_analysis[name][m] = []

    for m in metrics_non_dropout:
        subgroup_analysis[name][m] = []

for name, pert_list in pert_data.subgroup["test_subgroup"].items():
    for pert in pert_list:
        for m in metrics:
            subgroup_analysis[name][m].append(deeper_res[pert][m])

        for m in metrics_non_dropout:
            subgroup_analysis[name][m].append(non_dropout_res[pert][m])

for name, result in subgroup_analysis.items():
    for m in result.keys():
        mean_value = np.mean(subgroup_analysis[name][m])
        logger.info("test_" + name + "_" + m + ": " + str(mean_value))
