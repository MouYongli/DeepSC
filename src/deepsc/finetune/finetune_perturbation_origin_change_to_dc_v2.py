"""
Fine-tune perturbation prediction model.
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
import pandas as pd
import matplotlib
from torch import nn
from torch.nn import functional as F
from torch_geometric.loader import DataLoader
from gears import PertData, GEARS
from gears.inference import compute_metrics, deeper_analysis, non_dropout_analysis
from gears.utils import create_cell_graph_dataset_for_prediction

# Import from our self-contained scgpt_pert package
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

from deepsc.models.generation_deepsc_v2.model import DeepSC

matplotlib.rcParams["savefig.transparent"] = False
warnings.filterwarnings("ignore")

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

set_seed(42)

# settings for data processing
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
pad_value = 0  # for padding values
pert_pad_id = 0
include_zero_gene = "all"
max_seq_len = 1536

load_param_prefixs = [
    "encoder",
    "value_encoder",
    "transformer_encoder",
]

# settings for optimizer
lr = 1e-4  # or 1e-4
batch_size = 64
eval_batch_size = 64
epochs = 10
schedule_interval = 1
early_stop = 10

# settings for the model
embsize = 512  # embedding dimension
d_hid = 512  # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 12  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 8  # number of heads in nn.MultiheadAttention
n_layers_cls = 3
dropout = 0  # dropout probability
use_fast_transformer = True  # whether to use fast transformer

# logging
log_interval = 100

# DeepSC pretrained model and vocabulary
load_model = "/home/angli/DeepSC/results/pretraining_1201/latest_checkpoint.ckpt"
vocab_csv_path = "/home/angli/baseline/DeepSC-117-t86/scripts/data/preprocessing/gene_map_tp10k.csv"

# DeepSC model architecture (must match pretrained model)
embedding_dim = 256
num_genes = 35210  # vocabulary size
num_layers = 10
num_heads = 8
num_bins = 5  # number of expression bins (MUST match pretrained model!)
alpha = 0.5  
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


# dataset and evaluation choices
data_name = "norman"
split = "simulation"
if data_name == "norman":
    perts_to_plot = ["SAMD1+ZBTB1"]
elif data_name == "adamson":
    perts_to_plot = ["KCTD16+ctrl"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

save_dir = Path(f"./save/dev_perturb_{data_name}-{time.strftime('%b%d-%H-%M')}/")
save_dir.mkdir(parents=True, exist_ok=True)
print(f"saving to {save_dir}")

add_file_handler(logger, save_dir / "run.log")
logger.info(f"Running on {time.strftime('%Y-%m-%d %H:%M:%S')}")


pert_data = PertData("./archive/data")
pert_data.load(data_name=data_name)
pert_data.prepare_split(split=split, seed=1)
pert_data.get_dataloader(batch_size=batch_size, test_batch_size=eval_batch_size)
# DeepSC gene mapping preparation
# Load DeepSC gene mapping from CSV (id=0 is padding in DeepSC, so CSV ids need +1)
deepsc_gene_map_file = Path("/home/angli/baseline/DeepSC-117-t86/scripts/data/preprocessing/gene_map_tp10k.csv")
if deepsc_gene_map_file.exists():
    logger.info(f"Loading DeepSC gene mapping from {deepsc_gene_map_file}")
    deepsc_gene_df = pd.read_csv(deepsc_gene_map_file)

    # Create mapping: feature_name -> deepsc_id (with +1 offset for padding)
    # Note: Multiple feature_names may map to the same id
    deepsc_gene_to_id = {}
    for _, row in deepsc_gene_df.iterrows():
        feature_name = row["feature_name"]
        csv_id = row["id"]
        # DeepSC uses id=0 as padding, so add 1 to all CSV ids
        deepsc_id = csv_id + 1
        deepsc_gene_to_id[feature_name] = deepsc_id

    genes = pert_data.adata.var["gene_name"].tolist()
    # Map genes in pert_data to DeepSC ids
    pert_data.adata.var["deepsc_id"] = [
        deepsc_gene_to_id.get(gene, 0) for gene in pert_data.adata.var["gene_name"]
    ]

    # Statistics
    matched_genes = np.sum(pert_data.adata.var["deepsc_id"] > 0)
    total_genes = len(pert_data.adata.var["gene_name"])
    unique_deepsc_ids = len(set(deepsc_gene_to_id.values()))

    logger.info(
        f"DeepSC mapping: matched {matched_genes}/{total_genes} genes "
        f"to {unique_deepsc_ids} unique DeepSC ids (id=0 reserved for padding)."
    )

    # Create gene_ids array for DeepSC (parallel to the scGPT gene_ids)
    gene_ids = np.array(pert_data.adata.var["deepsc_id"], dtype=int)
else:
    logger.warning(f"DeepSC gene mapping file not found: {deepsc_gene_map_file}")
    deepsc_gene_to_id = {}
    gene_ids = None

n_genes=len(genes)

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

criterion = masked_mse_loss
criterion_cls = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, schedule_interval, gamma=0.9)
scaler = torch.cuda.amp.GradScaler(enabled=True)

# Build gene name to column index mapping
name2col = {g: i for i, g in enumerate(genes)}


def construct_pert_flags(batch_data, batch_size, n_genes, device):
    """
    Construct perturbation flags from batch_data.pert for new version of gears.
    """
    pert_flags = torch.zeros(batch_size, n_genes, device=device, dtype=torch.long)

    for r, p in enumerate(batch_data.pert):
        for g in p.split("+"):
            if g and g != "ctrl":
                j = name2col.get(g, -1)
                if j != -1:
                    pert_flags[r, j] = 1

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
                input_gene_ids = torch.arange(n_genes, device=device, dtype=torch.long)
            else:
                input_gene_ids = (
                    ori_gene_values.nonzero()[:, 1].flatten().unique().sort()[0]
                )
            # sample input_gene_id
            if len(input_gene_ids) > max_seq_len:
                input_gene_ids = torch.randperm(len(input_gene_ids), device=device)[
                    :max_seq_len
                ]
            input_values = ori_gene_values[:, input_gene_ids]
            input_pert_flags = pert_flags[:, input_gene_ids]
            target_values = target_gene_values[:, input_gene_ids]

            mapped_input_gene_ids = map_raw_id_to_vocab_id(input_gene_ids, gene_ids)
            mapped_input_gene_ids = mapped_input_gene_ids.repeat(batch_size, 1)
            src_key_padding_mask = torch.zeros_like(
                input_values, dtype=torch.bool, device=device
            )

        with torch.cuda.amp.autocast(enabled=True):
            output_diregression_output, gene_emb, expr_emb = model(
                mapped_input_gene_ids,
                input_values,
                input_pert_flags,
            )
            output_values = output_diregression_output

            masked_positions = torch.ones_like(
                input_values, dtype=torch.bool
            )  # Use all
            loss = loss_mse = criterion(output_values, target_values, masked_positions)

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
    loader: DataLoader, model: TransformerGenerator, device: torch.device
) -> Dict:
    """
    Run model in inference mode using a given data loader
    """

    model.eval()
    model.to(device)
    pert_cat = []
    pred = []
    truth = []
    pred_de = []
    truth_de = []
    results = {}
    logvar = []

    for itr, batch in enumerate(loader):
        batch.to(device)
        pert_cat.extend(batch.pert)

        with torch.no_grad():
            p = model.pred_perturb(
                batch,
                include_zero_gene=include_zero_gene,
                gene_ids=gene_ids,
                genes=genes,  # Required for new GEARS format
            )
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
    model: TransformerGenerator, pert_list: List[str], pool_size: Optional[int] = None
) -> Dict:
    """
    Predict the gene expression values for the given perturbations.
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
                pred_gene_values = model.pred_perturb(
                    batch_data, include_zero_gene, gene_ids=gene_ids, amp=True, genes=genes
                )
                preds.append(pred_gene_values)
            preds = torch.cat(preds, dim=0)
            results_pred["_".join(pert)] = np.mean(preds.detach().cpu().numpy(), axis=0)

    return results_pred

logger.info("Running test evaluation...")
test_loader = pert_data.dataloader["test_loader"]
test_res = eval_perturb(test_loader, best_model, device)
logger.info("Test evaluation completed")

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
