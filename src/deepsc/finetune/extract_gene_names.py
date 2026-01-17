"""
Extract and save gene names used in perturbation prediction dataset.
This script loads the GEARS dataset and extracts all gene names.
"""
import os
import json
import numpy as np
from pathlib import Path
from gears import PertData

# Configuration
data_name = "norman"
split = "simulation"
data_path = "./data"
output_dir = "/home/angli/DeepSC/results"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

print(f"Loading {data_name} dataset from {data_path}...")

# Load perturbation data
pert_data = PertData(data_path)
pert_data.load(data_name=data_name)
pert_data.prepare_split(split=split, seed=1)

# Extract gene names
genes = pert_data.adata.var["gene_name"].tolist()
n_genes = len(genes)

print(f"Total genes in dataset: {n_genes}")

# Save gene names to text file (one gene per line)
gene_names_txt = Path(output_dir) / f"{data_name}_gene_names.txt"
with open(gene_names_txt, "w") as f:
    for gene in genes:
        f.write(f"{gene}\n")

print(f"Saved gene names to: {gene_names_txt}")

# Save gene names to JSON file
gene_names_json = Path(output_dir) / f"{data_name}_gene_names.json"
with open(gene_names_json, "w") as f:
    json.dump({
        "dataset": data_name,
        "split": split,
        "num_genes": n_genes,
        "genes": genes
    }, f, indent=2)

print(f"Saved gene names (JSON) to: {gene_names_json}")

# Extract additional dataset information
print("\n" + "="*60)
print("Dataset Information:")
print("="*60)
print(f"Dataset name: {data_name}")
print(f"Split: {split}")
print(f"Number of genes: {n_genes}")
print(f"Number of cells: {pert_data.adata.n_obs}")

# Extract unique perturbations
unique_perts = set()
for pert in pert_data.adata.obs["condition"]:
    unique_perts.add(pert)

print(f"Number of unique perturbations: {len(unique_perts)}")

# Save perturbation information
pert_info = {
    "dataset": data_name,
    "split": split,
    "num_cells": int(pert_data.adata.n_obs),
    "num_genes": n_genes,
    "num_unique_perturbations": len(unique_perts),
    "perturbations": sorted(list(unique_perts))
}

pert_info_json = Path(output_dir) / f"{data_name}_perturbation_info.json"
with open(pert_info_json, "w") as f:
    json.dump(pert_info, f, indent=2)

print(f"Saved perturbation info to: {pert_info_json}")

# Print first 10 genes as sample
print("\n" + "="*60)
print("Sample genes (first 10):")
print("="*60)
for i, gene in enumerate(genes[:10]):
    print(f"{i+1}. {gene}")

# Print some perturbation examples
print("\n" + "="*60)
print("Sample perturbations (first 10):")
print("="*60)
for i, pert in enumerate(sorted(list(unique_perts))[:10]):
    print(f"{i+1}. {pert}")

print("\n" + "="*60)
print("Extraction completed successfully!")
print("="*60)
