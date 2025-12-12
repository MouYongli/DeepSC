"""
Extract perturbation genes from norman dataset and compare with vocabulary.
"""
import os
import json
import pandas as pd
from pathlib import Path
from gears import PertData

# Configuration
data_name = "norman"
split = "simulation"
data_path = "./data"
vocab_csv_file = "/home/angli/baseline/DeepSC-117-t86/scripts/data/preprocessing/gene_map_tp10k.csv"
output_dir = "/home/angli/DeepSC/results"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

print("="*70)
print("Extracting Perturbation Genes from Norman Dataset")
print("="*70)

# Load perturbation data
print(f"\nLoading {data_name} dataset from {data_path}...")
pert_data = PertData(data_path)
pert_data.load(data_name=data_name)
pert_data.prepare_split(split=split, seed=1)
pert_data.get_dataloader(batch_size=64, test_batch_size=64)

# Extract all perturbations and split by '+'
print(f"\nExtracting perturbation genes...")
all_pert_genes = set()
all_perturbations = []

# Iterate through train, val, test loaders
for loader_name, loader in [
    ("train", pert_data.dataloader["train_loader"]),
    ("val", pert_data.dataloader["val_loader"]),
    ("test", pert_data.dataloader["test_loader"])
]:
    print(f"\nProcessing {loader_name} loader...")
    for batch_idx, batch_data in enumerate(loader):
        # batch_data.pert is a list of perturbation strings
        for pert_str in batch_data.pert:
            all_perturbations.append(pert_str)
            # Split by '+' to get individual genes
            genes = pert_str.split('+')
            for gene in genes:
                gene = gene.strip()
                if gene and gene != 'ctrl':  # Exclude empty strings and 'ctrl'
                    all_pert_genes.add(gene)

print(f"\nTotal perturbations: {len(all_perturbations)}")
print(f"Unique perturbations: {len(set(all_perturbations))}")
print(f"Unique perturbation genes (excluding 'ctrl'): {len(all_pert_genes)}")

# Read vocabulary CSV
print(f"\nReading vocabulary CSV from: {vocab_csv_file}")
vocab_df = pd.read_csv(vocab_csv_file)
vocab_genes = set(vocab_df['feature_name'].tolist())
print(f"Total genes in vocabulary: {len(vocab_genes)}")

# Find genes in perturbations but not in vocab
missing_pert_genes = [gene for gene in sorted(all_pert_genes) if gene not in vocab_genes]
matched_pert_genes = [gene for gene in sorted(all_pert_genes) if gene in vocab_genes]

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"Total unique perturbation genes: {len(all_pert_genes)}")
print(f"Perturbation genes found in vocabulary: {len(matched_pert_genes)}")
print(f"Perturbation genes NOT found in vocabulary: {len(missing_pert_genes)}")
print(f"Match rate: {len(matched_pert_genes)/len(all_pert_genes)*100:.2f}%")

# Save all unique perturbation genes
all_pert_genes_file = Path(output_dir) / "norman_perturbation_genes_unique.txt"
with open(all_pert_genes_file, "w") as f:
    for gene in sorted(all_pert_genes):
        f.write(f"{gene}\n")
print(f"\nAll unique perturbation genes saved to: {all_pert_genes_file}")

# Save missing perturbation genes
if missing_pert_genes:
    missing_file = Path(output_dir) / "norman_perturbation_genes_missing_in_vocab.txt"
    with open(missing_file, "w") as f:
        for gene in missing_pert_genes:
            f.write(f"{gene}\n")
    print(f"Missing perturbation genes saved to: {missing_file}")

# Save matched perturbation genes
if matched_pert_genes:
    matched_file = Path(output_dir) / "norman_perturbation_genes_matched_in_vocab.txt"
    with open(matched_file, "w") as f:
        for gene in matched_pert_genes:
            f.write(f"{gene}\n")
    print(f"Matched perturbation genes saved to: {matched_file}")

# Save summary to JSON
summary = {
    "dataset": data_name,
    "split": split,
    "total_perturbations": len(all_perturbations),
    "unique_perturbations": len(set(all_perturbations)),
    "perturbation_genes": {
        "total_unique": len(all_pert_genes),
        "matched": len(matched_pert_genes),
        "missing": len(missing_pert_genes),
        "match_rate": f"{len(matched_pert_genes)/len(all_pert_genes)*100:.2f}%"
    },
    "vocabulary": {
        "total_genes": len(vocab_genes),
        "csv_file": vocab_csv_file
    }
}

summary_file = Path(output_dir) / "norman_perturbation_genes_comparison.json"
with open(summary_file, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Summary saved to: {summary_file}")

# Print all perturbation genes (sorted)
print("\n" + "="*70)
print(f"All {len(all_pert_genes)} unique perturbation genes:")
print("="*70)
for i, gene in enumerate(sorted(all_pert_genes), 1):
    in_vocab = "✓" if gene in vocab_genes else "✗"
    print(f"{i:3d}. {gene:20s} {in_vocab}")

# Print missing genes
if missing_pert_genes:
    print("\n" + "="*70)
    print(f"Missing perturbation genes ({len(missing_pert_genes)}):")
    print("="*70)
    for i, gene in enumerate(missing_pert_genes, 1):
        print(f"{i}. {gene}")

# Sample perturbations
print("\n" + "="*70)
print("Sample perturbations (first 20 unique):")
print("="*70)
unique_perts = sorted(set(all_perturbations))
for i, pert in enumerate(unique_perts[:20], 1):
    genes = pert.split('+')
    genes_status = []
    for g in genes:
        g = g.strip()
        if g == 'ctrl':
            genes_status.append(f"{g}")
        elif g in vocab_genes:
            genes_status.append(f"{g}✓")
        else:
            genes_status.append(f"{g}✗")
    print(f"{i:3d}. {pert:25s} -> {' + '.join(genes_status)}")

print("\n" + "="*70)
print("Extraction and comparison completed!")
print("="*70)
