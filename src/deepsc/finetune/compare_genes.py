"""
Compare gene names from norman dataset with vocabulary CSV file.
"""
import pandas as pd
import json
from pathlib import Path

# Paths
norman_genes_file = "/home/angli/DeepSC/results/norman_gene_names.txt"
vocab_csv_file = "/home/angli/baseline/DeepSC-117-t86/scripts/data/preprocessing/gene_map_tp10k.csv"
output_dir = "/home/angli/DeepSC/results"

print("="*70)
print("Comparing Norman Dataset Genes with Vocabulary CSV")
print("="*70)

# Read norman genes
print(f"\nReading norman genes from: {norman_genes_file}")
with open(norman_genes_file, "r") as f:
    norman_genes = [line.strip() for line in f if line.strip()]

print(f"Total genes in norman dataset: {len(norman_genes)}")

# Read vocabulary CSV
print(f"\nReading vocabulary CSV from: {vocab_csv_file}")
vocab_df = pd.read_csv(vocab_csv_file)
print(f"Vocabulary CSV shape: {vocab_df.shape}")
print(f"Vocabulary CSV columns: {list(vocab_df.columns)}")

# Get feature_name column
if 'feature_name' in vocab_df.columns:
    vocab_genes = set(vocab_df['feature_name'].tolist())
    print(f"Total genes in vocabulary: {len(vocab_genes)}")
else:
    print("ERROR: 'feature_name' column not found in vocabulary CSV!")
    print(f"Available columns: {list(vocab_df.columns)}")
    exit(1)

# Find genes in norman but not in vocab
missing_genes = [gene for gene in norman_genes if gene not in vocab_genes]
matched_genes = [gene for gene in norman_genes if gene in vocab_genes]

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"Total genes in norman dataset: {len(norman_genes)}")
print(f"Genes found in vocabulary: {len(matched_genes)}")
print(f"Genes NOT found in vocabulary: {len(missing_genes)}")
print(f"Match rate: {len(matched_genes)/len(norman_genes)*100:.2f}%")

# Save missing genes to file
missing_genes_file = Path(output_dir) / "norman_genes_missing_in_vocab.txt"
with open(missing_genes_file, "w") as f:
    for gene in missing_genes:
        f.write(f"{gene}\n")

print(f"\nMissing genes saved to: {missing_genes_file}")

# Save matched genes to file
matched_genes_file = Path(output_dir) / "norman_genes_matched_in_vocab.txt"
with open(matched_genes_file, "w") as f:
    for gene in matched_genes:
        f.write(f"{gene}\n")

print(f"Matched genes saved to: {matched_genes_file}")

# Save summary to JSON
summary = {
    "norman_dataset": {
        "total_genes": len(norman_genes),
        "matched_genes": len(matched_genes),
        "missing_genes": len(missing_genes),
        "match_rate": f"{len(matched_genes)/len(norman_genes)*100:.2f}%"
    },
    "vocabulary": {
        "total_genes": len(vocab_genes),
        "csv_file": vocab_csv_file
    }
}

summary_file = Path(output_dir) / "norman_vocab_comparison.json"
with open(summary_file, "w") as f:
    json.dump(summary, f, indent=2)

print(f"Summary saved to: {summary_file}")

# Print first 20 missing genes
if missing_genes:
    print("\n" + "="*70)
    print(f"First 20 missing genes (out of {len(missing_genes)}):")
    print("="*70)
    for i, gene in enumerate(missing_genes[:20]):
        print(f"{i+1}. {gene}")

# Print first 20 matched genes
if matched_genes:
    print("\n" + "="*70)
    print(f"First 20 matched genes (out of {len(matched_genes)}):")
    print("="*70)
    for i, gene in enumerate(matched_genes[:20]):
        print(f"{i+1}. {gene}")

print("\n" + "="*70)
print("Comparison completed!")
print("="*70)
