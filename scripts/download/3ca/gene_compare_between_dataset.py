from pathlib import Path

import pandas as pd
from utils import path_of_file


def merge_gene_files(file_paths):
    unique_genes = set()

    for file_path in file_paths:
        try:
            with open(file_path, "r") as f:
                genes = f.read().splitlines()
                unique_genes.update(genes)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    return sorted(unique_genes)


if __name__ == "__main__":
    csv_path = "target_dataset_files.csv"
    df = pd.read_csv(csv_path)

    if all(col in df.columns for col in ["path", "filename"]):
        file_paths = []
        for row in df.itertuples(index=False):
            file_path = Path(row.path)
            file_name = row.filename
            path_of_gene_file = path_of_file(file_path, "gene")
            if path_of_gene_file.exists():
                file_paths.append(path_of_gene_file)

        merged_genes = merge_gene_files(file_paths)

        with open("merged_genes.txt", "w") as f:
            for gene in merged_genes:
                f.write(f"{gene}\n")
