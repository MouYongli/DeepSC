from pathlib import Path

import pandas as pd
import scanpy as sc
import scipy.io
import scipy.sparse
from utils import path_of_file

if __name__ == "__main__":
    csv_path = "target_dataset_files.csv"
    df = pd.read_csv(csv_path)

    if all(col in df.columns for col in ["path", "filename"]):
        for row in df.itertuples(index=False):
            file_path = Path(row.path)
            file_name = row.filename
            path_of_mtx_file = file_path / file_name
            print(file_path)
            if not file_path.exists():
                print(f"File {file_path} does not exist")
                continue

            files_in_directory = [f.name for f in file_path.iterdir() if f.is_file()]
            parent_folder = file_path.parent
            files_in_parent_directory = [
                f.name for f in parent_folder.iterdir() if f.is_file()
            ]
            lower_files = [f.lower() for f in files_in_directory]
            lower_files_in_parent_directory = [
                f.lower() for f in files_in_parent_directory
            ]
            mtx_files = [f for f in files_in_directory if f.endswith(".mtx")]
            if not mtx_files:
                print("Not found matrix file")
                continue

            path_of_gene_file = path_of_file(file_path, "gene")
            path_of_cell_file = path_of_file(file_path, "cell")

            genes = pd.read_csv(path_of_gene_file, header=None, names=["gene_name"])
            genes.index = range(1, len(genes) + 1)
            cells = pd.read_csv(path_of_cell_file)
            cells.index = range(1, len(cells) + 1)
            X = scipy.io.mmread(path_of_mtx_file)
            X = X.transpose()
            X = scipy.sparse.csr_matrix(X)

            adata = sc.AnnData(X=X, obs=cells, var=genes)
            print(type(adata.X))

            adata.obs.index = adata.obs.index.astype(str)
            adata.obs = adata.obs.astype(str)

            adata.obs = adata.obs.fillna("")

            adata.write("transformed_adata.h5ad")

            adata.write(file_path / "transformed_adata.h5ad")

    else:
        print("CSV 文件中没有 'path' 列")
