from pathlib import Path

import pandas as pd
import scanpy as sc

if __name__ == "__main__":
    csv_path = "target_dataset_with_datasetid.csv"
    df = pd.read_csv(csv_path)
    invalid_dataset_ids = set()
    valid_dataset_ids = set(df["dataset_id"])

    if all(col in df.columns for col in ["path", "Study_uuid", "dataset_id"]):
        for row in df.itertuples(index=False):
            file_path = Path(row.path)
            uuid = row.dataset_id
            path_of_adata = file_path / "transformed_adata.h5ad"

            if not file_path.exists():
                print(f"File {file_path} does not exist")
                invalid_dataset_ids.add(uuid)  # 记录要删除的 id
                continue

            adata = sc.read_h5ad(path_of_adata)

            if adata.shape[0] < 2000:
                print(f"Dataset {uuid} has less than 2000 cells")
                invalid_dataset_ids.add(uuid)
                continue
            if adata.shape[1] < 10000:
                print(f"Dataset {uuid} has less than 10000 genes")
                invalid_dataset_ids.add(uuid)
                continue

            sc.pp.filter_cells(adata, min_genes=200)
            sc.pp.filter_genes(adata, min_cells=3)
            unique_filename = uuid + ".h5ad"
            output_file_path = Path(file_path) / unique_filename
            adata.write(output_file_path)
            print(f"Finished filtering and saving {uuid} to {output_file_path}")

    valid_dataset_ids.difference_update(invalid_dataset_ids)
    df = df[df["dataset_id"].isin(valid_dataset_ids)]  # 仅保留有效的 dataset_id
    df.to_csv("updated_target_dataset_with_datasetid.csv", index=False)
    print(
        f"已更新 CSV 文件并保存至 updated_target_dataset_with_datasetid.csv ，删除了 {len(invalid_dataset_ids)} 个无效的数据集！"
    )
