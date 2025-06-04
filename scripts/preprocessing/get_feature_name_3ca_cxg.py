import os

import scanpy as sc

from .config import CELLXGENE_DATASET_PATH, TRIPLECA_DATASET_PATH


def get_feature_name_3ca_cxg():
    """
    This function extracts feature names from .h5ad files in the 3CA dataset and the CellxGene dataset,
    and saves them to text files.
    """
    # 设置基础路径
    tripleca_path = TRIPLECA_DATASET_PATH
    all_feature_names = set()

    # 遍历子文件夹
    for root, dirs, files in os.walk(tripleca_path):
        for file in files:
            if file.endswith(".h5ad"):
                h5ad_path = os.path.join(root, file)
                try:
                    adata = sc.read_h5ad(h5ad_path)
                    if "feature_name" in adata.var.columns:
                        feature_names = set(adata.var["feature_name"])
                        all_feature_names.update(feature_names)
                        print(f"Loaded {len(feature_names)} feature_names from {file}")
                    else:
                        print(f"[Warning] 'feature_name' not found in {file}")
                except Exception as e:
                    print(f"[Error] Failed to read {h5ad_path}: {e}")

    # 读取 CellxGene 的 .h5ad 文件
    adata_cxg = sc.read_h5ad(CELLXGENE_DATASET_PATH)

    # 提取 var 中的 feature_name 列
    cellxgene_feature_names = adata_cxg.var["feature_name"].tolist()

    # 保存为文件
    output_3ca = "/home/angli/DeepSC/scripts/preprocessing/feature_name_3ca.txt"
    output_cxg = "/home/angli/DeepSC/scripts/preprocessing/feature_name_cxg.txt"
    with open(output_3ca, "w") as f:
        for feature in sorted(all_feature_names):
            f.write(feature + "\n")

    with open(output_cxg, "w") as f:
        for feature in sorted(cellxgene_feature_names):
            f.write(feature + "\n")

    return output_3ca, output_cxg
