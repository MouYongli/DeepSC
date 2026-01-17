import os

import anndata as ad


def get_feature_name_3ca_cxg(output_path: str, tripleca_path: str, cellxgene_path: str):
    """
    This function extracts feature names from .h5ad files in the 3CA dataset and the CellxGene dataset,
    and saves them to text files.
    """
    # Set base path
    all_feature_names = set()

    # Traverse subfolders
    for root, dirs, files in os.walk(tripleca_path):
        for file in files:
            if file.endswith(".h5ad"):
                h5ad_path = os.path.join(root, file)
                try:
                    adata = ad.read_h5ad(h5ad_path)
                    if "feature_name" in adata.var.columns:
                        feature_names = set(adata.var["feature_name"])
                        all_feature_names.update(feature_names)
                        print(f"Loaded {len(feature_names)} feature_names from {file}")
                    else:
                        print(f"[Warning] 'feature_name' not found in {file}")
                except Exception as e:
                    print(f"[Error] Failed to read {h5ad_path}: {e}")

    # Read CellxGene .h5ad files
    cxg_path = os.path.join(cellxgene_path, "heart", "partition_0.h5ad")
    adata_cxg = ad.read_h5ad(cxg_path)

    # Extract feature_name column from var
    cellxgene_feature_names = adata_cxg.var["feature_name"].tolist()

    output_3ca = os.path.join(output_path, "3ca_gene_names.txt")
    output_cxg = os.path.join(output_path, "cxg_gene_names.txt")

    with open(output_3ca, "w") as f:
        for feature in sorted(all_feature_names):
            f.write(feature + "\n")

    with open(output_cxg, "w") as f:
        for feature in sorted(cellxgene_feature_names):
            f.write(feature + "\n")

    return output_3ca, output_cxg
