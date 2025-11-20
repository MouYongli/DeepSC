import scanpy as sc
import anndata as ad
import os

def merge_datasets(file_list, output_path, dataset_name):
    """
    Merge multiple h5ad files into one

    Parameters:
    -----------
    file_list: list
        List of file paths to merge
    output_path: str
        Path to save merged file
    dataset_name: str
        Name of the dataset for logging
    """
    print(f"\n{'='*60}")
    print(f"Merging {dataset_name}")
    print(f"{'='*60}")

    # Read all datasets
    adatas = []
    for i, file_path in enumerate(file_list, 1):
        print(f"[{i}/{len(file_list)}] Reading: {file_path}")
        adata = sc.read_h5ad(file_path)
        print(f"  Shape: {adata.shape}")

        # Add split information to obs
        split_name = os.path.basename(file_path).replace('.h5ad', '').split('_')[-1]
        adata.obs['split'] = split_name

        adatas.append(adata)

    # Concatenate
    print(f"\nConcatenating {len(adatas)} datasets...")
    merged = ad.concat(adatas, join='outer', merge='same')

    print(f"Merged shape: {merged.shape}")
    print(f"Split distribution:")
    print(merged.obs['split'].value_counts())

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"\nSaving to: {output_path}")
    merged.write(output_path)
    print(f"Done!\n")

    return merged


def main():
    base_dir = "/home/angli/scBERT/data/preprocessed"

    # Merge Segerstolpe datasets
    segerstolpe_files = [
        f"{base_dir}/segerstolpe/Segerstolpe_train.h5ad",
        f"{base_dir}/segerstolpe/Segerstolpe_valid.h5ad",
        f"{base_dir}/segerstolpe/Segerstolpe_test.h5ad"
    ]
    segerstolpe_output = f"{base_dir}/segerstolpe/Segerstolpe_merged.h5ad"

    merge_datasets(
        segerstolpe_files,
        segerstolpe_output,
        "Segerstolpe (train + valid + test)"
    )

    # Merge human pancreas datasets
    hpancreas_files = [
        f"{base_dir}/human_pancreas/hPancreas_train.h5ad",
        f"{base_dir}/human_pancreas/hPancreas_test.h5ad"
    ]
    hpancreas_output = f"{base_dir}/human_pancreas/hPancreas_merged.h5ad"

    merge_datasets(
        hpancreas_files,
        hpancreas_output,
        "hPancreas (train + test)"
    )

    print("\n" + "="*60)
    print("All datasets merged successfully!")
    print("="*60)
    print(f"\nMerged files:")
    print(f"  1. {segerstolpe_output}")
    print(f"  2. {hpancreas_output}")


if __name__ == "__main__":
    main()
