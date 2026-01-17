from gears import PertData
from tqdm import tqdm

if __name__ == "__main__":
    data_name = "adamson"
    split = "simulation"
    pert_data = PertData("./data")  # TODO: change to the data path
    pert_data.load(data_name=data_name)
    pert_data.prepare_split(split=split, seed=1)
    pert_data.get_dataloader(batch_size=64, test_batch_size=64)
    original_genes = pert_data.adata.var["gene_name"].tolist()
    num_genes = len(original_genes)

    train_loader = pert_data.dataloader["train_loader"]
    valid_loader = pert_data.dataloader["val_loader"]
    node_map = pert_data.node_map

    adata = pert_data.adata
    gene2idx = pert_data.node_map
    cond2name = dict(adata.obs[["condition", "condition_name"]].values)
    gene_raw2id = dict(zip(adata.var.index.values, adata.var.gene_name.values))
    data_iter = tqdm(
        train_loader,
        desc="[Finetune Cell Type Annotation]",
        ncols=150,
        position=1,
    )
    for index, batch in enumerate(data_iter):
        ori_gene_values = batch.x[:, 0].view(64, 5060)
        print(f"Batch {index}: ori_gene_values.shape = {ori_gene_values.shape}")
        print(f"Batch {index}: batch.y.shape = {batch.y.shape}")

        # Check positions where ori_gene_values is 0 but batch.y is not 0
        ori_zero_mask = ori_gene_values == 0  # Positions where ori_gene_values is 0
        batch_nonzero_mask = batch.y != 0  # Positions where batch.y is not 0

        # Find positions satisfying both conditions: ori_gene_values is 0 AND batch.y is not 0
        mismatch_mask = ori_zero_mask & batch_nonzero_mask
        num_mismatches = mismatch_mask.sum().item()

        print(
            f"Batch {index}: Found {num_mismatches} positions where ori_gene_values is 0 but batch.y is not 0"
        )

        # If there are mismatched positions, print some additional information
        if num_mismatches > 0:
            total_positions = ori_gene_values.numel()
            percentage = (num_mismatches / total_positions) * 100
            print(
                f"Batch {index}: Mismatch ratio: {percentage:.4f}% ({num_mismatches}/{total_positions})"
            )

            # Print coordinates and values of the first few mismatched positions
            mismatch_indices = mismatch_mask.nonzero()[
                : min(5, num_mismatches)
            ]  # Show at most 5
            print(f"Batch {index}: First few mismatched positions (row, col):")
            for i, (row, col) in enumerate(mismatch_indices):
                ori_val = ori_gene_values[row, col].item()
                batch_val = batch.y[row, col].item()
                print(
                    f"  [{row.item()}, {col.item()}]: ori_gene_values={ori_val}, batch.y={batch_val}"
                )

        print("-" * 50)
