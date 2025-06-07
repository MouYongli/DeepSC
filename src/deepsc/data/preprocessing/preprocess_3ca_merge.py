import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import argparse


def get_parse():
    parser = argparse.ArgumentParser(
        description="Merge multiple .pth sparse tensors into batch files with up to MAX_ROWS_PER_BATCH rows each."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing input .pth files to merge.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where the merged batch files and metadata CSV will be saved.",
    )
    parser.add_argument(
        "--max_rows_per_batch",
        type=int,
        default=200000,
        help="Maximum number of rows allowed in a single merged batch file. Default: 200000",
    )
    return parser.parse_args()


def save_current_batch(
    rows_list,
    cols_list,
    values_list,
    total_rows_in_this_batch,
    output_dir,
    current_batch_idx,
):
    if not rows_list:
        print(f"Batch {current_batch_idx} is empty. Skipping save.")
        return

    # Determine the number of columns for the sparse tensor
    # If cols_list is empty (e.g. all rows are empty), max_col should be 0, so size is (total_rows, 0)
    # or handle as appropriate (e.g. if schema demands at least 1 col, then max_col = 0, tensor_cols = 1)
    # For COO tensors, if values is empty, indices can be empty, size matters for shape.
    # If there are rows but no non-zero values, cols_list could be empty.
    if not cols_list:  # No non-zero elements in this batch
        max_col_val = -1  # This will result in tensor_cols = 0
    else:
        max_col_val = max(cols_list)

    tensor_cols = max_col_val + 1

    indices = torch.tensor([rows_list, cols_list], dtype=torch.int64)
    values = torch.tensor(values_list, dtype=torch.int32)

    # Ensure size is appropriate even if total_rows_in_this_batch is 0 (handled by outer check)
    # or tensor_cols is 0
    size = (total_rows_in_this_batch, tensor_cols)

    try:
        tensor = torch.sparse_coo_tensor(indices, values, size).coalesce()
    except RuntimeError as e:
        print(f"Error creating sparse tensor for batch {current_batch_idx}: {e}")
        print(
            f"Details: indices shape {indices.shape}, values shape {values.shape}, size {size}"
        )
        print(
            f"Max: {max(rows_list) if rows_list else 'N/A'}, Max col index: {max(cols_list) if cols_list else 'N/A'}"
        )
        print(f"Total rows in this batch: {total_rows_in_this_batch}")
        # Consider saving problematic data for debugging
        # torch.save({'indices': indices, 'values': values, 'size': size},
        # os.path.join(output_dir, f"debug_batch_data_{current_batch_idx}.pth"))
        raise  # Re-raise the exception after printing info

    output_path = os.path.join(output_dir, f"batch_{current_batch_idx}.pth")
    torch.save(tensor, output_path)
    print(f"Saved: {output_path}, shape: {tensor.shape}, nnz: {tensor._nnz()}")


def merge_batches(INPUT_DIR, OUTPUT_DIR, MAX_ROWS_PER_BATCH):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_batch_rows, all_batch_cols, all_batch_values = [], [], []
    num_rows_in_current_batch = 0
    batch_idx = 0
    metadata_records = []

    files_to_process = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".pth")])

    for filename in tqdm(files_to_process):
        tensor_path = os.path.join(INPUT_DIR, filename)
        try:
            sparse_tensor = torch.load(tensor_path).coalesce()
        except Exception as e:
            print(f"Skipping file {filename} due to load error: {e}")
            continue

        # Make sure indices and values are on CPU and are NumPy arrays for processing
        original_indices = sparse_tensor.indices().cpu().numpy()
        original_values = sparse_tensor.values().cpu().numpy()

        # original_rows_from_file are 0-indexed relative to this specific file
        original_rows_from_file = original_indices[0]
        original_cols_from_file = original_indices[1]

        num_total_rows_in_file = sparse_tensor.size(0)

        # If a file is empty (has 0 rows or 0 nnz), skip it
        if num_total_rows_in_file == 0:
            print(
                f"Skipping empty file (0 rows reported by sparse_tensor.size(0)): {filename}"
            )
            continue
        if original_values.size == 0:  # No non-zero elements
            # If we want to preserve the number of rows even if they are all empty
            # This logic becomes more complex if we need to add empty rows to a batch.
            # For now, assume we only care about rows with non-zero elements or just skip if no nnz.
            print(f"Skipping file with no non-zero elements: {filename}")
            continue

        # Sort by row index first, then column index to process in order
        # This is important for the chunking logic if not already sorted.
        # COO tensors are not guaranteed to have sorted indices after coalescing,
        # but coalesce() does sort them. So, this explicit sort might be redundant
        # if coalesce() always sorts, but it's safer.
        sort_order = np.lexsort((original_cols_from_file, original_rows_from_file))
        original_rows_from_file = original_rows_from_file[sort_order]
        original_cols_from_file = original_cols_from_file[sort_order]
        original_values = original_values[sort_order]

        # current_row_offset_in_file is the starting row index in the current file we are processing
        current_row_offset_in_file = 0

        while current_row_offset_in_file < num_total_rows_in_file:
            print(
                f"Processing file: {filename}, current offset: {current_row_offset_in_file}"
            )
            rows_we_can_take_for_current_batch = (
                MAX_ROWS_PER_BATCH - num_rows_in_current_batch
            )
            rows_remaining_in_file = num_total_rows_in_file - current_row_offset_in_file

            num_rows_to_process_from_file_chunk = min(
                rows_we_can_take_for_current_batch, rows_remaining_in_file
            )

            # Define the slice of rows from the original file to process in this iteration
            original_row_start_for_chunk = current_row_offset_in_file
            original_row_end_for_chunk = (
                current_row_offset_in_file + num_rows_to_process_from_file_chunk
            )

            # Find all non-zero elements that fall within this chunk of original rows
            # mask will select elements where original_rows_from_file is in [original_row_start_for_chunk,
            # original_row_end_for_chunk -1]
            element_mask_for_chunk = (
                original_rows_from_file >= original_row_start_for_chunk
            ) & (original_rows_from_file < original_row_end_for_chunk)

            if np.any(element_mask_for_chunk):
                # Get the actual row, col, val data for these elements
                src_rows = original_rows_from_file[element_mask_for_chunk]
                src_cols = original_cols_from_file[element_mask_for_chunk]
                src_vals = original_values[element_mask_for_chunk]

                # Remap original row indices to the new batch's row indices
                # Original row `r` (from src_rows) which is
                # `original_row_start_for_chunk <= r < original_row_end_for_chunk`
                # will become `num_rows_in_current_batch + (r - original_row_start_for_chunk)`
                remapped_rows = (
                    num_rows_in_current_batch
                    + (src_rows - original_row_start_for_chunk)
                ).astype(np.int32)

                all_batch_rows.extend(remapped_rows)
                all_batch_cols.extend(src_cols.astype(np.int32))
                all_batch_values.extend(
                    src_vals
                )  # Assuming src_vals is already desired dtype (int32)

            # Record metadata for this specific segment
            # This segment corresponds to original rows [original_row_start_for_chunk, original_row_end_for_chunk - 1]
            # and will be placed in the current batch at rows
            # [num_rows_in_current_batch, num_rows_in_current_batch + num_rows_to_process_from_file_chunk - 1]
            if (
                num_rows_to_process_from_file_chunk > 0
            ):  # only add metadata if rows were actually taken
                metadata_records.append(
                    {
                        "original_file": filename,
                        "original_file_row_start": original_row_start_for_chunk,
                        "original_file_row_end": original_row_end_for_chunk - 1,
                        "batch_file": f"batch_{batch_idx}.pth",
                        "batch_file_row_start": num_rows_in_current_batch,
                        "batch_file_row_end": num_rows_in_current_batch
                        + num_rows_to_process_from_file_chunk
                        - 1,
                    }
                )

            num_rows_in_current_batch += num_rows_to_process_from_file_chunk
            current_row_offset_in_file += num_rows_to_process_from_file_chunk

            if num_rows_in_current_batch >= MAX_ROWS_PER_BATCH:
                save_current_batch(
                    all_batch_rows,
                    all_batch_cols,
                    all_batch_values,
                    num_rows_in_current_batch,
                    OUTPUT_DIR,
                    batch_idx,
                )
                all_batch_rows, all_batch_cols, all_batch_values = [], [], []
                num_rows_in_current_batch = 0
                batch_idx += 1

    # Save any remaining data in the last batch
    if (
        all_batch_rows or num_rows_in_current_batch > 0
    ):  # Check if there's anything to save
        # If num_rows_in_current_batch > 0 but all_batch_rows is empty, it means we added empty rows.
        # The save_current_batch handles all_batch_rows being empty (no non-zero elements).
        # total_rows_in_this_batch should reflect the allocated row space.
        save_current_batch(
            all_batch_rows,
            all_batch_cols,
            all_batch_values,
            num_rows_in_current_batch,
            OUTPUT_DIR,
            batch_idx,
        )

    # Export metadata
    metadata_df = pd.DataFrame(metadata_records)
    metadata_df.to_csv(
        os.path.join(OUTPUT_DIR, "batch_mapping_summary.csv"), index=False
    )

    print("Processing complete.")
    print(f"Batched files saved in: {OUTPUT_DIR}")
    print(f"Metadata saved to: {os.path.join(OUTPUT_DIR, 'batch_mapping_summary.csv')}")


if __name__ == "__main__":
    args = get_parse()
    INPUT_DIR = args.input_dir
    OUTPUT_DIR = args.output_dir
    MAX_ROWS_PER_BATCH = args.max_rows_per_batch
    merge_batches(INPUT_DIR, OUTPUT_DIR, MAX_ROWS_PER_BATCH)
