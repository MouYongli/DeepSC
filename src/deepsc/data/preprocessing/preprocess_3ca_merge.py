import os

import pandas as pd
import scipy.sparse as sp
from tqdm import tqdm

import argparse


def get_parse():
    parser = argparse.ArgumentParser(
        description="Merge multiple .npz CSR sparse matrices into batch files with up to MAX_ROWS_PER_BATCH rows each."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing input .npz files to merge.",
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
    batch_matrices, total_rows_in_this_batch, output_dir, current_batch_idx
):
    if not batch_matrices or total_rows_in_this_batch == 0:
        print(f"Batch {current_batch_idx} is empty. Skipping save.")
        return

    batch_matrix = sp.vstack(batch_matrices, format="csr")
    output_path = os.path.join(output_dir, f"batch_{current_batch_idx}.npz")
    sp.save_npz(output_path, batch_matrix)
    print(f"Saved: {output_path}, shape: {batch_matrix.shape}, nnz: {batch_matrix.nnz}")


def merge_batches(INPUT_DIR, OUTPUT_DIR, MAX_ROWS_PER_BATCH):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    batch_matrices = []
    num_rows_in_current_batch = 0
    batch_idx = 0
    metadata_records = []

    files_to_process = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".npz")])

    for filename in tqdm(files_to_process):
        matrix_path = os.path.join(INPUT_DIR, filename)
        try:
            csr_matrix = sp.load_npz(matrix_path).tocsr()
        except Exception as e:
            print(f"Skipping file {filename} due to load error: {e}")
            continue

        num_total_rows_in_file = csr_matrix.shape[0]

        if num_total_rows_in_file == 0 or csr_matrix.nnz == 0:
            print(f"Skipping empty file: {filename}")
            continue

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

            row_start = current_row_offset_in_file
            row_end = current_row_offset_in_file + num_rows_to_process_from_file_chunk
            chunk_matrix = csr_matrix[row_start:row_end]

            batch_matrices.append(chunk_matrix)

            # Record metadata
            if num_rows_to_process_from_file_chunk > 0:
                metadata_records.append(
                    {
                        "original_file": filename,
                        "original_file_row_start": row_start,
                        "original_file_row_end": row_end - 1,
                        "batch_file": f"batch_{batch_idx}.npz",
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
                    batch_matrices, num_rows_in_current_batch, OUTPUT_DIR, batch_idx
                )
                batch_matrices = []
                num_rows_in_current_batch = 0
                batch_idx += 1

    # Save any remaining data in the last batch
    if batch_matrices and num_rows_in_current_batch > 0:
        save_current_batch(
            batch_matrices, num_rows_in_current_batch, OUTPUT_DIR, batch_idx
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
