import csv
import os
import random
from typing import Dict, List, Tuple

import scipy.sparse

import argparse


def collect_file_samples(original_dir: str) -> List[Tuple[str, int]]:
    """
    Collect sample counts from NPZ files in the specified directory.

    Args:
        original_dir (str): Directory containing original NPZ files

    Returns:
        List[Tuple[str, int]]: List of tuples containing (file_path, sample_count)
    """
    file_sample_tuples = []
    for dirpath, _, filenames in os.walk(original_dir):
        for fname in filenames:
            if fname.endswith(".npz"):
                path = os.path.join(dirpath, fname)
                print("check path", path)
                try:
                    matrix = scipy.sparse.load_npz(path)
                    file_sample_tuples.append((path, matrix.shape[0]))
                except Exception as e:
                    print(f"Failed loading {path}: {e}")
    return file_sample_tuples


def generate_global_indices(
    file_sample_tuples: List[Tuple[str, int]],
) -> List[Tuple[str, int]]:
    """
    Generate global indices for all samples across all files.

    Args:
        file_sample_tuples (List[Tuple[str, int]]): List of (file_path, sample_count) tuples

    Returns:
        List[Tuple[str, int]]: List of (file_path, row_index) tuples for all samples
    """
    all_indices = []
    for path, nrows in file_sample_tuples:
        for i in range(nrows):
            all_indices.append((path, i))
    return all_indices


def shuffle_and_allocate_indices(
    all_indices: List[Tuple[str, int]], target_chunk_size: int = 200_000, seed: int = 42
) -> Dict[str, Dict[int, List[int]]]:
    """
    Shuffle global indices and allocate them to chunks.

    Args:
        all_indices (List[Tuple[str, int]]): List of (file_path, row_index) tuples
        target_chunk_size (int, optional): Target size for each chunk. Defaults to 200_000.
        seed (int, optional): Random seed for reproducible shuffling. Defaults to 42.

    Returns:
        Dict[str, Dict[int, List[int]]]: Allocation structure as
        {file_path: {chunk_id: [row_ids]}}
    """
    random.seed(seed)
    random.shuffle(all_indices)

    alloc = {}  # {file_path: {chunk_id: [row_ids]}}
    for i, (path, row) in enumerate(all_indices):
        chunk_id = i // target_chunk_size
        if path not in alloc:
            alloc[path] = {}
        if chunk_id not in alloc[path]:
            alloc[path][chunk_id] = []
        alloc[path][chunk_id].append(row)

    return alloc


def save_allocation_plan(
    alloc: Dict[str, Dict[int, List[int]]], shuffel_plan_path: str
) -> None:
    """
    Save the allocation plan to a CSV file.

    Args:
        alloc (Dict[str, Dict[int, List[int]]]): Allocation structure as
        {file_path: {chunk_id: [row_ids]}}
        shuffel_plan_path (str): Output path for the shuffled plan CSV file

    Returns:
        None
    """
    with open(shuffel_plan_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source_file", "chunk_id", "rows"])  # header
        for path, chunk_map in alloc.items():
            for chunk_id, row_list in chunk_map.items():
                row_str = " ".join(map(str, row_list))
                writer.writerow([path, chunk_id, row_str])


def main():
    """
    Main function to generate shuffled dataset index.

    Parses command line arguments and orchestrates the dataset shuffling process:
    1. Collects sample counts from NPZ files
    2. Generates global indices for all samples
    3. Shuffles indices and allocates them to chunks
    4. Saves the allocation plan to a CSV file
    """
    parser = argparse.ArgumentParser(description="Generate shuffled dataset index")
    parser.add_argument(
        "--original_dir", required=True, help="Directory containing original NPZ files"
    )
    parser.add_argument(
        "--shuffel_plan_path",
        required=True,
        help="Output path for the shuffled plan CSV file",
    )
    args = parser.parse_args()

    original_dir = args.original_dir
    shuffel_plan_path = args.shuffel_plan_path
    target_chunk_size = 200_000

    # Step 1: Collect file samples
    file_sample_tuples = collect_file_samples(original_dir)

    # Step 2: Generate global indices
    all_indices = generate_global_indices(file_sample_tuples)

    # Step 3: Shuffle and allocate indices
    alloc = shuffle_and_allocate_indices(all_indices, target_chunk_size)

    # Step 4: Save allocation plan
    save_allocation_plan(alloc, shuffel_plan_path)


if __name__ == "__main__":
    main()
