import os
from typing import List

import cellxgene_census

import argparse
from deepsc.data.download.cellxgene.data_config import VERSION


def define_partition(partition_idx, id_list, partition_size) -> List[str]:
    """Return a sublist (partition) of indices from ``id_list``.

    The partition is computed by slicing ``id_list`` with
    ``start = partition_idx * partition_size`` and
    ``end = start + partition_size``.

    Args:
        partition_idx: Zero-based partition index.
        id_list: Full list of obs indices.
        partition_size: Maximum number of indices per partition.

    Returns:
        A list of obs indices for the requested partition. May be empty if
        ``partition_idx`` is out of range.
    """
    i = partition_idx * partition_size
    return id_list[i : i + partition_size]


def loadToList(query_name, soma_id_dir) -> List[int]:
    """Load obs indices from a plain-text file into a list of ints.

    The file must be located at ``{soma_id_dir}/{query_name}.idx``, with one
    integer per line.

    Args:
        query_name: Name of the query (used as filename prefix).
        soma_id_dir: Directory containing the ``.idx`` file.

    Returns:
        A list of integers parsed from the index file.

    Raises:
        FileNotFoundError: If the ``.idx`` file does not exist.
        ValueError: If any line is not an integer.
    """
    file_path = os.path.join(soma_id_dir, f"{query_name}.idx")
    with open(file_path, "r") as fp:
        idx_list = fp.readlines()
    idx_list = [int(x.strip()) for x in idx_list]
    return idx_list


def download_partition(
    partition_idx, query_name, output_dir, index_dir, partition_size
):
    """Download one partition of a query as an ``.h5ad`` file.

    Workflow:
      1) Load obs indices from the ``.idx`` file.
      2) Slice to the requested partition.
      3) Open census at the configured ``VERSION``.
      4) Fetch an AnnData with ``obs_coords`` set to the partition indices.
      5) Save to ``{output_dir}/{query_name}/partition_{partition_idx}.h5ad``.

    Args:
        partition_idx: Zero-based partition index to download.
        query_name: Query name (used for file discovery and output dir).
        output_dir: Directory where the ``.h5ad`` file will be saved.
        index_dir: Directory containing ``{query_name}.idx``.
        partition_size: Maximum number of indices per partition.

    Returns:
        The file system path to the written ``.h5ad`` file.

    Raises:
        FileNotFoundError: If ``{query_name}.idx`` is missing.
        OSError: If the output directory cannot be created or the file
            cannot be written.
    """
    # Define the index partition.
    id_list = loadToList(query_name, index_dir)
    id_partition = define_partition(partition_idx, id_list, partition_size)
    with cellxgene_census.open_soma(census_version=VERSION) as census:
        adata = cellxgene_census.get_anndata(
            census,
            organism="Homo sapiens",
            obs_coords=id_partition,
        )
    # Read the subset from census and write to disk.
    query_dir = os.path.join(output_dir, query_name)
    if not os.path.exists(query_dir):
        os.makedirs(query_dir)
    query_adata_path = os.path.join(query_dir, f"partition_{partition_idx}.h5ad")
    adata.write_h5ad(query_adata_path)
    return query_adata_path


def del_partition(partition_idx, query_name, output_dir, index_dir, partition_size):
    """Delete a previously written partition file.

    Args:
        partition_idx: Partition index of the file to delete.
        query_name: Query name (output subdirectory).
        output_dir: Root directory for outputs.
        index_dir: Unused; kept for API parity with ``download_partition``.
        partition_size: Unused; kept for API parity with ``download_partition``.

    Raises:
        FileNotFoundError: If the target file does not exist.
        OSError: If deletion fails for other reasons.
    """
    query_dir = os.path.join(output_dir, query_name)
    query_adata_path = os.path.join(query_dir, f"partition_{partition_idx}.h5ad")
    os.remove(query_adata_path)


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Download a given partition of a query as an .h5ad file."
    )
    parser.add_argument(
        "--query-name",
        type=str,
        required=True,
        help="Query name (also the prefix of the .idx file).",
    )
    parser.add_argument(
        "--partition-idx",
        type=int,
        required=True,
        help="Zero-based partition index to download.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to store the output .h5ad file.",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        required=True,
        help="Directory where the {query_name}.idx file is located.",
    )
    parser.add_argument(
        "--max-partition-size",
        type=int,
        required=True,
        help="Maximum number of indices (rows) in each partition.",
    )
    return parser


if __name__ == "__main__":
    arg_parser = _build_arg_parser()
    args = arg_parser.parse_args()

    download_partition(
        partition_idx=args.partition_idx,
        query_name=args.query_name,
        output_dir=args.output_dir,
        index_dir=args.index_dir,
        partition_size=args.max_partition_size,
    )
