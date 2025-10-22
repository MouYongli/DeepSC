"""This script is used to retrieve cell soma ids from cellxgene census."""
import os
import argparse
import cellxgene_census
import pandas as pd

from typing import List
from deepsc.data.download.cellxgene.data_config import VALUE_FILTER, VERSION

def retrieve_soma_idx(query_name: str) -> List[str]:
    """
    Retrieve cell soma_joinid list from cellxgene census based on the query.
    Args:
        query_name: str, the query name
    Returns:
        List[str], the list of cell soma ids
    """
    with cellxgene_census.open_soma(census_version=VERSION) as census:
        obs = census["census_data"]["homo_sapiens"].obs
        cell_metadata = obs.read(
            value_filter=VALUE_FILTER[query_name], column_names=["soma_joinid"]
        )
        cell_metadata = [batch.to_pandas() for batch in cell_metadata]
        cell_metadata = pd.concat(cell_metadata, ignore_index=True)
        return cell_metadata["soma_joinid"].tolist()

def convert_to_file(idx_list: List[str], query_name: str, output_dir: str) -> None:
    """
    Convert the retrieved idx_list to file by query_name.
    Args:
        idx_list: List[str], the list of cell soma ids
        query_name: str, the query name
        output_dir: str, the directory to store the output idx file
    """
    # set up the dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_path = os.path.join(output_dir, f"{query_name}.idx")

    # write to the file
    with open(file_path, "w") as fp:
        for item in idx_list:
            fp.write(f"{item}\n")


def build_soma_idx(query_name: str, output_dir: str) -> None:
    """
    Build the soma idx for cells under query_name.
    Args:
        query_name: str, the query name
        output_dir: str, the directory to store the output idx file
    """
    idx_list = retrieve_soma_idx(query_name)
    convert_to_file(idx_list, query_name, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build soma index list based on query")
    parser.add_argument(
        "--query-name",
        type=str,
        required=True,
        help="query name to build the index",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to store the output idx file",
    )
    args = parser.parse_args()
    build_soma_idx(args.query_name, args.output_dir)
