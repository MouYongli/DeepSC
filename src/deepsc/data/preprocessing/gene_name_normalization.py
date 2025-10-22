import csv
import logging
import os

import argparse
from deepsc.data.preprocessing.get_feature_name_3ca_cxg import get_feature_name_3ca_cxg
from deepsc.utils import setup_logging


def map_genes_to_hgnc(input_gene_file, hgnc_database_file, output_file):
    """Map genes from input file to HGNC database entries.

    Reads the HGNC database to build a case-insensitive mapping of gene symbols
    to their Ensembl IDs and approved names. Then matches genes from the input
    file against this database and writes the results to a CSV output file.

    Args:
        input_gene_file (str): Path to the input file containing gene names,
            one per line.
        hgnc_database_file (str): Path to the HGNC database file in TSV format.
        output_file (str): Path to the output CSV file where matched gene
            mappings will be written.

    Returns:
        None: Results are written to the output file.
    """
    # read HGNC database and build a case-insensitive mapping
    symbol_to_info = {}
    with open(hgnc_database_file, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            # Handle potential None values by using empty string as fallback
            eid = (row["Ensembl gene ID"] or "").strip()
            approved = (row["Approved symbol"] or "").strip()

            # Handle alias symbols with None check
            alias_value = (row["Alias symbol"] or "").strip()
            aliases = alias_value.split(",") if alias_value else []

            # Handle previous symbols with None check
            previous_value = (row["Previous symbol"] or "").strip()
            previous = previous_value.split(",") if previous_value else []
            # Only process if we have a valid approved symbol
            if approved:
                all_names = [approved] + aliases + previous
                for name in all_names:
                    if name.strip():  # Only add non-empty names
                        symbol_to_info[name.strip().upper()] = (eid, approved)

    # match genes from input file against the HGNC database
    matched_rows = []
    with open(input_gene_file, "r") as f:
        for line in f:
            gene = line.strip()
            if not gene:
                logging.info("Empty gene name in this line.")
                continue
            gene_upper = gene.upper()
            row_data = {"feature_name": gene}

            if gene_upper in symbol_to_info:
                eid, approved_name = symbol_to_info[gene_upper]
                row_data["Ensembl id"] = eid
                row_data["Approved Name"] = approved_name
                matched_rows.append(row_data)

    matched_rows.sort(key=lambda x: x["feature_name"].upper())

    # store the intermediate results in a CSV file
    fieldnames = ["feature_name", "Ensembl id", "Approved Name"]
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(matched_rows)


# function to read gene files and merge them based on Approved Name
def read_gene_file(file_path, source):
    """Read gene mapping file and create entries with source information.

    Reads a CSV file containing gene mappings (feature_name, Ensembl id,
    Approved Name) and creates entries with additional metadata indicating
    which dataset the genes occur in.

    Args:
        file_path (str): Path to the CSV file containing gene mappings.
        source (str): Source identifier, either "cxg" for CellxGene or
            "3ca" for 3CA dataset.

    Returns:
        list: List of dictionaries containing gene mapping information
            with occurrence flags for each dataset.
    """
    entries = []
    with open(file_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entry = {
                "feature_name": row["feature_name"].strip(),
                "Ensembl id": row["Ensembl id"].strip(),
                "Approved Name": row["Approved Name"].strip(),
                "occur in cellxgene": source == "cxg",
                "occur in 3ca": source == "3ca",
            }
            entries.append(entry)
    return entries


def merge_gene_mappings(cxg_file, ca3_file, merged_output):
    """Merge gene mappings from CellxGene and 3CA datasets.

    Combines gene mapping files from two sources (CellxGene and 3CA),
    ensuring that only genes with approved names present in both datasets
    are included in the final output. Assigns unique IDs to each approved
    gene name and writes the merged results to a CSV file.

    Args:
        cxg_file (str): Path to the CellxGene gene mapping CSV file.
        ca3_file (str): Path to the 3CA gene mapping CSV file.
        merged_output (str): Path to the output CSV file for merged results.

    Returns:
        None: Results are written to the merged output file.
    """

    entries_cxg = read_gene_file(cxg_file, "cxg")
    entries_3ca = read_gene_file(ca3_file, "3ca")

    # build dictionaries
    cxg_dict = {(e["Approved Name"], e["feature_name"]): e for e in entries_cxg}
    ca3_dict = {(e["Approved Name"], e["feature_name"]): e for e in entries_3ca}

    # collect all Approved Names
    approved_name_set_cxg = set([e["Approved Name"] for e in entries_cxg])
    approved_name_set_3ca = set([e["Approved Name"] for e in entries_3ca])

    merged = {}

    # add entries that appear in both files (exact match on Approved Name and feature_name)
    for key in cxg_dict:
        if key in ca3_dict:
            merged[key] = {
                **cxg_dict[key],
                "occur in cellxgene": True,
                "occur in 3ca": True,
            }

    #  add remaining entries, ensuring Approved Name exists in the other file
    for key, entry in cxg_dict.items():
        if key not in merged and entry["Approved Name"] in approved_name_set_3ca:
            merged[key] = entry

    for key, entry in ca3_dict.items():
        if key not in merged and entry["Approved Name"] in approved_name_set_cxg:
            merged[key] = entry

    final_entries = list(merged.values())
    final_entries.sort(
        key=lambda x: (x["Approved Name"].upper(), x["feature_name"].upper())
    )

    # Assign unique IDs to each unique Approved Name
    approved_to_id = {
        name: idx
        for idx, name in enumerate(sorted({e["Approved Name"] for e in final_entries}))
    }
    for entry in final_entries:
        entry["id"] = approved_to_id[entry["Approved Name"]]

    fieldnames = [
        "feature_name",
        "Ensembl id",
        "Approved Name",
        "occur in cellxgene",
        "occur in 3ca",
        "id",
    ]
    logging.info(f"Writing merged gene mappings to {merged_output}")
    with open(merged_output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_entries)


def process_gene_names(
    cxg_input_path: str,
    ca3_input_path: str,
    gene_map_path: str,
    hgnc_database_path: str,
    output_path: str,
):
    """Process and normalize gene names from both CellxGene and 3CA datasets.

    Orchestrates the complete gene name normalization pipeline by mapping
    genes from both datasets to HGNC database entries and then merging
    the results into a unified gene mapping file.

    Args:
        cxg_input_path (str): Path to the CellxGene input gene names file.
        ca3_input_path (str): Path to the 3CA input gene names file.
        gene_map_path (str): Path to the final merged gene mapping output file.
        hgnc_database_path (str): Path to the HGNC database file.
        output_path (str): Directory path for storing intermediate output files.

    Returns:
        None: Results are written to the specified output files.
    """
    cxg_mapped_output = os.path.join(output_path, "cxg_matched_genes.csv")
    ca3_mapped_output = os.path.join(output_path, "3ca_matched_genes.csv")
    main_log_file = setup_logging("preprocessing", "./logs")
    map_genes_to_hgnc(cxg_input_path, hgnc_database_path, cxg_mapped_output)
    map_genes_to_hgnc(ca3_input_path, hgnc_database_path, ca3_mapped_output)
    merge_gene_mappings(cxg_mapped_output, ca3_mapped_output, gene_map_path)


def gene_name_normalization(
    hgnc_database_path: str,
    output_path: str,
    gene_map_path: str,
):
    """Normalize gene names from CellxGene and 3CA datasets using HGNC database.

    Main entry point for the gene name normalization pipeline. This function
    coordinates the extraction of gene names from both CellxGene and 3CA
    datasets, maps them to standardized HGNC symbols, and produces a unified
    gene mapping file for downstream analysis.

    Args:
        hgnc_database_path (str, optional): Path to the HGNC database file.
        output_path (str, optional): Directory path for intermediate output files.
        gene_map_path (str, optional): Path to the final gene mapping output file.

    Returns:
        None: Results are written to the gene mapping file.
    """
    # use the old cellxgene feature names to generate old gene map
    # cxg_input_path = "/home/angli/DeepSC/scripts/preprocessing/cxg_gene_names.txt"
    cxg_input_path = os.path.join(output_path, "cxg_gene_names.txt")
    ca3_input_path = os.path.join(output_path, "3ca_gene_names.txt")
    process_gene_names(
        cxg_input_path=cxg_input_path,
        ca3_input_path=ca3_input_path,
        hgnc_database_path=hgnc_database_path,
        output_path=output_path,
        gene_map_path=gene_map_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hgnc_database_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--gene_map_path", type=str, required=True)
    parser.add_argument("--tripleca_path", type=str, required=True)
    parser.add_argument("--cellxgene_path", type=str, required=True)
    args = parser.parse_args()

    cxg_feature_names, ca3_feature_names = get_feature_name_3ca_cxg(
        output_path=args.output_path,
        tripleca_path=args.tripleca_path,
        cellxgene_path=args.cellxgene_path,
    )

    print(f"CXG feature names saved to: {cxg_feature_names}")
    print(f"3CA feature names saved to: {ca3_feature_names}")
    logging.info(f"CXG feature names saved to: {cxg_feature_names}")
    logging.info(f"3CA feature names saved to: {ca3_feature_names}")
    # gene_name_normalization()
    gene_name_normalization(
        hgnc_database_path=args.hgnc_database_path,
        output_path=args.output_path,
        gene_map_path=args.gene_map_path,
    )
