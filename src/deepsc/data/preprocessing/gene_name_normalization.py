import csv
import logging
import os

from deepsc.data.preprocessing.config import GENE_MAP_PATH, HGNC_DATABASE, SCRIPTS_ROOT
from deepsc.data.preprocessing.get_feature_name_3ca_cxg import get_feature_name_3ca_cxg
from deepsc.utils import setup_logging


def map_genes_to_hgnc(input_gene_file, hgnc_database_file, output_file):
    # read HGNC database and build a case-insensitive mapping
    symbol_to_info = {}
    with open(hgnc_database_file, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            eid = row["Ensembl gene ID"].strip()
            approved = row["Approved symbol"].strip()
            aliases = (
                row["Alias symbol"].strip().split(",")
                if row["Alias symbol"].strip()
                else []
            )
            previous = (
                row["Previous symbol"].strip().split(",")
                if row["Previous symbol"].strip()
                else []
            )
            all_names = [approved] + aliases + previous
            for name in all_names:
                symbol_to_info[name.upper()] = (eid, approved)

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

    # 为每个唯一的 Approved Name 分配 ID
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
    hgnc_database_path: str,
):
    cxg_mapped_output = os.path.join(
        SCRIPTS_ROOT, "preprocessing", "cxg_matched_genes.csv"
    )
    ca3_mapped_output = os.path.join(
        SCRIPTS_ROOT, "preprocessing", "3ca_matched_genes.csv"
    )
    main_log_file = setup_logging("preprocessing", "./logs")
    map_genes_to_hgnc(cxg_input_path, hgnc_database_path, cxg_mapped_output)
    map_genes_to_hgnc(ca3_input_path, hgnc_database_path, ca3_mapped_output)
    merge_gene_mappings(cxg_mapped_output, ca3_mapped_output, GENE_MAP_PATH)


def gene_name_normalization(
    hgnc_database_path: str = HGNC_DATABASE,
):
    """
    Main function to normalize gene names from CXG and 3CA datasets.
    """
    # cxg_input_path, ca3_input_path = get_feature_name_3ca_cxg()
    cxg_input_path = "/home/angli/DeepSC/scripts/preprocessing/cxg_gene_names.txt"
    ca3_input_path = "/home/angli/DeepSC/scripts/preprocessing/3ca_gene_names.txt"
    process_gene_names(
        cxg_input_path=cxg_input_path,
        ca3_input_path=ca3_input_path,
        hgnc_database_path=hgnc_database_path,
    )


if __name__ == "__main__":
    cxg_feature_names, ca3_feature_names = get_feature_name_3ca_cxg()
    print(f"CXG feature names saved to: {cxg_feature_names}")
    print(f"3CA feature names saved to: {ca3_feature_names}")
    logging.info(f"CXG feature names saved to: {cxg_feature_names}")
    logging.info(f"3CA feature names saved to: {ca3_feature_names}")
    gene_name_normalization()
