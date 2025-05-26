import csv

# two dictionary files
file1 = "/home/angli/DeepSC/scripts/download/3ca/cxg_matched_genes.txt"
file2 = "/home/angli/DeepSC/scripts/download/3ca/matched_genes.txt"

# output files
merged_output = "/home/angli/DeepSC/scripts/download/3ca/merged_cxg_3ca_name_id.txt"
different_name_output = (
    "/home/angli/DeepSC/scripts/download/3ca/different_name_cxg_3ca.txt"
)


# read file function
def read_file(path):
    pairs = []
    with open(path, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) >= 2:
                pairs.append((row[0].strip(), row[1].strip()))
    return pairs


# load the two files and merge
pairs1 = read_file(file1)
pairs2 = read_file(file2)
all_pairs = pairs1 + pairs2

unique_pairs = set()
duplicate_pairs = set()
id_to_names = {}

for name, gene_id in all_pairs:
    key = (name, gene_id)
    if key in unique_pairs:
        duplicate_pairs.add(key)
    else:
        unique_pairs.add(key)
        if gene_id not in id_to_names:
            id_to_names[gene_id] = set()
        id_to_names[gene_id].add(name)

# write in merged_output file (unique pairs)
with open(merged_output, "w") as f_out:
    writer = csv.writer(f_out, delimiter="\t")
    for name, gene_id in sorted(unique_pairs):
        writer.writerow([name, gene_id])

# write in different_name_output file (same gene id with multiple names)
with open(different_name_output, "w") as f_diff:
    writer = csv.writer(f_diff, delimiter="\t")
    for gene_id, names in id_to_names.items():
        if len(names) > 1:
            for name in sorted(names):
                writer.writerow([name, gene_id])
