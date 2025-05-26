import csv

# Input files
genes_file = (
    "/home/angli/DeepSC/scripts/download/3ca/not_modified_merged_unique_genes.txt"
)
results_file = "/home/angli/DeepSC/scripts/download/3ca/HGNC_database.txt"

# output files
matched_file = "/home/angli/DeepSC/scripts/download/3ca/cxg_matched_genes.txt"
unmatched_file = "/home/angli/DeepSC/scripts/download/3ca/cxg_unmatched_gene_name.txt"
final_unmatched_file = (
    "/home/angli/DeepSC/scripts/download/3ca/cxg_final_unmatched_gene_name.txt"
)

# read gene list
with open(genes_file, "r") as f:
    gene_list = [line.strip() for line in f if line.strip()]

# dictionary from different symbols to Ensembl IDs
symbol_to_ensembl = {}
alias_to_ensembl = {}
previous_to_ensembl = {}

with open(results_file, "r") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        ensembl_id = row["Ensembl gene ID"].strip()
        approved = row["Approved symbol"].strip()
        alias = row["Alias symbol"].strip()
        previous = row["Previous symbol"].strip()
        if approved:
            symbol_to_ensembl[approved] = ensembl_id
        if alias:
            alias_to_ensembl[alias] = ensembl_id
        if previous:
            previous_to_ensembl[previous] = ensembl_id


matched = []
unmatched = []

for gene in gene_list:
    if gene in symbol_to_ensembl:
        matched.append((gene, symbol_to_ensembl[gene]))
    elif gene in alias_to_ensembl:
        matched.append((gene, alias_to_ensembl[gene]))
    elif gene in previous_to_ensembl:
        matched.append((gene, previous_to_ensembl[gene]))
    else:
        unmatched.append(gene)


with open(unmatched_file, "w") as f:
    for gene in unmatched:
        f.write(gene + "\n")

# Process unmatched genes to check against upper case versions
final_unmatched = []
for gene in unmatched:
    upper_gene = gene.upper()
    if upper_gene in symbol_to_ensembl:
        matched.append((gene, symbol_to_ensembl[upper_gene]))
    elif upper_gene in alias_to_ensembl:
        matched.append((gene, alias_to_ensembl[upper_gene]))
    elif upper_gene in previous_to_ensembl:
        matched.append((gene, previous_to_ensembl[upper_gene]))
    else:
        final_unmatched.append(gene)

# write matched genes to file
matched.sort()
with open(matched_file, "w") as f:
    for gene, eid in matched:
        f.write(f"{gene}\t{eid}\n")

# write unmatched genes to file
final_unmatched.sort()
with open(final_unmatched_file, "w") as f:
    for gene in final_unmatched:
        f.write(gene + "\n")
