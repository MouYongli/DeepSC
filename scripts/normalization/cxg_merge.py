# organ list of cellxgene
query_list_file = "query_list.txt"

with open(query_list_file, "r", encoding="utf-8") as f:
    organs = [line.strip() for line in f if line.strip()]

unique_genes = set()

# iterate through each organ and read the corresponding gene file
for organ in organs:
    txt_filename = f"{organ}_genes_expr_gt_50.txt"
    try:
        with open(txt_filename, "r", encoding="utf-8") as gene_file:
            for line in gene_file:
                gene = line.strip()
                if gene:
                    unique_genes.add(gene)
    except FileNotFoundError:
        print(f"cannot found {txt_filename} for organ {organ}, skipping...")


output_file = "merged_unique_genes.txt"
with open(output_file, "w", encoding="utf-8") as out_file:
    for gene in sorted(unique_genes):
        out_file.write(gene + "\n")
