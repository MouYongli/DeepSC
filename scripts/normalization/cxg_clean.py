import re


def extract_ensg_name(name):
    """
    Extract the Ensembl gene ID from a given gene name.
    If the name contains an Ensembl ID, return it without version number.
    """
    match = re.search(r"ENSG\d+(?:\.\d+)?", name)
    if match:
        ensg_id = match.group()
        return ensg_id.split(".")[0]
    return name.strip()


input_file = "merged_unique_genes.txt"
output_file = "../result/found_ENSG_merged_unique_genes.txt"

with open(input_file, "r") as infile:
    lines = infile.readlines()

processed = []
not_modified = []
for line in lines:
    original = line.strip()
    if not original:
        continue
    new_name = extract_ensg_name(original)
    if new_name != original:
        processed.append((original, new_name))
    else:
        not_modified.append(original)

with open(output_file, "w") as outfile:
    for original, new_name in sorted(processed):
        outfile.write(f"{original}\t{new_name}\n")

with open("../result/not_modified_merged_unique_genes.txt", "w") as notmod_file:
    for name in sorted(not_modified):
        notmod_file.write(name + "\n")
