from scripts.preprocessing.gene_name_preprocess import process_gene_names

cxg_input_path = "/home/angli/DeepSC/scripts/normalization_0527/cxg_gene_names.txt"
ca3_input_path = "/home/angli/DeepSC/scripts/normalization_0527/3ca_gene_names.txt"
hgnc_database_path = (
    "/home/angli/DeepSC/scripts/normalization_0527/result_0527/HGNC_database.txt"
)


process_gene_names(cxg_input_path, ca3_input_path, hgnc_database_path)
