MAJOR_TISSUE_LIST = [
    "heart",
    "blood",
    "brain",
    "lung",
    "kidney",
    "intestine",
    "pancreas",
]
CANCER_LIST = [
    "malignant ovarian serous tumor",
    "glioblastoma",
    "lung adenocarcinoma",
    "squamous cell lung carcinoma",
    "small cell lung carcinoma",
    "non-small cell lung carcinoma",
    "B-cell non-Hodgkin lymphoma",
    "follicular lymphoma",
    "gastric cancer",
    "blastoma",
    "pilocytic astrocytoma",
    "acute myeloid leukemia",
    "tubular adenoma",
    "clear cell renal carcinoma",
    "adenocarcinoma",
    "tubulovillous adenoma",
    "colorectal cancer",
    "Wilms tumor",
    "acute promyelocytic leukemia",
    "neuroendocrine carcinoma",
    "chromophobe renal cell carcinoma",
]
VERSION = "2025-01-30"


#  build the value filter dict for each tissue
VALUE_FILTER = {
    tissue: f"suspension_type != 'na' and disease == 'normal' and tissue_general == '{tissue}'"
    for tissue in MAJOR_TISSUE_LIST
}
# build the value filter dict for cells related with other tissues
# since tileDB does not support `not in ` operator, we will just use `!=` to filter out the other tissues
VALUE_FILTER["others"] = "suspension_type != 'na' and disease == 'normal'"
for tissue in MAJOR_TISSUE_LIST:
    VALUE_FILTER["others"] = (
        f"{VALUE_FILTER['others']} and (tissue_general != '{tissue}')"
    )

VALUE_FILTER["pan-cancer"] = "suspension_type != 'na'"
cancer_condition = ""
for disease in CANCER_LIST:
    if cancer_condition == "":
        cancer_condition = f"(disease == '{disease}')"
    else:
        cancer_condition = f"{cancer_condition} or (disease == '{disease}')"
VALUE_FILTER["pan-cancer"] = f"(suspension_type != 'na') and ({cancer_condition})"

if __name__ == "__main__":
    # print(VALUE_FILTER["others"])
    # print(MAJOR_TISSUE_LIST)
    print(VALUE_FILTER["pan-cancer"])
