MAJOR_TISSUE_LIST = [
    "heart",
    "blood",
    "brain",
    "lung",
    "kidney",
    "intestine",
    "pancreas",
]
VERSION = "2025-01-30"

import os
from pathlib import Path

# Use environment variable with sensible default
PROJECT_ROOT = os.getenv("DEEPSC_PROJECT_ROOT", str(Path(__file__).parent.parent.parent.parent.parent))
CANCER_LIST_PATH = os.getenv(
    "DEEPSC_CANCER_LIST_PATH", 
    os.path.join(PROJECT_ROOT, "scripts", "download", "cellxgene", "cancer_list.txt")
)

# Handle case where file might not exist
try:
    with open(CANCER_LIST_PATH) as f:
        CANCER_LIST = [line.rstrip("\n") for line in f]
except FileNotFoundError:
    print(f"Warning: Cancer list file not found at {CANCER_LIST_PATH}. Using empty list.")
    CANCER_LIST = []

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
