#!/bin/sh
# output directory for the index
OUTPUT_DIR="/home/angli/DeepSC/Data"
QUERY_LIST="/home/angli/DeepSC/scripts/download/cellxgene/query_list.txt"

while read QUERY; do
    echo "building index for ${QUERY}"
    python -m scripts.download.cellxgene.build_index_list --query-name ${QUERY} --output-dir ${OUTPUT_DIR}
done < ${QUERY_LIST}
