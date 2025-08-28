#!/bin/bash
set -a
source .env
set +a

while read QUERY_NAME; do
    INDEX_DIR="$INDEX_PATH_CELLXGENE"
    OUTPUT_DIR="$DATA_PATH_CELLXGENE"

    echo "Downloading: ${QUERY_NAME}"
    MAX_PARTITION_SIZE=200000
    TOTAL_NUM=$(wc -l "${INDEX_DIR}/${QUERY_NAME}.idx" | awk '{ print $1 }')
    TOTAL_PARTITION=$((TOTAL_NUM / MAX_PARTITION_SIZE))

    for i in $(seq 0 $TOTAL_PARTITION); do
        echo "Downloading ${QUERY_NAME} partition ${i}/${TOTAL_PARTITION}"
        python -m deepsc.data.download.cellxgene.download_partition \
            --query-name "${QUERY_NAME}" \
            --index-dir "${INDEX_DIR}" \
            --output-dir "${OUTPUT_DIR}" \
            --partition-idx "${i}" \
            --max-partition-size "${MAX_PARTITION_SIZE}"
    done
done < ${QUERY_PATH_CELLXGENE}
