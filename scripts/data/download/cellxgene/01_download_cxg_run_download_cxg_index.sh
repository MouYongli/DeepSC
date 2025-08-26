#!/bin/sh
set -a
source .env
set +a

while read QUERY; do
    echo "building index for ${QUERY}"
    python -m deepsc.data.download.cellxgene.build_index_list --query-name ${QUERY} --output-dir ${INDEX_PATH_CELLXGENE}
done < ${QUERY_PATH_CELLXGENE}
