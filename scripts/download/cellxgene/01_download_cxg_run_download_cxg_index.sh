#!/bin/sh

cd "$(dirname "$0")/../../.."

set -a
source .env
set +a

while read QUERY; do
    echo "building index for ${QUERY}"
    python -m scripts.download.cellxgene.build_index_list --query-name ${QUERY} --output-dir ${DATA_PATH_CELLXGENE}
done < ${QUERY_PATH_CELLXGENE}
