set -a
source .env
set +a

python -m deepsc.data.download.tripleca.merge_and_filter_dataset \
    --log_path "$DEEPSC_LOGS_ROOT" \
    --dataset_root_path "$DATA_PATH_3CA"
