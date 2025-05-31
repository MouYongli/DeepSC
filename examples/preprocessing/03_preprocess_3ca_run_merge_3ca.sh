INPUT_DIR="/home/angli/DeepSC/data/3ac/mapped_batch_data"
OUTPUT_DIR="/home/angli/DeepSC/data/3ac/merged_batch_data

python -m scripts.preprocessing.preprocess_3ca_merge \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR" \
