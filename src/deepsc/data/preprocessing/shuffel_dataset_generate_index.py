import csv
import os
import random

import scipy.sparse

import argparse


def main():
    parser = argparse.ArgumentParser(description="Generate shuffled dataset index")
    parser.add_argument(
        "--original_dir", required=True, help="Directory containing original NPZ files"
    )
    parser.add_argument(
        "--shuffel_plan_path",
        required=True,
        help="Output path for the shuffled plan CSV file",
    )
    args = parser.parse_args()

    original_dir = args.original_dir
    shuffel_plan_path = args.shuffel_plan_path
    target_chunk_size = 200_000

    # 第一步：收集每个文件中有多少个样本（行数）
    file_sample_tuples = []
    for dirpath, _, filenames in os.walk(original_dir):
        for fname in filenames:
            if fname.endswith(".npz"):
                path = os.path.join(dirpath, fname)
                print("check path", path)
                try:
                    matrix = scipy.sparse.load_npz(path)
                    file_sample_tuples.append((path, matrix.shape[0]))
                except Exception as e:
                    print(f"Failed loading {path}: {e}")

    # 第二步：生成全局索引 [(file_path, row_idx)]
    all_indices = []
    for path, nrows in file_sample_tuples:
        for i in range(nrows):
            all_indices.append((path, i))

    # 第三步：打乱 + 构建 alloc 结构
    random.seed(42)
    random.shuffle(all_indices)

    alloc = {}  # {file_path: {chunk_id: [row_ids]}}
    for i, (path, row) in enumerate(all_indices):
        chunk_id = i // target_chunk_size
        if path not in alloc:
            alloc[path] = {}
        if chunk_id not in alloc[path]:
            alloc[path][chunk_id] = []
        alloc[path][chunk_id].append(row)

    # 保存为CSV文件，每行包含：source_file, chunk_id, rows（空格分隔）
    with open(shuffel_plan_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source_file", "chunk_id", "rows"])  # header
        for path, chunk_map in alloc.items():
            for chunk_id, row_list in chunk_map.items():
                row_str = " ".join(map(str, row_list))
                writer.writerow([path, chunk_id, row_str])


if __name__ == "__main__":
    main()
