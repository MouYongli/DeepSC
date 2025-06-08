from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io


def plot_3d_matrix(matrix, file_name, output_path):
    """
    绘制矩阵的三维图。
    - matrix: 矩阵 (2D numpy array)
    - file_name: 绘图的文件名，用于展示标题
    """
    # 准备数据
    rows, cols = matrix.shape
    X, Y = np.meshgrid(range(cols), range(rows))  # 生成 X 和 Y 网格
    Z = matrix  # 矩阵值作为 Z 轴

    # 创建新的 3D 图表
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 绘制 3D 曲面图
    surf = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="k", alpha=0.8)

    # 添加颜色条用于显示值范围
    fig.colorbar(surf, shrink=0.5, aspect=10)

    # 设置标题和轴标签
    ax.set_title(f"3D Plot of Matrix from {file_name}", fontsize=14)
    ax.set_xlabel("Gene (X-axis)", fontsize=12)
    ax.set_ylabel("Cell (Y-axis)", fontsize=12)
    ax.set_zlabel("Count (Z-axis)", fontsize=12)
    ax.set_zlim(0, 200)  # 设置 Z 轴范围

    plt.savefig(output_path)
    plt.close()


# To-DO: the dataset cannot be named with uuid, they should be identified with dataset id
# To-DO: the dataset cannot be named with uuid, they should be identified with dataset id
# To-DO: the dataset cannot be named with uuid, they should be identified with dataset id


if __name__ == "__main__":
    csv_path = "target_dataset_files.csv"
    df = pd.read_csv(csv_path)

    results_df = pd.DataFrame(
        columns=[
            "Study_uuid",
            "Total_elements",
            "Proportion_zeros",
            "Proportion_less_than_5",
            "Proportion_less_than_10",
            "Proportion_less_than_20",
        ]
    )

    if all(col in df.columns for col in ["path", "filename"]):
        for row in df.itertuples(index=False):
            file_path = Path(row.path)
            file_name = row.filename
            uuid = row.Study_uuid
            path_of_mtx_file = file_path / file_name

            if not file_path.exists():
                print(f"File {file_path} does not exist")
                continue

            matrix = scipy.io.mmread(path_of_mtx_file).transpose().tocsc()
            dense_matrix = matrix.toarray()
            plot_3d_matrix(dense_matrix, file_name, Path("../report") / f"{uuid}.png")
            df = pd.DataFrame(matrix.toarray())

            total_elements = df.size
            count_zero = (df == 0).sum().sum()
            count_less_than_5 = (df < 5).sum().sum()
            count_less_than_10 = (df < 10).sum().sum()
            count_less_than_20 = (df < 20).sum().sum()

            # 转换为百分数
            ratio_zero = (
                (count_zero / total_elements * 100) if total_elements > 0 else 0
            )
            ratio_less_than_5 = (
                (count_less_than_5 / total_elements * 100) if total_elements > 0 else 0
            )
            ratio_less_than_10 = (
                (count_less_than_10 / total_elements * 100) if total_elements > 0 else 0
            )
            ratio_less_than_20 = (
                (count_less_than_20 / total_elements * 100) if total_elements > 0 else 0
            )

            # 保留两位小数
            ratio_zero = round(ratio_zero, 5)
            ratio_less_than_5 = round(ratio_less_than_5, 5)
            ratio_less_than_10 = round(ratio_less_than_10, 5)
            ratio_less_than_20 = round(ratio_less_than_20, 5)

            # 存入 DataFrame
            results_df.loc[len(results_df)] = [
                uuid,
                total_elements,
                ratio_zero,
                ratio_less_than_5,
                ratio_less_than_10,
                ratio_less_than_20,
            ]

            print(f"完成 processing {file_name} (UUID: {uuid})")

    results_df.to_csv("noiseAnalysis.csv", index=False)
    print(" saved to noiseAnalysis.csv")
