import os
import os.path as osp
import uuid

import pandas as pd
import requests
from bs4 import BeautifulSoup

here = os.path.dirname(os.path.abspath(__file__))

base_url = "https://www.weizmann.ac.il/sites/3CA/"

all_dfs = []  # 用于存储所有的 DataFrame

if __name__ == "__main__":
    with open(osp.join(here, "organ_list.txt"), "r") as file:
        organ_list = file.read().splitlines()

    # 合并 base_url 和 organ_list 生成完整链接
    organ_links = [osp.join(base_url, organ) for organ in organ_list]

    # 遍历每个链接并解析表格
    for link in organ_links:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(link, headers=headers)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
        else:
            print(f"Failed to fetch {link}")
            continue  # 跳过错误的链接

        table = soup.find("table")
        if not table:
            print(f"No table found in {link}")
            continue  # 跳过没有表格的页面

        data = []
        headers = []

        # 获取表头
        header_row = table.find("tr")
        if header_row:
            headers = [th.text.strip() for th in header_row.find_all("th")]

        # 遍历表格数据
        for row in table.find_all("tr")[1:]:  # 跳过表头
            cols = row.find_all("td")
            row_data = []
            row_links = {}

            for idx, col in enumerate(cols):
                text = col.text.strip()
                link_tag = col.find("a", href=True)

                # 如果该列有链接，则记录
                if link_tag:
                    link_href = link_tag["href"]
                    col_name = headers[idx] if headers else f"col_{idx}"
                    row_links[f"{col_name}_link"] = link_href  # 记录超链接列

                row_data.append(text)

            # 添加数据行
            row_dict = dict(zip(headers, row_data))
            row_dict.update(row_links)  # 把超链接列添加进去

            organ_name = link.replace(base_url, "")
            organ_name = organ_name.replace("-", " ")
            row_dict["Organ"] = organ_name
            uuid_str = str(uuid.uuid4())
            row_dict["Study_uuid"] = uuid_str
            data.append(row_dict)

        # 转换为 DataFrame
        df = pd.DataFrame(data)

        # 存储 df
        all_dfs.append(df)

    # **合并所有表格**
    merged_df = all_dfs[0]
    if all_dfs:
        merged_df = pd.concat(all_dfs, ignore_index=True)

        desired_order = [
            "Study_uuid",
            "Organ",
            "Title",
            "Title_link",
            "Data",
            "Data_link",
            "Meta data",
            "Meta data_link",
            "Cell types",
            "Cell types_link",
            "Summary",
            "Summary_link",
            "Disease",
            "Technology",
            "#samples",
            "#cells",
            "Meta programs",
            "Meta programs_link",
            "CNAs",
            "CNAs_link",
            "UMAP",
            "UMAP_link",
            "Cell cycle",
            "Cell cycle_link",
        ]
        dfnew = merged_df[desired_order]
        dfnew.to_csv("data_info.csv", index=False)
        print("Merged table saved to merged_table_data.csv")
    else:
        print("No valid tables found.")

# %%
