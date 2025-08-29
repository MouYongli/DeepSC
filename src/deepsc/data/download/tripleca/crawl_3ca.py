import os.path as osp
import uuid

import pandas as pd
import requests
from bs4 import BeautifulSoup

from deepsc.data.download.tripleca.config import BASE_URL, ORGAN_LIST


def data_crawl():
    """Crawl organ study tables and return merged DataFrame."""

    all_dfs = []
    organ_links = [osp.join(BASE_URL, organ) for organ in ORGAN_LIST]

    # Iterate over each link and parse the table
    for link in organ_links:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(link, headers=headers)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
        else:
            print(f"Failed to fetch {link}")
            continue  # Skip invalid links

        table = soup.find("table")
        if not table:
            print(f"No table found in {link}")
            continue  # Skip pages without a table

        data = []
        headers = []

        # Extract table header
        header_row = table.find("tr")
        if header_row:
            headers = [th.text.strip() for th in header_row.find_all("th")]

        # Iterate over table rows (skip header row)
        for row in table.find_all("tr")[1:]:
            cols = row.find_all("td")
            row_data = []
            row_links = {}

            for idx, col in enumerate(cols):
                text = col.text.strip()
                link_tag = col.find("a", href=True)

                # If column contains a link, record it
                if link_tag:
                    link_href = link_tag["href"]
                    col_name = headers[idx] if headers else f"col_{idx}"
                    row_links[f"{col_name}_link"] = link_href

                row_data.append(text)

            # Construct row dictionary
            row_dict = dict(zip(headers, row_data))
            row_dict.update(row_links)

            organ_name = link.replace(BASE_URL, "").replace("-", " ")
            row_dict["Organ"] = organ_name
            row_dict["Study_uuid"] = str(uuid.uuid4())
            data.append(row_dict)

        df = pd.DataFrame(data)
        all_dfs.append(df)

    # Merge all dataframes
    if not all_dfs:
        print("No valid tables found.")
        return None

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

    return merged_df[desired_order]
