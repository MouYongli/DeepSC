import pandas as pd
import os
import os.path as osp

here = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(osp.join(here, 'merged_table_data.csv'))
output_path = osp.join(here, 'raw')
for uuid in df["Study_uuid"]:
    folder_path = os.path.join(output_path, uuid)
    os.makedirs(folder_path, exist_ok=True) 