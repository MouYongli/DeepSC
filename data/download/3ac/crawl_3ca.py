import os
import os.path as osp

here = os.path.dirname(os.path.abspath(__file__))

base_url = "https://www.weizmann.ac.il/sites/3CA/"


if __name__ == "__main__":
    with open(osp.join(here, 'organ_list.txt'), 'r') as file:
        organ_list = file.read().splitlines()
        print(organ_list)