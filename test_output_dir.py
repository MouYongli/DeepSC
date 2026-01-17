#!/usr/bin/env python3
import os

import sys

# Test if the modification works
sys.path.insert(0, "/home/angli/DeepSC/src")

print(f"Current working directory: {os.getcwd()}")

# Read the cell_type_annotation.py file and check if it has os.getcwd()
with open("/home/angli/DeepSC/src/deepsc/finetune/cell_type_annotation.py", "r") as f:
    content = f.read()
    if "os.getcwd()" in content:
        print("✓ Code has been modified to use os.getcwd()")
    else:
        print("✗ Code still uses old timestamp format")

    if 'timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")' in content:
        print("✗ Old timestamp code still exists!")
    else:
        print("✓ Old timestamp code has been removed")

# Check the actual line
import re

match = re.search(r"self\.output_dir = (.+)", content)
if match:
    print(f"Found: self.output_dir = {match.group(1)}")
