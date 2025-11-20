#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate label_dict and label files for a dataset
This is useful when you need to run prediction with a specific checkpoint
"""

import numpy as np
import scanpy as sc
import pickle as pkl
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True, help='Path to h5ad data file')
parser.add_argument("--output_prefix", type=str, default="", help='Prefix for output files (e.g., "finetune_zheng68k_")')
args = parser.parse_args()

print("="*60)
print("Generating label_dict and label files")
print("="*60)

# Read data
print(f"\nReading data from: {args.data_path}")
data = sc.read_h5ad(args.data_path)
print(f"Data shape: {data.shape}")

# Generate label_dict and label
label_dict, label = np.unique(np.array(data.obs['celltype']), return_inverse=True)

print(f"\nFound {len(label_dict)} unique cell types:")
for i, ct in enumerate(label_dict):
    count = np.sum(label == i)
    print(f"  {i}: {ct} ({count} cells)")

# Save files
if args.output_prefix:
    label_dict_file = f'{args.output_prefix}label_dict'
    label_file = f'{args.output_prefix}label'
else:
    label_dict_file = 'label_dict'
    label_file = 'label'

with open(label_dict_file, 'wb') as fp:
    pkl.dump(label_dict, fp)
print(f"\nSaved label_dict to: {label_dict_file}")

with open(label_file, 'wb') as fp:
    pkl.dump(label, fp)
print(f"Saved label to: {label_file}")

print(f"\nlabel_dict shape: {label_dict.shape}")
print(f"label shape: {label.shape}")
print("\n" + "="*60)
print("Done!")
print("="*60)
