import gc
import json
import math
import os
import random
from functools import reduce

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import (
    ShuffleSplit,
    StratifiedKFold,
    StratifiedShuffleSplit,
    cross_validate,
    train_test_split,
)
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CyclicLR, StepLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import argparse

# from utils import *
from datetime import datetime
from time import time

# from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
# from torch.utils.tensorboard import SummaryWriter


SEED = 2021

# Control sources of randomness
torch.manual_seed(SEED)
random.seed(SEED)

############################# Test lr on cross_organ dataset ##############################
data = sc.read_h5ad("./Data/human_15organ_subset_normed.h5ad")
methods = np.unique(data.obs["orig.ident"])
index_methods = data.obs["orig.ident"]

label = data.obs.celltype
data = data.X

for val_i in range(len(methods)):
    print(methods[val_i])
    train_index = index_methods != methods[val_i]
    val_index = index_methods == methods[val_i]
    X_train, y_train = data[train_index], label[train_index]
    X_test, y_test = data[val_index], label[val_index]
    cv_results = {}
    for c in [1e-3, 1e-2, 1e-1, 1]:
        # print("c={}".format(c))
        lr = LogisticRegression(random_state=0, penalty="l1", C=c, solver="liblinear")
        res = cross_validate(lr, X_train, y_train, scoring=["accuracy"])
        cv_results[c] = np.mean(res["test_accuracy"])
    # print(cv_results)
    # choose best c and calc performance on val_dataset
    best_ind = np.argmax(list(cv_results.values()))
    c = list(cv_results.keys())[best_ind]
    # print("best c={}".format(c))
    lr = LogisticRegression(random_state=0, penalty="l1", C=c, solver="liblinear")
    lr.fit(X_train, y_train)
    # print("train set accuracy: " + str(np.around(lr.score(X_train, y_train), 4)))
    print("test set accuracy: " + str(np.around(lr.score(X_test, y_test), 4)))
    val_macro_f1 = f1_score(y_test, lr.predict(X_test), average="macro")
    print("test set macro F1: " + str(np.around(val_macro_f1, 4)))
