import numpy as np
import torch
import torch.nn.functional as F
from itertools import repeat
import collections
import math
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def map_dict_device(arg, device):
    return {k: v.to(device) for k, v in arg.items()}

def add_argument(parser, name, *args, **kwargs):
    name = name.strip('-')
    l = ['--' + name]
    if '-' in name:
        l.append('--' + name.replace('-', '_'))
    if '_' in name:
        l.append('--' + name.replace('_', '-'))
    parser.add_argument(*l, *args, **kwargs)

def train_precompute(args, device, dataset):
    """computing sample numbers, base ratios of datasets for Cfair model"""
    if args.data == "clr-mnist":
        train_target_attrs, train_target_labels = (
            dataset.train_loader.att,
            dataset.train_loader.eo_lbl,
        )
    else:
        train_target_attrs, train_target_labels = (
            dataset.train_loader.A,
            dataset.train_loader.Y,
        )
    train_idx = train_target_attrs == 0
    train_base_0, train_base_1 = np.mean(train_target_labels[train_idx]), np.mean(
        train_target_labels[~train_idx]
    )
    train_y_1 = np.mean(train_target_labels)

    reweight_target_tensor = None
    # For reweighing purpose.
    if args.arch == "cfair":
        reweight_target_tensor = torch.tensor(
            [1.0 / (1.0 - train_y_1), 1.0 / train_y_1]
        ).to(device)
    elif args.arch == "cfair-eo":
        reweight_target_tensor = torch.tensor([1.0, 1.0]).to(device)

    reweight_attr_0_tensor = torch.tensor(
        [1.0 / (1.0 - train_base_0), 1.0 / (train_base_0)]
    ).to(device)
    reweight_attr_1_tensor = torch.tensor(
        [1.0 / (1.0 - train_base_1), 1.0 / (train_base_1)]
    ).to(device)
    reweight_attr_tensors = [reweight_attr_0_tensor, reweight_attr_1_tensor]
    
    print("Average value of A = {}".format(np.mean(train_target_attrs)))
    print("A: sensitive attribute = 0/1")
    return reweight_target_tensor, reweight_attr_tensors
