from __future__ import print_function
import time
import sys
import os
from os import getenv
from os.path import join as pjoin, exists as pexists
import random
import numpy as np
import wandb
import json
from itertools import tee

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import train_precompute
from utils.log_utils import AverageMeterSet
from metrics.list import calculate_metrics


def train(
    args, model, device, dataset, train_loader, epoch, train_log, mode_name="TRAIN"
):
    model.train()
    start_epoch = time.time()
    average_meters = AverageMeterSet()
    averages = average_meters.averages()

    A_weights, Y_weights, AY_weights = (
        dataset.get_A_proportions(),
        dataset.get_Y_proportions(),
        dataset.get_AY_proportions(),
    )
    reweight_target_tensor, reweight_attr_tensors = train_precompute(
        args, args.device, dataset
    )

    torch.autograd.set_detect_anomaly(True)

    num = args.replicate
    train_loader, new_train_loader = tee(train_loader)
    for _ in range(int(num)):
        new_train_loader, train_loader = tee(new_train_loader)
        for batch_idx, (xs, ys, attrs) in enumerate(train_loader):

            if not args.model_name == "conv" and not args.model_name == "ffvae":
                xs = xs.reshape(len(xs), -1)

            xs, ys, attrs = (
                torch.from_numpy(xs).to(device),
                torch.from_numpy(ys).to(device),
                torch.from_numpy(attrs).to(device),
            )
            ys, attrs = (
                ys.type(torch.LongTensor).to(device),
                attrs.type(torch.LongTensor).to(device),
            )

            pre_softmax, cost_dict = model(
                xs, ys, attrs, "train", A_weights, Y_weights, AY_weights, reweight_target_tensor, reweight_attr_tensors)

            # get the index of the max log-probability
            if args.model_name != "laftr":
                pred = pre_softmax.argmax(dim=1, keepdim=True)
            else:
                pred = pre_softmax > 0.5

            acc = pred.eq(ys.view_as(pred)).sum().item() / xs.size(0)

            stats = dict((n, c.item()) for (n, c) in cost_dict.items())
            stats["acc"] = acc
            average_meters.update_dict(stats)

            args.global_step += 1

            if batch_idx % args.log_interval == 0:
                averages = average_meters.averages()
                print(epoch, batch_idx, averages)
                if train_log:
                    train_log.record(args.global_step, averages)

    print(
        "Epoch ", epoch, " ended. %.1fm/ep" % (
            float(time.time() - start_epoch) / 60.0)
    )
    return averages


def test(args, model, device, dataset, test_loader, test_log, mode_name=""):
    model.eval()
    average_meters = AverageMeterSet()
    test_data, test_pre_softmax = [], []
    test_original_label, test_label, test_sensitiveattr = [], [], []

    A_weights, Y_weights, AY_weights = (
        dataset.get_A_proportions(),
        dataset.get_Y_proportions(),
        dataset.get_AY_proportions(),
    )
    reweight_target_tensor, reweight_attr_tensors = train_precompute(
        args, args.device, dataset
    )

    torch.autograd.set_detect_anomaly(True)
    with torch.no_grad():
        for xs, ys, attrs in test_loader:
            if not args.model_name == "conv" and not args.model_name == "ffvae":
                xs = xs.reshape(len(xs), -1)
            xs, ys, attrs = (
                torch.from_numpy(xs).to(device),
                torch.from_numpy(ys).to(device),
                torch.from_numpy(attrs).to(device),
            )
            ys, attrs = (
                ys.type(torch.LongTensor).to(device),
                attrs.type(torch.LongTensor).to(device),
            )

            pre_softmax, cost_dict = model(
                xs, ys, attrs, "test", A_weights, Y_weights, AY_weights, reweight_target_tensor, reweight_attr_tensors)

            test_pre_softmax.append(pre_softmax.data)
            test_label.append(ys.data)
            test_sensitiveattr.append(attrs.data)

            stats = dict((n, c.item()) for (n, c) in cost_dict.items())
            average_meters.update_dict(stats)

        test_data = None
        test_pre_softmax = torch.cat(test_pre_softmax, dim=0)

        if args.model_name != "laftr":
            test_probability = F.softmax(test_pre_softmax, dim=1)
            test_predicted = test_pre_softmax.argmax(dim=1)
        else:
            test_probability = torch.ones((test_pre_softmax.shape[0], 2))
            test_probability[:, 0], test_probability[:, 1] = (
                1 - test_pre_softmax,
                test_pre_softmax,
            )
            test_predicted = ((test_pre_softmax > 0.5) > 0).type(torch.uint8)

        test_original_label = None
        test_label = torch.cat(test_label, dim=0)
        test_sensitiveattr = torch.cat(
            test_sensitiveattr, dim=0)

    averages = average_meters.averages()
    print("{}Costs:\n{}".format(mode_name + " ", averages))

    # Metrics:
    metrics_input = dict(
        data=test_data,
        pre_softmax=test_pre_softmax.cpu().numpy(),
        probability=test_probability.cpu().numpy(),
        predicted=test_predicted.cpu().numpy(),
        original_label=test_original_label,
        label=test_label.cpu().numpy(),
        sensitiveattr=test_sensitiveattr.cpu().numpy(),
        positive_pred=dataset.pos_label,  # defined by dataset collector
        privileged_vals=dataset.privileged_vals,  # defined by dataset collector
        sensitive_dict=dataset.sensitive_dict,  # defined by dataset collector
    )
    metrics = calculate_metrics(metrics_input)
    print("{}Metrics:\n{}".format(mode_name + " ", metrics))

    logs = {**averages, **metrics}

    CALC_TO_METRICS = {
        "DiffCal+": "diff:0-calibrationPosto1-calibrationPos",
        "DiffCal-": "diff:0-calibrationNegto1-calibrationNeg",
        "DP": "CV",
        "DiffAcc": "diff:0-accuracyto1-accuracy",
        "EqOp(y=1)": "diff:0-TPRto1-TPR",
        "EqOp(y=0)": "diff:0-FPRto1-FPR",
    }

    mtrs = {}
    for k, v in CALC_TO_METRICS.items():
        mtrs[k] = metrics[v]

    logs.update(mtrs)

    print(logs)
    if test_log:
        test_log.record(args.global_step, logs)
    return logs


def train_ffvae(
    args, model, device, dataset, train_loader, epoch, train_log, mode_name="TRAIN"
):
    model.train()

    start_epoch = time.time()
    average_meters = AverageMeterSet()
    averages = average_meters.averages()

    reweight_target_tensor, reweight_attr_tensors = torch.tensor([1.0, 1.0]).to(
        device), [torch.tensor([1.0, 1.0]).to(device), torch.tensor([1.0, 1.0]).to(device)]

    A_weights, Y_weights, AY_weights = dataset.get_A_proportions(
    ), dataset.get_Y_proportions(), dataset.get_AY_proportions()

    torch.autograd.set_detect_anomaly(True)

    for batch_idx, (xs, ys, attrs) in enumerate(train_loader):
        xs, ys, attrs = (
            torch.from_numpy(xs).to(device),
            torch.from_numpy(ys).to(device),
            torch.from_numpy(attrs).to(device),
        )
        ys, attrs = (
            ys.type(torch.LongTensor).to(device),
            attrs.type(torch.LongTensor).to(device),
        )

        _, cost_dict = model(xs, ys, attrs, "ffvae_train",  A_weights, Y_weights,
                             AY_weights, reweight_target_tensor, reweight_attr_tensors)

        stats = dict((n, c.item()) for (n, c) in cost_dict.items())
        average_meters.update_dict(stats)

        args.global_step += 1

        if batch_idx % args.log_interval == 0:
            averages = average_meters.averages()
            print("EPOCH: ", epoch, " BATCH IDX: ",
                  batch_idx, " AVGs: ", averages)
            if train_log:
                train_log.record(args.global_step, averages)

    print(
        "FFvae Training",
        "Epoch ",
        epoch,
        " ended. %.1fm/ep" % (float(time.time() - start_epoch) / 60.0),
    )
    return averages


def test_ffvae(
    args, model, device, dataset, test_loader, test_log, epoch, mode_name=""
):
    model.eval()
    average_meters = AverageMeterSet()
    ones = torch.ones(args.batch_size, dtype=torch.long, device=device)
    zeros = torch.zeros(args.batch_size, dtype=torch.long, device=device)

    A_weights, Y_weights, AY_weights = dataset.get_A_proportions(
    ), dataset.get_Y_proportions(), dataset.get_AY_proportions()
    reweight_target_tensor, reweight_attr_tensors = torch.tensor([1.0, 1.0]).to(
        device), [torch.tensor([1.0, 1.0]).to(device), torch.tensor([1.0, 1.0]).to(device)]

    with torch.no_grad():
        for batch_idx, (xs, ys, attrs) in enumerate(test_loader):
            xs, ys, attrs = (
                torch.from_numpy(xs).to(device),
                torch.from_numpy(ys).to(device),
                torch.from_numpy(attrs).to(device),
            )
            ys, attrs = (
                ys.type(torch.LongTensor).to(device),
                attrs.type(torch.LongTensor).to(device),
            )

            _, cost_dict = model(xs, ys, attrs, "test",  A_weights, Y_weights,
                                 AY_weights, reweight_target_tensor, reweight_attr_tensors)

            stats = dict((n, c.item()) for (n, c) in cost_dict.items())
            average_meters.update_dict(stats)
            args.global_step += 1

            if batch_idx % args.log_interval == 0:
                averages = average_meters.averages()
                print(
                    "VALID EPOCH: ",
                    epoch,
                    " BATCH IDX: ",
                    batch_idx,
                    " AVGs: ",
                    averages,
                )

    return averages
