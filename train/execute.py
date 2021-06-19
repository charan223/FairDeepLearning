from __future__ import print_function
import sys
import json
import os
from os.path import join as pjoin, exists as pexists
from importlib import import_module
import random
import numpy as np
import torch
import wandb
import time
import glob
from utils.options import parse_args
from utils.log_utils import save_args
from dataloaders.dataset_loader import Dataset
from dataloaders.adult_loader import AdultDataset
from models import ModelFactory, MODEL_REGISTRY
from utils.log_utils import (
    TrainLog,
    AverageMeterSet,
    save_dict2json,
    save_torch_model,
    append_metric_type,
)
from utils.early_stopping import EarlyStopping, EarlyStoppingVAE
from metrics.list import calculate_metrics, get_metrics, CostTemplate
from models.model_ffvae import MLP

POS_INF = +1000000000.0  # +sys.float_info.max
NEG_INF = -1000000000.0  # -sys.float_info.max


def init_logs(args, wandb, model, dataset):
    """Initialize train, valid, and test logger objects, along with returning a dictionary of the metrics that will be calculated.

    Parameters
    ----------
    args : object
        The command line arguments object obtained from argparser
    wandb : object
        The initialized wandb singleton object obtained from the wandb logging library
    model : torch.nn.Module
        A pytorch model
    dataset: torch.utils.Dataset
        The pytorch Dataset

    Returns
    -------
    train_log, valid_log, test_log: TrainLog
        TrainLog objects that have been initialized to log to args.output_dir
    all_metrics: List[Metric]
        List of all performance and fairness metrics that will be calculated
    """

    train_log = TrainLog(args.output_dir, "train", wandb=wandb, init_tb=True)
    valid_log = TrainLog(args.output_dir, "valid", wandb=wandb)
    test_log = TrainLog(args.output_dir, "test", wandb=wandb)
    valid_log.set_summary_writer(train_log.get_summary_writer())
    test_log.set_summary_writer(train_log.get_summary_writer())

    valid_log_list, test_log_list = [], []

    train_iter, valid_iter, test_iter = get_iters(args, dataset)

    train_log.update_config(args)
    train_log.update_config({"clr_ratio": f"{args.beta_1,args.beta_2}"})
    train_log.update_config({"g_y_ratio": f"{args.egr,args.ogr}"})

    print("\nINITIAL EVAL (will be replaced by init values):\n")
    # each have diff is_better fn
    all_metrics = get_metrics(**args.metrics_args)
    mod = import_module("train", package="train")
    test = getattr(mod, "test")

    vlogs = test(args, model, args.device, dataset, valid_iter, valid_log, "VALID")
    tlogs = test(args, model, args.device, dataset, test_iter, test_log, "TEST")

    # override meaningless metrics due to possible bad init
    all_metrics_names = [m.name for m in all_metrics]
    init_vlogs = init_tlogs = {
        k: NEG_INF if k in all_metrics_names else POS_INF for k in vlogs.keys()
    }
    valid_log_list.append(init_vlogs)
    test_log_list.append(init_tlogs)

    return (
        train_log,
        valid_log,
        test_log,
        valid_log_list,
        test_log_list,
        vlogs,
        tlogs,
        all_metrics,
    )


def get_iters(args, dataset):
    """Get the train, valid, and test batch iterators

    Parameters
    ----------
    args : object
        The command line arguments object obtained from argparser
    dataset: torch.utils.Dataset
        The pytorch Dataset

    Returns
    -------
    train/valid/test_iters: dataloaders/dataset_loader.DatasetIterator
        Batch iterators for train, valid, and test data
    """

    def new_train_iter():
        return dataset.get_batch_iterator(
            "train", args.batch_size, shuffle=True, keep_remainder=False, seed=args.seed
        )

    def new_valid_iter():
        return dataset.get_batch_iterator(
            "valid", args.batch_size, shuffle=True, keep_remainder=False, seed=args.seed
        )

    def new_test_iter():
        return dataset.get_batch_iterator(
            "test", args.batch_size, shuffle=True, keep_remainder=False, seed=args.seed
        )

    return new_train_iter(), new_valid_iter(), new_test_iter()


def get_dataset_args(args, dataset):
    args.metrics_args = {
        "privileged_vals": dataset.privileged_vals,
        "sensitive_dict": dataset.sensitive_dict,
    }
    save_dict2json(
        args.metrics_args, os.path.join(args.output_dir, "metrics_args.json")
    )
    return


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    return


def get_args(wandb):
    """Create and return an ArgumentParser, while intializing the wandb logger

    Parameters
    ----------
    wandb : object
        The initialized wandb singleton object obtained from the wandb logging library

    Returns
    -------
    NamedTuple
        A tuple of the parsed cmdline arguments
    """

    args = parse_args()
    print("\nARGS:\n{}".format(args))
    args.command_txt = " ".join(sys.argv)

    print(f"ifwandb = {args.ifwandb}")
    if args.ifwandb:
        # Load wandb settings and initialize global wandb object
        with open(args.wandb_init) as f:
            wandb_args = json.load(f)

        wandb_args["name"] = args.wandb_name
        wandb_args["tags"] = [args.arch, "mila"]
        wandb.init(**wandb_args)
        # End wandb init
    else:
        wandb = None

    # setup
    args.use_cuda = not args.no_cuda and torch.cuda.is_available()

    if not args.name:
        args.name = "%s" % ("".join(time.ctime().split()))

    if "laftr" in args.arch:
        args.model_name = "laftr"
    elif "cfair" in args.arch:
        args.model_name = "cfair"
    else:
        args.model_name = args.arch

    args.output_dir = os.path.join(args.output_dir, args.name)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    args.output_dir = os.path.join(args.output_dir, args.wandb_name)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    args.global_step = 0

    if args.data == "adult":
        args.input_dim = 112
        args.measure_sensattr = args.sensattr
        if "laftr" in args.model_name:
            args.zdim = 8
            args.aud_steps = 1
            args.adepth, args.edepth, args.cdepth = 0, 0, 0
            args.awidths, args.ewidths, args.cwidths = 0, 0, 0
            args.batch_size = 64
        elif "cfair" in args.model_name:
            args.zdim = 60
            args.adepth, args.edepth, args.cdepth = 1, 0, 0
            args.awidths, args.ewidths, args.cwidths = 50, 0, 0
            args.batch_size = 64
        elif "ffvae" in args.model_name:
            args.zdim = 60
            args.adepth, args.edepth, args.cdepth = 1, 1, 1
            args.awidths, args.ewidths, args.cwidths = 200, 200, 200
            args.batch_size = 64

    save_args(args, os.path.join(args.output_dir, "args.json"))
    args.device = torch.device("cuda" if args.use_cuda else "cpu")

    return args


def init_early_stopping(args, all_metrics, vlogs, tlogs, init_vlogs, init_tlogs, model):
    # prep early stopping
    print("\nEARLY STOPPING SETUP:\n")
    chpt_str = "model_e%d.chpt" % 0
    early_stopping, early_stopping_vae = None, None
    epoch = 0

    if args.estopping:
        early_stopping = EarlyStopping(
            args.patience, "loss", CostTemplate().is_better_than
        )
        early_stopping(init_vlogs, init_tlogs, pjoin(args.output_dir, chpt_str))

        save_torch_model(
            args,
            epoch,
            model,
            pjoin(args.output_dir, "start_" + chpt_str),
            early_stop=early_stopping.get_status(),
        )

        early_stopping_vae = EarlyStoppingVAE()

    else:
        save_torch_model(
            args, epoch, model, pjoin(args.output_dir, "start_" + chpt_str)
        )

    return early_stopping, early_stopping_vae


def save_early_stopping_logs(
    args, early_stopping, train_log, valid_log_list, test_log_list
):
    save_dict2json(valid_log_list, pjoin(args.output_dir, "valid_logs.json"))
    save_dict2json(test_log_list, pjoin(args.output_dir, "test_logs.json"))

    if args.estopping:
        early_stopping.save(pjoin(args.output_dir, "early_stop.json"))
        early_stopping_stats = early_stopping.get_status()

        best_es_main_cost_ctr = early_stopping_stats["best_call_counter"]

        early_stopping_updt = {
            f"es/{k}/test": v
            for k, v in early_stopping_stats["corresponding_test"].items()
        }
        early_stopping_updt.update(
            {
                f"es/{k}/val": v
                for k, v in early_stopping_stats["corresponding_valid"].items()
            }
        )

        train_log.add_to_wandb_summary(early_stopping_updt)
    train_log.send_files(f"{args.output_dir}/*.json")
    train_log.send_files(f"{args.output_dir}/*.msgpack")
    train_log.send_files(f"{args.output_dir}/*.server.mila.quebec")
    return


def get_model(args):
    model = ModelFactory.build_model(args.model_name, args).to(args.device)
    return model


def train_vae(args, model, early_stopping, dataset, train_log, test_log):
    mod = import_module("train", package="train")
    test_ffvae = getattr(mod, "test_ffvae")
    train_ffvae = getattr(mod, "train_ffvae")

    for epoch in range(1, args.vae_epochs + 1):
        train_iter, valid_iter, test_iter = get_iters(args, dataset)
        train_logs_vae = train_ffvae(
            args, model, args.device, dataset, train_iter, epoch, train_log, "TRAIN"
        )
        wandb_trainvae_metrics = append_metric_type("TRAIN_VAE", train_logs_vae)

        valid_logs_vae = test_ffvae(
            args, model, args.device, dataset, valid_iter, valid_log, epoch, "VALID"
        )
        wandb_validvae_metrics = append_metric_type("VALID", valid_logs_vae)

        train_log.log(wandb_trainvae_metrics, commit=False)
        train_log.log(wandb_validvae_metrics, commit=False)

        if early_stopping.stop(
            valid_logs_vae["ffvae_cost/avg"], valid_logs_vae["disc_cost/avg"]
        ):
            print("FFVAE training stopped after ", epoch, " epochs")
            break
    return


def train_model(
    args,
    model,
    early_stopping,
    dataset,
    train_log,
    valid_log,
    test_log,
    valid_log_list,
    test_log_list,
):
    mod = import_module("train", package="train")
    test = getattr(mod, "test")
    train = getattr(mod, "train")

    for epoch in range(1, args.epochs + 1):
        train_iter, valid_iter, test_iter = get_iters(args, dataset)
        chpt_str = "model_e%d.chpt" % epoch

        train_logs_class = train(
            args, model, args.device, dataset, train_iter, epoch, train_log, "TRAIN"
        )
        vlogs = test(args, model, args.device, dataset, valid_iter, valid_log, "VALID")
        tlogs = test(args, model, args.device, dataset, test_iter, test_log, "TEST")

        valid_log_list.append(vlogs)
        test_log_list.append(tlogs)

        wandb_train_metrics = append_metric_type("TRAIN_CLASS", train_logs_class)
        wandb_valid_metrics = append_metric_type("VALID", vlogs)
        wandb_test_metrics = append_metric_type("TEST", tlogs)

        train_log.log(wandb_train_metrics, commit=False)
        train_log.log(wandb_valid_metrics, commit=False)
        train_log.log(wandb_test_metrics, commit=False)
        train_log.log({"epoch": epoch}, commit=True)

        if args.estopping:
            early_stopping(vlogs, tlogs, pjoin(args.output_dir, chpt_str))
            status = early_stopping.get_status()
            if status["should_stop"] or epoch == args.epochs:
                print("\nEARLY STOPPING at epoch:", epoch)
                train_log.add_to_wandb_summary({"es/epoch": epoch})
                save_torch_model(
                    args,
                    epoch,
                    model,
                    pjoin(args.output_dir, "final_" + chpt_str),
                    early_stop=early_stopping.get_status(),
                )
                break
            save_torch_model(
                args,
                epoch,
                model,
                pjoin(args.output_dir, chpt_str),
                early_stop=early_stopping.get_status(),
            )
        else:
            save_torch_model(args, epoch, model, pjoin(args.output_dir, chpt_str))

    return early_stopping


if __name__ == "__main__":
    os.environ["WANDB_MODE"] = "dryrun"
    args = get_args(wandb)

    print("\nDATA:\n")
    dataset = Dataset(args)
    args.inp_shape = dataset.inp_shape
    get_dataset_args(args, dataset)
    set_seed(args.seed)

    model = get_model(args)
    print("\nMODEL:\n{}".format(model))

    (
        train_log,
        valid_log,
        test_log,
        valid_log_list,
        test_log_list,
        vlogs,
        tlogs,
        all_metrics,
    ) = init_logs(args, wandb, model, dataset)

    if args.estopping:
        early_stopping, early_stopping_vae = init_early_stopping(
            args, all_metrics, vlogs, tlogs, valid_log_list[0], test_log_list[0], model
        )
    else:
        early_stopping, early_stopping_vae = None, None

    print("\nTRAINING:\n")
    time_start = time.time()
    if "ffvae" in args.model_name:
        train_vae(args, model, early_stopping_vae, dataset, train_log, test_log)

    train_model(
        args,
        model,
        early_stopping,
        dataset,
        train_log,
        valid_log,
        test_log,
        valid_log_list,
        test_log_list,
    )

    time_end = time.time()
    print("Total time used for training = {} seconds.".format(time_end - time_start))

    save_early_stopping_logs(
        args, early_stopping, train_log, valid_log_list, test_log_list
    )

    print("\n*** DONE ***")
