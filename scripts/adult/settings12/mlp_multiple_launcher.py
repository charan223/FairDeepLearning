"""
For reproducing Setting 1, 2 experiments with Mlp model on Adult dataset
"""

import pandas as pd
import numpy as np
import itertools
import subprocess
import re
import git
import os
import hashlib


seeds = [3, 4, 5]
clr_ratios = [(0.5, 0.5), (0.01, 0.01), (0.66, 0.33), (0.33, 0.33), (0.1, 0.1), (0.06, 0.36)]
archs = ['mlp']
sensitiveattrs = ['age']


fair_coeffs = [1.0]
adv_coeffs = [1.0]
gammas = [0.1]
alphas = [10]


widths = [32]
isshades = [True]
aud_steps =[2]
batchnorm = [True]
edepths = [2]
ewidths = [32]
adepths = [2]
awidths = [32]
zdims = [16]

# not used
measure_sensitiveattrs = ['age']
d_coeffs = [1]
which_classes = [0]
green_yellows = [True]
g_y_ratio = [(0.5, 0.5)]
green_1s = [0]
green_widths = [0]
green_2s = [False]
e_o_ratio = [(0.0, 0.0)]
digit_patterns = [0]

all_experiments = list(itertools.product(digit_patterns, clr_ratios, isshades,
                                         green_1s, green_2s, green_widths, e_o_ratio, batchnorm,
                                         widths, sensitiveattrs, edepths, ewidths, adepths, awidths,
                                         zdims, seeds, archs,
                                         fair_coeffs, aud_steps, 
                                         adv_coeffs, 
                                         gammas, alphas, 
                                         d_coeffs, which_classes,
                                         green_yellows, g_y_ratio, measure_sensitiveattrs))

print(f"Total number of jobs = {len(all_experiments)}")
for idx, exp in enumerate(all_experiments):

    MAIN_CMD = f"./scripts/slurm_launcher.sh --wd {exp[8]}"\
               f" --bck blue-red"\
               f" --label even-odd"\
               f" --ddir ./data/"\
               f" --dsetname adult"\
               f" --odir /scratch/charanr/fairness-project/output/"\
               f" --dpt 2"\
               f" --dp {exp[0]}"\
               f" --beta_1 {exp[1][0]}"\
               f" --beta_2 {exp[1][1]}"\
               f" --green_1 {exp[3]}"\
               f" --gw {exp[5]}"\
               f" --epr {exp[6][0]}"\
               f" --opr {exp[6][1]}"\
               f" --num_epochs 150"\
               f" --sattr {exp[9]}"\
               f" --ptnc 5"\
               f" --edpt {exp[10]}"\
               f" --ewd {exp[11]}"\
               f" --adpt {exp[12]}"\
               f" --awd {exp[13]}"\
               f" --zdim {exp[14]}"\
               f" --seed {exp[15]}"\
               f" --arch {exp[16]}"\
               f" --fair_coeff {exp[17]}"\
               f" --aud_steps {exp[18]}"\
               f" --adv_coeff {exp[19]}"\
               f" --gamma {exp[20]}"\
               f" --alpha {exp[21]}"\
               f" --d_coeff {exp[22]}"\
               f" --which_class {exp[23]}"\
               f" --egr {exp[25][0]}"\
               f" --ogr {exp[25][1]}"\
               f" --measure_sattr {exp[26]}"\
                   
    hashed_id = int(hashlib.sha1(MAIN_CMD.encode(
        'utf-8')).hexdigest(), 16) % (10 ** 8)
    MAIN_CMD = f"{MAIN_CMD} --wdbn deepfairness-{hashed_id}"

    print("###################################")
    print(f"EXPERIMENT HASHED ID = {hashed_id}")
    print(f"COMMAND RUNNING = \n {MAIN_CMD} \n")
    print("###################################")
    CMD = MAIN_CMD.split(" ")

    process = subprocess.Popen(CMD, stdout=subprocess.PIPE)

    out, err = process.communicate()

    print(out)
