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

dataset = ['adult']
seeds = [3, 4, 5]
clr_ratios = [(0.5, 0.5), (0.01, 0.01), (0.66, 0.33), (0.1, 0.1), (0.06, 0.36)]
sensitiveattrs = ['age']
pos_ratios = [(0.5, 0.5)]

archs = ['mlp']

fair_coeffs = [0.1]
aud_steps = [2]
adv_coeffs = [0.1]
gammas = [0.1]
alphas = [10]

widths = [32]
edepths = [2]
ewidths = [32]
adepths = [2]
awidths = [32]
cdepths = [2]
cwidths = [32]
zdims = [16]

data_dir = ['./data/']
out_dir = ['/scratch/charanr/fairness-project/output/']
replicate = [1]
num_epochs = [150]
patiences = [5]

all_experiments = list(itertools.product(data_dir, out_dir, 
                                         dataset,
                                         clr_ratios, pos_ratios,
                                         sensitiveattrs,
                                         edepths, ewidths, adepths, 
                                         awidths, cdepths, cwidths,
                                         zdims,
                                         seeds, archs,
                                         fair_coeffs, aud_steps,
                                         adv_coeffs, gammas, alphas,
                                         replicate, num_epochs, patiences))


print(f"Total number of jobs = {len(all_experiments)}")
for idx, exp in enumerate(all_experiments):

    MAIN_CMD = f"./scripts/slurm_launcher.sh" \
               f" --ddir {exp[0]}"\
               f" --odir {exp[1]}"\
               f" --dsetname {exp[2]}"\
               f" --beta_1 {exp[3][0]}"\
               f" --beta_2 {exp[3][1]}"\
               f" --egr {exp[4][0]}"\
               f" --ogr {exp[4][1]}"\
               f" --sattr {exp[5]}"\
               f" --edpt {exp[6]}"\
               f" --ewd {exp[7]}"\
               f" --adpt {exp[8]}"\
               f" --awd {exp[9]}"\
               f" --cdpt {exp[10]}"\
               f" --cwd {exp[11]}"\
               f" --zdim {exp[12]}"\
               f" --seed {exp[13]}"\
               f" --arch {exp[14]}"\
               f" --fair_coeff {exp[15]}"\
               f" --aud_steps {exp[16]}"\
               f" --adv_coeff {exp[17]}"\
               f" --gamma {exp[18]}"\
               f" --alpha {exp[19]}"\
               f" --replicate {exp[20]}"\
               f" --num_epochs {exp[21]}"\
               f" --ptnc {exp[22]}"\

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
