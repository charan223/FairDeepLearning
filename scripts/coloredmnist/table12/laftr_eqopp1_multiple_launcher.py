import pandas as pd
import numpy as np
import itertools
import subprocess
import re
import git
import os
import hashlib


seeds = [3]

# clr_ratios = [(0.5, 0.5)]
clr_ratios = [(0.1, 0.1), (0.01, 0.01), (0.001, 0.001), (0.1, 0.9), (0.01, 0.99), (0.5, 0.5)]

green_yellows = [True]

g_y_ratio = [(0.5, 0.5)]
# g_y_ratio = [(0.5, 0.5), (0.75, 0.25), (0.9, 0.1)]

sensitiveattrs = ['bck']
measure_sensitiveattrs = ['bck']
# sensitiveattrs = ['bck', 'color_gy']
# sensitiveattrs = ['green_1', 'bck', 'color_gy']

# fair_coeffs = [1.0]
fair_coeffs = [0.1, 0.5, 1.0, 2.0, 4.0]

adv_coeffs = [1.0]
# adv_coeffs = [0.1, 1.0, 10.0, 100.0, 1000.0]

gammas = [0.1]
alphas = [10]
# gammas = [10, 50, 100]
# alphas = [10, 100, 1000]

archs = ['laftr-eqopp1']



d_coeffs = [0.1]
which_classes = [0]
# which_classes = [0, 1]

widths = [32]
isshades = [True]
aud_steps =[2]
batchnorm = [True]
edepths = [2]
ewidths = [32]
adepths = [2]
awidths = [32]
zdims = [16]

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

'''
df = pd.read_csv(r'all_exp.csv')

done_experiments = []
for i, row in enumerate(df.iterrows()):
    clr_ratios = list(map(float, row[1]['clr_ratio'].replace("(", "").replace(")", "").split(',')))
    clr_ratio = (clr_ratios[0], clr_ratios[1])
    #e_o_ratios = list(map(float, row[1]['e_o_ratio'].replace("(", "").replace(")", "").split(',')))
    #e_o_ratio = (e_o_ratios[0], e_o_ratios[1])
    g_y_ratios = list(map(float, row[1]['g_y_ratio'].replace("(", "").replace(")", "").split(',')))
    g_y_ratio = (g_y_ratios[0], g_y_ratios[1])
    done_experiments.append((row[1]['digit_pattern'], clr_ratio, row[1]['useshade'],
      row[1]['green_1'], row[1]['green_2'], row[1]['gw'], e_o_ratio[0],  row[1]['batchnorm'],
      row[1]['cwidths'], row[1]['sensattr'], row[1]['edepth'], row[1]['ewidths'], row[1]['adepth'],
      row[1]['awidths'], row[1]['zdim'], row[1]['seed'], row[1]['arch'], row[1]['fair_coeff'], row[1]['aud_steps'],
      row[1]['adv_coeff'], row[1]['gamma'], row[1]['alpha'], row[1]['d_coeff'], row[1]['which_class'],
      row[1]['green_yellow'], g_y_ratio, row[1]['measure_sensattr']))

tbd_experiments = [x for x in all_experiments if x not in done_experiments]
print(len(all_experiments), len(tbd_experiments))


print(f"Total number of jobs = {len(tbd_experiments)}")
for idx, exp in enumerate(tbd_experiments):
'''


print(f"Total number of jobs = {len(all_experiments)}")
for idx, exp in enumerate(all_experiments):
    # exp_id = f"mila-laftr-arch_{exp[17]}_seed_{exp[16]}_b1_{exp[1][0]}_b2_{exp[1][1]}_g1_{exp[3]}_g2_{exp[4]}_width_{exp[]}"

    MAIN_CMD = f"./scripts/slurm_launcher.sh --wd {exp[8]}"\
               f" --bck blue-red"\
               f" --label even-odd"\
               f" --ddir ./data/"\
               f" --dsetname clr-mnist"\
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