import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os 

if not os.path.exists("./csvs"):
    os.makedirs("./csvs")
if not os.path.exists("./mnist_plots"):
    os.makedirs("./mnist_plots")

# get specific rows from the wandb csv file
def parse_csv(tab, table_type, clr_ratio, g_y_ratio, sensattr, metric_names, l):
  return tab[(tab['table_type'] == table_type) & 
        (tab['clr_ratio'] == clr_ratio) &
        (tab['g_y_ratio'] == g_y_ratio) &
        (tab['sensattr'] == sensattr)][metric_names[l]].values

# load processed files (mean taken over seeds and maximum metrics over hyperparameters)
mlp_csv = './mnist/mnist-mlp-mean_over_seed_max_over_hp.csv'
laftar_csv = pd.read_csv('./mnist/mnist-laftr-mean_over_seed_max_over_hp.csv')
cfair_mod_csv = pd.read_csv('./mnist/mnist-cfair-mean_over_seed_max_over_hp.csv')
ffvae_csv = './mnist/mnist-ffvae-mean_over_seed_max_over_hp.csv'
conv_csv = './mnist/mnist-conv-mean_over_seed_max_over_hp.csv'

laftar_dp_csv =     laftar_csv[laftar_csv["arch"] == "laftr-dp"]
laftar_eqodd_csv =  laftar_csv[laftar_csv["arch"] == "laftr-eqodd"]
laftar_eqopp0_csv = laftar_csv[laftar_csv["arch"] == "laftr-eqopp0"]
laftar_eqopp1_csv = laftar_csv[laftar_csv["arch"] == "laftr-eqopp1"]
cfair_csv =    cfair_mod_csv[cfair_mod_csv["arch"] == "cfair"]

# labels, metrics for plotting
labels = ['acc', 'p-acc', 'up-acc', 'DP', 'EqOpp0', 'EqOpp1', 'EqOdd']
mean_metric_names = ['acc_mean_max', '0-acc_mean_max', '1-acc_mean_max', 'DP_mean_max', 'EqOp0_mean_max', 'EqOp1_mean_max', 'EqOdd_mean_max']
std_metric_names = ['acc_std_corresp', '0-acc_std_corresp', '1-acc_std_corresp', 'DP_std_corresp', 'EqOp0_std_corresp', 'EqOp1_std_corresp', 'EqOdd_std_corresp']
clr = ['deepskyblue', 'khaki', 'lightcoral', 'mediumseagreen', 'lightslategrey', 'plum', 'turquoise', 'cornflowerblue']
clr_ratios = ["(0.5, 0.5)", "(0.1, 0.1)", "(0.01, 0.01)", "(0.001, 0.001)", "(0.1, 0.9)", "(0.01, 0.99)"]
g_y_ratios = ["(0.5, 0.5)", "(0.75, 0.25)", "(0.9, 0.1)"]
table_type = 4
sensattr = "bck"

# load csvs into an array
csvs = []
csvs.append(pd.read_csv(mlp_csv))
csvs.append(pd.read_csv(conv_csv))
csvs.append(laftar_dp_csv)
csvs.append(laftar_eqopp0_csv)
csvs.append(laftar_eqopp1_csv)
csvs.append(laftar_eqodd_csv)
csvs.append(cfair_csv)
csvs.append(pd.read_csv(ffvae_csv))

# generate plot for each color ratio, pos_ratio (g_y_ratio)
for j in range(12):
    if j<=5:
      clr_ratio = clr_ratios[j]
      g_y_ratio = "(0.5, 0.5)"
      
    elif 6 <= j <= 8:
      clr_ratio = "(0.5, 0.5)"
      g_y_ratio = g_y_ratios[j-6]
      sensattr = "bck"

    elif 9 <= j <= 11:
      clr_ratio = "(0.5, 0.5)"
      g_y_ratio = g_y_ratios[j-9]
      sensattr = "color_gy"

    # parse for metric values (length of bar plots)
    values = []
    val = []
    for k in range(8):
      for l in range(7):
        tab = csvs[k]
        if l == 0:
          first_val = parse_csv(tab, table_type, clr_ratio, g_y_ratio, sensattr, mean_metric_names, 1)
          second_val = parse_csv(tab, table_type, clr_ratio, g_y_ratio, sensattr, mean_metric_names, 2)
          avg_val = (first_val + second_val)/2
          val.append(avg_val)
        elif l == 6:
          first_val = parse_csv(tab, table_type, clr_ratio, g_y_ratio, sensattr, mean_metric_names, 4)
          second_val = parse_csv(tab, table_type, clr_ratio, g_y_ratio, sensattr, mean_metric_names, 5)
          avg_val = (first_val + second_val)/2
          val.append(avg_val)
        else:
          val.append(parse_csv(tab, table_type, clr_ratio, g_y_ratio, sensattr, mean_metric_names, l))
      val = [l.tolist() for l in val]
      values.append(val)
      val = []
    values = np.asarray(values).squeeze()

    # parse for standard deviation metric values (error bars of bar plots)
    std_values = []
    std_val = []
    for k in range(8):
      for l in range(7):
        tab = csvs[k]
        if l == 0:
          first_val = parse_csv(tab, table_type, clr_ratio, g_y_ratio, sensattr, std_metric_names, 1)
          second_val = parse_csv(tab, table_type, clr_ratio, g_y_ratio, sensattr, std_metric_names, 2)
          avg_val = (first_val + second_val)/2
          std_val.append(avg_val)
        elif l == 6:
          first_val = parse_csv(tab, table_type, clr_ratio, g_y_ratio, sensattr, std_metric_names, 4)
          second_val = parse_csv(tab, table_type, clr_ratio, g_y_ratio, sensattr, std_metric_names, 5)
          avg_val = (first_val + second_val)/2
          std_val.append(avg_val)
        else:
          std_val.append(parse_csv(tab, table_type, clr_ratio, g_y_ratio, sensattr, std_metric_names, l))
      std_val = [l.tolist() for l in std_val]  
      std_values.append(std_val)
      std_val = []
    std_values = np.asarray(std_values).squeeze()
    
    # barplot code
    x = np.arange(len(labels))
    width = 0.6
    fig, ax = plt.subplots(figsize=(15,10))
    rects1 = ax.bar(x - 3*width/6, values[0], width/6, yerr=std_values[0], ecolor='gray', capsize=3, label='Mlp', color=clr[0])
    rects1 = ax.bar(x - 2*width/6, values[1], width/6, yerr=std_values[1], ecolor='gray', capsize=3, label='Cnn', color=clr[1])
    rects2 = ax.bar(x - 1*width/6, values[2], width/6, yerr=std_values[2], ecolor='gray', capsize=3, label='Laftr-DP', color=clr[2])
    rects3 = ax.bar(x - 0*width/6, values[3], width/6, yerr=std_values[3], ecolor='gray', capsize=3, label='Laftr-EqOpp0', color=clr[3])
    rects4 = ax.bar(x + 1*width/6, values[4], width/6, yerr=std_values[4], ecolor='gray', capsize=3, label='Laftr-EqOpp1', color=clr[4])
    rects5 = ax.bar(x + 2*width/6, values[5], width/6, yerr=std_values[5], ecolor='gray', capsize=3, label='Laftr-EqOdd', color=clr[5])
    rects6 = ax.bar(x + 3*width/6, values[6], width/6, yerr=std_values[6], ecolor='gray', capsize=3, label='Cfair', color=clr[6])
    rects8 = ax.bar(x + 4*width/6, values[7], width/6, yerr=std_values[7], ecolor='gray', capsize=3, label='Ffvae', color=clr[7])

    ax.set_ylabel('metric value', fontsize=25)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=25)
    ax.legend(bbox_to_anchor=(0.5, 1.17), loc='upper center', ncol=4, prop={'size': 22}, frameon=False)
    plt.margins(0)
    plt.gcf().subplots_adjust(bottom=0.2)
    ax.yaxis.grid()
    ax.set_axisbelow(True)
    ax.set_ylim([0,1.05])

    # labelling and saving plots
    if j<=5:
      label_name = "Sens: bck (u-elg, u-inelg) = " + clr_ratios[j]
    elif 6 <=j <=8:
      label_name = "Sens: bck (left-elg, left-inelg) = " + g_y_ratios[j-6]
    elif 9 <=j <=11:
      label_name = "Sens: pos (left-elg, left-inelg) = " + g_y_ratios[j-9]

    if j <= 12:
      plt.savefig("mnist_plots/" + str(label_name) + ".png", bbox_inches = 'tight',  pad_inches = 0)
