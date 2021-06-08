import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os 

if not os.path.exists("./csvs"):
    os.makedirs("./csvs")
if not os.path.exists("./adult-plots-std"):
    os.makedirs("./adult-plots-std")

# get specific rows from the wandb csv file
def parse_csv(tab, table_type, clr_ratio, sensattr, metric_names, l):
        return tab[(tab['table_type'] == table_type) & 
                      (tab['clr_ratio'] == clr_ratio) &
                      (tab['sensattr'] == sensattr)][metric_names[l]].values


# load processed files (mean taken over seeds and maximum metrics over hyperparameters)
mlp_csv = './adult/adult-mlp-mean_over_seed_max_over_hp.csv'
laftar_csv = pd.read_csv('./adult/adult-laftr-mean_over_seed_max_over_hp.csv')
cfair_mod_csv = pd.read_csv('./adult/adult-cfair-mean_over_seed_max_over_hp.csv')
ffvae_csv = './adult/adult-ffvae-mean_over_seed_max_over_hp.csv'
conv_csv = './adult/adult-conv-mean_over_seed_max_over_hp.csv'

laftar_dp_csv =     laftar_csv[laftar_csv["arch"] == "laftr-dp"]
laftar_eqodd_csv =  laftar_csv[laftar_csv["arch"] == "laftr-eqodd"]
laftar_eqopp0_csv = laftar_csv[laftar_csv["arch"] == "laftr-eqopp0"]
laftar_eqopp1_csv = laftar_csv[laftar_csv["arch"] == "laftr-eqopp1"]
cfair_csv =    cfair_mod_csv[cfair_mod_csv["arch"] == "cfair"]
cfair_eo_csv =    cfair_mod_csv[cfair_mod_csv["arch"] == "cfair-eo"]

# labels, metrics for plotting
models = ['Mlp', 'Laftr-DP', 'Laftr-EqOpp0', 'Laftr-EqOpp1', 'Laftr-EqOdd', 'Cfair', 'Cfair-EO', 'Ffvae']
labels = ['acc', 'p-acc', 'up-acc', 'DP', 'EqOpp0', 'EqOpp1', 'EqOdd']
mean_metric_names = ['acc_mean_max', '1-acc_mean_max', '0-acc_mean_max', 'DP_mean_max', 'EqOp0_mean_max', 'EqOp1_mean_max', 'EqOdd_mean_max']
std_metric_names = ['acc_std_corresp', '1-acc_std_corresp', '0-acc_std_corresp', 'DP_std_corresp', 'EqOp0_std_corresp', 'EqOp1_std_corresp', 'EqOdd_std_corresp']
clr = ['deepskyblue', 'khaki', 'lightcoral', 'mediumseagreen', 'lightslategrey', 'plum', 'turquoise']
clr_ratios = ["(0.5, 0.5)", "(0.1, 0.1)", "(0.01, 0.01)", "(0.66, 0.33)", "(0.33, 0.33)", "(0.06, 0.36)"]
table_type = 4
sensattr = "age"

# load csvs into an array
csvs = []
csvs.append(pd.read_csv(mlp_csv))
csvs.append(laftar_dp_csv)
csvs.append(laftar_eqopp0_csv)
csvs.append(laftar_eqopp1_csv)
csvs.append(laftar_eqodd_csv)
csvs.append(cfair_csv)
csvs.append(cfair_eo_csv)
csvs.append(pd.read_csv(ffvae_csv))


# generate plot for each color ratio
for j in range(7):
    if 0 <= j <= 5:
      clr_ratio = clr_ratios[j]
      sensattr = "age"
    elif j == 6:
      clr_ratio = "(0.5, 0.5)"
      sensattr = "sex"

    # parse for standard deviation metric values
    std_values = []
    std_val = []
    for k in range(8):
      for l in range(7):
        tab = csvs[k]
        if l == 0:
          first_val = parse_csv(tab, table_type, clr_ratio, sensattr, std_metric_names, 1)
          second_val = parse_csv(tab, table_type, clr_ratio, sensattr, std_metric_names, 2)
          avg_val = (first_val + second_val)/2
          std_val.append(avg_val)
        elif l == 6:
          first_val = parse_csv(tab, table_type, clr_ratio, sensattr, std_metric_names, 4)
          second_val = parse_csv(tab, table_type, clr_ratio, sensattr, std_metric_names, 5)
          avg_val = (first_val + second_val)/2
          std_val.append(avg_val)
        else:
          std_val.append(parse_csv(tab, table_type, clr_ratio, sensattr, std_metric_names, l))         
      std_values.append(std_val)
      std_val = []
    std_values = np.asarray(std_values).squeeze()

    # heatmap code
    x = np.arange(len(labels)) 
    width = 0.6 
    fig, ax = plt.subplots()
    std_values = np.round(np.nan_to_num(std_values), 2)
    im = ax.imshow(std_values)
    im.set_clim(vmin=0.0, vmax=0.1)
    plt.colorbar(im)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(models)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.gcf().subplots_adjust(top=0.15)

    for p in range(len(models)):
        for q in range(len(labels)):
            text = ax.text(q, p, ("%.2f" % std_values[p, q]).lstrip('0'),
                          ha="center", va="center", color="w")
            
    fig.tight_layout()

    # labelling and saving plots
    if j == 0:
      label_name = "age-ratio (0.5, 0.5)"
    elif j == 1:
      label_name = "age-ratio (0.1, 0.1)"
    elif j == 2:
      label_name = "age-ratio (0.01, 0.01)"
    elif j == 3:
      label_name = "age-ratio (0.66, 0.33)"
    elif j == 4:
      label_name = "age-ratio (0.33, 0.33)"
    elif j == 5:
      label_name = "age-ratio (0.06, 0.36)"
    elif j == 6:
      label_name = "sex-ratio (0.15, 0.38)"

    ax.set_title(label_name)

    if j <= 6:
      plt.savefig("adult-plots-std/" + str(label_name) + ".png", bbox_inches = 'tight', pad_inches = 0)