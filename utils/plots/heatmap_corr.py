import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

if not os.path.exists("./results"):
    os.makedirs("./results")

# compute acc, EqOdd metrics from other metrics
METRICS = {
    "acc": lambda x: (x['p-acc'] + x['up-acc']) / 2,
    "p-acc": lambda x: x['p-acc'],
    "up-acc": lambda x: x['up-acc'],
    "DP": lambda x: x["DP"],
    "EqOpp0": lambda x: x["EqOpp0"],
    "EqOpp1": lambda x: x["EqOpp1"],
    "EqOdd": lambda x: (x["EqOpp0"] + x["EqOpp1"]) / 2,
}

# column names (keys) in the csv generated from wandb and their corresponding short names (values)
CSV_COLS = {
    'beta_1': 'be',
    'beta_2': 'bo',
    'egr': 'le',
    'ogr': 'lo',
    'es/0-accuracy/test': 'p-acc',
    'es/1-accuracy/test': 'up-acc',
    'es/DP/test': 'DP',
    'es/EqOp(y=0)/test': 'EqOpp0',
    'es/EqOp(y=1)/test': 'EqOpp1'
}

metric_cols = list(METRICS.keys())
dataset_cols = list(CSV_COLS.values())[0:4]

# change column names, calculate all metrics
def preprocess_data(df):
    for k,v in CSV_COLS.items():
        df[v] = df[k]

    # take reference as 0.5 for be, bo, le, lo values
    for data_col in dataset_cols:
        df[data_col] = abs(0.5 - df[data_col])
    
    for k, func in METRICS.items():
        df.loc[:, k] = df.apply(func, axis=1)

    return df[dataset_cols + metric_cols]

# plot heatmaps given correlation values
def plot_data(corr_data, plot_filename):
    if 'adult' in plot_filename:
      label_names = ['Mlp', 'Laftr-DP', 'Laftr-EqOpp0', 'Laftr-EqOpp1', 'Laftr-EqOdd', 'Cfair', 'Cfair-EO', 'Ffvae']
    else:
      label_names = ['Mlp', 'Cnn', 'Laftr-DP', 'Laftr-EqOpp0', 'Laftr-EqOpp1', 'Laftr-EqOdd', 'Cfair', 'Ffvae']

    z = corr_data

    fig, ax = plt.subplots()
    im = ax.imshow(z,cmap="OrRd")
    im.set_clim(vmin=0.0, vmax=1.2)
    plt.colorbar(im)
    ax.set_xticks(np.arange(len(metric_cols)))
    ax.set_yticks(np.arange(len(label_names)))
    ax.set_xticklabels(metric_cols)
    ax.set_yticklabels(label_names)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.gcf().subplots_adjust(top=0.15)

    for p in range(len(label_names)):
        for q in range(len(metric_cols)):
            text = ax.text(q, p, ("%.2f" % z[p, q]).lstrip('0'),
                          ha="center", va="center", color="k")
            
    fig.tight_layout()
    plt.savefig(plot_filename, bbox_inches = 'tight', pad_inches = 0)


files = ['adult1.csv', 'adult2.csv', 'adult12.csv', 'mnist12.csv', 'mnist1.csv', 'mnist2.csv', 'mnist3.csv', 'mnist4.csv', 'mnist34.csv']

for file in files:
  full_z = np.zeros((8, 7))

  if 'adult' in file:
    prjs = ['mlp', 'laftr-dp', 'laftr-eqopp0', 'laftr-eqopp1', 'laftr-eqodd', 'cfair', 'cfair-eo', 'ffvae']
  else:
    prjs = ['mlp', 'conv', 'laftr-dp', 'laftr-eqopp0', 'laftr-eqopp1', 'laftr-eqodd', 'cfair', 'ffvae']

  for i in range(len(prjs)):
    prj = prjs[i]

    # parse csv
    csvfile = f'data/{file}'
    df = pd.read_csv(csvfile, sep=',')
    df = df[(df['arch']==prj)]
    data = preprocess_data(df)
    plot_dataset = data[dataset_cols + metric_cols]

    # get correlation between dataset and metric columns
    corr_dataset = plot_dataset.corr(method='spearman')
    corr_out = corr_dataset.loc[dataset_cols, metric_cols]

    # settings 1, 2 in paper
    if "1" in file or "2" in file:
      # use p=0 for be, p=1 for bo   
      p = 0
    else: # for settings 3, 4 in paper
      # use p=2 for le, p=3 for lo   
      p = 2

    # correlation values of all models in each file
    full_z[i] = corr_out.values[p,:7]

  plot_filename = "./results/corr_" + file.split('.')[0] + ".png"
  plot_data(full_z, plot_filename)
