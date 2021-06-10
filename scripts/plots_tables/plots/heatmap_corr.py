import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

adult_csv = "./scripts/plots_tables/csvs/adult.csv"
mnist_csv = "./scripts/plots_tables/csvs/mnist.csv"
out_plots_folder = './scripts/plots_tables/plots/corrheatmaps/'
if not os.path.exists(out_plots_folder):
    os.makedirs(out_plots_folder)

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
MNIST_CSV_COLS = {
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

ADULT_CSV_COLS = {
    'beta_1': 'be',
    'beta_2': 'bo',
    'egr': 'le',
    'ogr': 'lo',
    'TEST/0-accuracy': 'p-acc',
    'TEST/1-accuracy': 'up-acc',
    'TEST/DP': 'DP',
    'TEST/EqOp(y=0)': 'EqOpp0',
    'TEST/EqOp(y=1)': 'EqOpp1'
}
new_metric_cols = ["acc", "EqOdd"]
metric_cols = list(METRICS.keys())
data_features = ['be', 'bo', 'le', 'lo']

# change column names, calculate all metrics
def preprocess_data(df, csv_cols):
    for k, v in csv_cols.items():
        df[v] = df[k]

    # take reference as 0.5 for be, bo, le, lo values
    for data_col in data_features:
        df[data_col] = abs(0.5 - df[data_col])

    for k, func in METRICS.items():
        df.loc[:, k] = df.apply(func, axis=1)

    return df[list(csv_cols.values()) + new_metric_cols]
    #return df[data_features + metric_cols]

# plot heatmaps given correlation values
def plot_data(corr_data, plot_filename):
    if 'adult' in plot_filename:
        label_names = ['Mlp', 'Laftr-DP', 'Laftr-EqOpp0',
                       'Laftr-EqOpp1', 'Laftr-EqOdd', 'Cfair', 'Cfair-EO', 'Ffvae']
    else:
        label_names = ['Mlp', 'Cnn', 'Laftr-DP', 'Laftr-EqOpp0',
                       'Laftr-EqOpp1', 'Laftr-EqOdd', 'Cfair', 'Ffvae']

    z = abs(corr_data)

    fig, ax = plt.subplots()
    im = ax.imshow(z, cmap="OrRd")
    im.set_clim(vmin=0.3, vmax=1.3)
    plt.colorbar(im)
    ax.set_xticks(np.arange(len(metric_cols)))
    ax.set_yticks(np.arange(len(label_names)))
    ax.set_xticklabels(metric_cols)
    ax.set_yticklabels(label_names)
    plt.setp(ax.get_xticklabels(), rotation=45,
             ha="right", rotation_mode="anchor")
    plt.gcf().subplots_adjust(top=0.15)

    for p in range(len(label_names)):
        for q in range(len(metric_cols)):
            text = ax.text(q, p, ("%.2f" % z[p, q]).lstrip('0'),
                           ha="center", va="center", color="k")

    fig.tight_layout()
    plt.savefig(plot_filename, bbox_inches='tight', pad_inches=0)



adult_data = pd.read_csv(adult_csv, sep=',')
mnist_data = pd.read_csv(mnist_csv, sep=',')

adult_setting1 = [(0.5, 0.5), (0.1, 0.1), (0.01, 0.01)]
adult_setting2 = [(0.5, 0.5), (0.66, 0.33), (0.06, 0.36)]
mnist_setting1 = [(0.5, 0.5), (0.1, 0.1), (0.01, 0.01), (0.001, 0.001)]
mnist_setting2 = [(0.5, 0.5), (0.1, 0.9), (0.01, 0.99)]
mnist_setting3 = [(0.5, 0.5), (0.75, 0.25), (0.9, 0.1)]
mnist_setting4 = [(0.5, 0.5), (0.75, 0.25), (0.9, 0.1)]


data = []
data.append(adult_data[(adult_data['sensattr'] == "age") & (adult_data['beta_1'].isin([
            x[0] for x in adult_setting1])) & (adult_data['beta_2'].isin([x[1] for x in adult_setting1]))])
data.append(adult_data[(adult_data['sensattr'] == "age") & (adult_data['beta_1'].isin([
            x[0] for x in adult_setting2])) & (adult_data['beta_2'].isin([x[1] for x in adult_setting2]))])
data.append(mnist_data[(mnist_data['sensattr'] == "bck") & (mnist_data['beta_1'].isin([
            x[0] for x in mnist_setting1])) & (mnist_data['beta_2'].isin([x[1] for x in mnist_setting1]))])
data.append(mnist_data[(mnist_data['sensattr'] == "bck") & (mnist_data['beta_1'].isin([
            x[0] for x in mnist_setting2])) & (mnist_data['beta_2'].isin([x[1] for x in mnist_setting2]))])
data.append(mnist_data[(mnist_data['sensattr'] == "bck") & (mnist_data['egr'].isin([
            x[0] for x in mnist_setting3])) & (mnist_data['ogr'].isin([x[1] for x in mnist_setting3]))])
data.append(mnist_data[(mnist_data['sensattr'] == "color_gy") & (mnist_data['egr'].isin([
            x[0] for x in mnist_setting4])) & (mnist_data['ogr'].isin([x[1] for x in mnist_setting4]))])

settings = ['adult1', 'adult2', 'mnist1', 'mnist2', 'mnist3', 'mnist4']
for k in range(len(settings)):
    full_z = np.zeros((8, 7))

    if 'adult' in settings[k]:
        prjs = ['mlp', 'laftr-dp', 'laftr-eqopp0', 'laftr-eqopp1',
                'laftr-eqodd', 'cfair', 'cfair-eo', 'ffvae']
        csv_cols = ADULT_CSV_COLS
    else:
        prjs = ['mlp', 'conv', 'laftr-dp', 'laftr-eqopp0',
                'laftr-eqopp1', 'laftr-eqodd', 'cfair', 'ffvae']
        csv_cols = MNIST_CSV_COLS

    for i in range(len(prjs)):
        prj = prjs[i]

        # parse csv
        df = data[k]
        df = df[(df['arch'] == prj)]
        data_processed = preprocess_data(df, csv_cols)
        plot_dataset = data_processed[data_features + metric_cols]

        # get correlation between dataset and metric columns
        corr_dataset = plot_dataset.corr(method='spearman')
        corr_out = corr_dataset.loc[data_features, metric_cols]

        # settings 1, 2 in paper
        if "1" in settings[k] or "2" in settings[k]:
            # use p=0 for be, p=1 for bo
            p = 0
        else:  # for settings 3, 4 in paper
            # use p=2 for le, p=3 for lo
            p = 2

        # correlation values of all models in each file
        full_z[i] = corr_out.values[p, :7]

        # settings 1, 2 in paper
        if "1" in settings[k] or "2" in settings[k]:
            # use p=0 for be, p=1 for bo
            p = 1
        else:  # for settings 3, 4 in paper
            # use p=2 for le, p=3 for lo
            p = 3

        # correlation values of all models in each file
        full_z[i] += corr_out.values[p, :7]

    plot_filename = out_plots_folder + "corr_" + settings[k] + ".png"
    plot_data(full_z/2, plot_filename)
