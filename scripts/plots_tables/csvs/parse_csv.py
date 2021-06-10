import pandas as pd
import os

if not os.path.exists("./scripts/plots_tables/csvs/processed_csvs/"):
    os.makedirs("./scripts/plots_tables/csvs/processed_csvs/")

csv_folder = './scripts/plots_tables/csvs/'
out_csv_folder = './scripts/plots_tables/csvs/processed_csvs/'

def filter_table(data_table, metric_cols, all_grouping_cols):
    for k, v in metric_cols.items():
        data_table[v] = data_table[k]
        data_table[v] = 1 - (data_table[v] - 1).abs()

    return data_table[list(metric_cols.values()) + all_grouping_cols]

def group_and_agg(tbl, mean_grouping_cols, metric_cols):
    grouped_tbl = tbl.groupby(mean_grouping_cols)

    avg_tbl = grouped_tbl[list(metric_cols.values())].agg(['mean'])
    avg_tbl.columns = ["_".join(x) for x in avg_tbl.columns.tolist()]
    avg_tbl.dropna()
    avg_tbl = avg_tbl.reset_index()

    std_tbl = grouped_tbl[list(metric_cols.values())].agg(['std'])
    std_tbl.columns = ["_".join(x) for x in std_tbl.columns.tolist()]
    std_tbl.dropna()
    std_tbl = std_tbl.reset_index()
    return grouped_tbl, avg_tbl, std_tbl


def group_and_max(tbl, max_grouping_cols, metric_cols):
    grouped_tbl = tbl.groupby(max_grouping_cols)
    col_list = [s + "_mean" for s in list(metric_cols.values())]
    avg_tbl = grouped_tbl[col_list].agg(['max'])
    avg_tbl.columns = ["_".join(x) for x in avg_tbl.columns.tolist()]
    avg_tbl.dropna()
    avg_tbl = avg_tbl.reset_index()
    return grouped_tbl, avg_tbl


for dataset in ['adult', 'mnist']:
    if dataset == 'adult':
        archs = ['mlp', 'laftr', 'cfair', 'ffvae']
    else:
        archs = ['mlp', 'conv', 'laftr', 'cfair', 'ffvae']

    for arch in archs:

        df_filename = os.path.join(csv_folder, dataset + '.csv')
        df = pd.read_csv(df_filename)

        if arch == "laftr":
            df = df[(df['arch'].isin(
                ["laftr-eqopp0", "laftr-eqopp1", "laftr-eqodd", "laftr-dp"]))]
        elif arch == "cfair":
            df = df[(df['arch'].isin(["cfair", "cfair-eo"]))]
        else:
            df = df[(df['arch'] == arch)]

        mean_filename = os.path.join(out_csv_folder, dataset + '-' + arch + '-mean_over_seed.csv')
        mean_max_filename = os.path.join(out_csv_folder, dataset + '-' + arch + '-mean_over_seed_max_over_hp.csv')
        table_num = 4

        if arch == "cfair":
            hyps = ['adv_coeff']
        elif arch == "laftr":
            hyps = ['fair_coeff']
        elif arch == "ffvae":
            hyps = ['gamma', 'alpha']
        else:
            hyps = []

        all_grouping_cols = ['clr_ratio',
                            'sensattr', 'arch', 'seed'] + hyps
        mean_grouping_cols = ['clr_ratio', 'sensattr', 'arch'] + hyps
        max_grouping_cols = ['clr_ratio', 'sensattr', 'arch']

        metric_cols = {'TEST/DP': 'DP', 'TEST/0-accuracy': '0-acc',
                        'TEST/1-accuracy': '1-acc', 'TEST/EqOp(y=0)': 'EqOp0',
                        'TEST/EqOp(y=1)': 'EqOp1'}

        if dataset == "mnist":
            all_grouping_cols = all_grouping_cols + ['g_y_ratio']
            mean_grouping_cols = mean_grouping_cols + ['g_y_ratio']
            max_grouping_cols = max_grouping_cols + ['g_y_ratio']

            metric_cols = {'es/DP/test': 'DP', 'es/0-accuracy/test': '0-acc',
                        'es/1-accuracy/test': '1-acc', 'es/EqOp(y=0)/test': 'EqOp0',
                        'es/EqOp(y=1)/test': 'EqOp1'}

        print(df.shape)
        table_df = filter_table(df, metric_cols, all_grouping_cols)
        print(table_df.shape)

        (grouped_tbl_mean, avg_tbl, std_tbl) = group_and_agg(table_df, mean_grouping_cols, metric_cols)
        print(avg_tbl.shape)
        avg_tbl.to_csv(mean_filename, index=False)

        (grouped_tbl_max, max_tbl) = group_and_max(avg_tbl, max_grouping_cols, metric_cols)
        print(max_tbl.shape)

        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            for k, v in metric_cols.items():
                pp = std_tbl.loc[grouped_tbl_max[v+"_mean"].idxmax()]
                max_tbl.index = pp.index
                new_v = v + '_std' + '_corresp'
                max_tbl[new_v] = pp[v + '_std']

        max_tbl['table_type'] = table_num
        max_tbl = max_tbl[['table_type'] + max_tbl.columns.tolist()[:-1]]
        print(max_tbl.shape)

        max_tbl.to_csv(mean_max_filename, index=False)
        print(max_tbl.columns)
