import pandas as pd
import os

csv_folder = './scripts/plots_tables/csvs/processed_csvs/'

def get_mnist_tables():

    for arch in ['mlp', 'conv', 'laftr-dp', 'laftr-eqopp1', 'laftr-eqopp0', 'laftr-eqodd', 'cfair', 'ffvae']:
        
        print("Dataset: Mnist, Model: ", arch)
        
        if 'laftr' in arch:
            model = 'laftr'
        elif 'cfair' in arch:
            model = 'cfair'
        else:
            model = arch

        filename = os.path.join(csv_folder, 'mnist-' + model + '-mean_over_seed_max_over_hp.csv')
        df = pd.read_csv(filename)
        df = df[df['arch'] == arch]

        columns = {'DP_mean_max': 'DP_mean', '0-acc_mean_max': '0-acc_mean',
                '1-acc_mean_max': '1-acc_mean', 'EqOp0_mean_max': 'EqOp0_mean',
                'EqOp1_mean_max': 'EqOp1_mean'}

        for k,v in columns.items():
            df[v] = df[k].round(decimals=2)

        grouped_df = df.groupby(['table_type', 'clr_ratio', 'g_y_ratio', 'sensattr', 'arch'])

        grouped_df = grouped_df[list(columns.values())].agg(tuple).reset_index()

        grouped_df["OverleafRow"] = grouped_df["0-acc_mean"].astype(str) + " & " + grouped_df["1-acc_mean"].astype(str) + " & " + grouped_df["DP_mean"].astype(str) + " & " + grouped_df["EqOp0_mean"].astype(str) + " & " + grouped_df["EqOp1_mean"].astype(str)

        clr_ratios_b = ["(0.5, 0.5)", "(0.1, 0.1)", "(0.01, 0.01)", "(0.001, 0.001)"]
        clr_ratios_ub = ["(0.5, 0.5)", "(0.1, 0.9)", "(0.01, 0.99)"]
        clr_ratio = "(0.5, 0.5)"

        g_y_ratios = ["(0.5, 0.5)", "(0.75, 0.25)", "(0.9, 0.1)"]
        g_y_ratio = "(0.5, 0.5)"

        print("Setting 1")
        for i in clr_ratios_b:
            for idx, row in grouped_df[grouped_df["table_type"] == 4].iterrows():
                if row["sensattr"] == "bck" and row["arch"] == arch:
                    if row["clr_ratio"] == i and row["g_y_ratio"] == g_y_ratio:
                        acc_avg = round((row["0-acc_mean"][0] + row["1-acc_mean"][0])/2, 2)
                        eqodd_avg = round((row["EqOp0_mean"][0] + row["EqOp1_mean"][0])/2, 2)
                        mytable = row["OverleafRow"].maketrans({"(": "", ")": "", ",": ""})
                        str_out = str(acc_avg) + " & " + row["OverleafRow"].translate(mytable) + " & " + str(eqodd_avg)
                        print(row["clr_ratio"] + " & " + str_out + "\cr \hline")

        print("\n\n")

        print("Setting 2")
        for i in clr_ratios_ub:
            for idx, row in grouped_df[grouped_df["table_type"] == 4].iterrows():
                if row["sensattr"] == "bck" and row["arch"] == arch:
                    if row["clr_ratio"] == i and row["g_y_ratio"] == g_y_ratio:
                        acc_avg = round((row["0-acc_mean"][0] + row["1-acc_mean"][0])/2, 2)
                        eqodd_avg = round((row["EqOp0_mean"][0] + row["EqOp1_mean"][0])/2, 2)
                        mytable = row["OverleafRow"].maketrans({"(": "", ")": "", ",": ""})
                        str_out = str(acc_avg) + " & " + row["OverleafRow"].translate(mytable) + " & " + str(eqodd_avg)
                        print(row["clr_ratio"] + " & " + str_out + "\cr \hline")

        print("\n\n")

        print("Setting 3")
        for i in g_y_ratios:
            for idx, row in grouped_df[grouped_df["table_type"] == 4].iterrows():
                if row["sensattr"] == "bck" and row["arch"] == arch:
                    if row["clr_ratio"] == clr_ratio and row["g_y_ratio"] == i:
                        acc_avg = round((row["0-acc_mean"][0] + row["1-acc_mean"][0])/2, 2)
                        eqodd_avg = round((row["EqOp0_mean"][0] + row["EqOp1_mean"][0])/2, 2)
                        mytable = row["OverleafRow"].maketrans({"(": "", ")": "", ",": ""})
                        str_out = str(acc_avg) + " & " + row["OverleafRow"].translate(mytable) + " & " + str(eqodd_avg)
                        print(row["g_y_ratio"] + " & " + str_out + "\cr \hline")

        print("\n\n")

        print("Setting 4")
        for i in g_y_ratios:
            for idx, row in grouped_df[grouped_df["table_type"] == 4].iterrows():
                if row["sensattr"] == "color_gy" and row["arch"] == arch:
                    if row["clr_ratio"] == clr_ratio and row["g_y_ratio"] == i:
                        acc_avg = round((row["0-acc_mean"][0] + row["1-acc_mean"][0])/2, 2)
                        eqodd_avg = round((row["EqOp0_mean"][0] + row["EqOp1_mean"][0])/2, 2)
                        mytable = row["OverleafRow"].maketrans({"(": "", ")": "", ",": ""})
                        str_out = str(acc_avg) + " & " + row["OverleafRow"].translate(mytable) + " & " + str(eqodd_avg)
                        print(row["g_y_ratio"] + " & " + str_out + "\cr \hline")

        print("\n\n")

def get_adult_tables():

    for arch in ['mlp', 'laftr-dp', 'laftr-eqopp1', 'laftr-eqopp0', 'laftr-eqodd', 'cfair', 'cfair-eo', 'ffvae']:
        
        print("Dataset: Adult, Model: ", arch)

        if 'laftr' in arch:
            model = 'laftr'
        elif 'cfair' in arch:
            model = 'cfair'
        else:
            model = arch

        filename = os.path.join(csv_folder, 'adult-' + model + '-mean_over_seed_max_over_hp.csv')
        df = pd.read_csv(filename)
        df = df[df['arch'] == arch]

        columns = {'DP_mean_max': 'DP_mean', '0-acc_mean_max': '0-acc_mean',
                '1-acc_mean_max': '1-acc_mean', 'EqOp0_mean_max': 'EqOp0_mean',
                'EqOp1_mean_max': 'EqOp1_mean'}

        for k,v in columns.items():
            df[v] = df[k].round(decimals=2)

        grouped_df = df.groupby(['table_type', 'clr_ratio', 'sensattr', 'arch'])

        grouped_df = grouped_df[list(columns.values())].agg(tuple).reset_index()

        grouped_df["OverleafRow"] = grouped_df["1-acc_mean"].astype(str) + " & " + grouped_df["0-acc_mean"].astype(str) + " & " + grouped_df["DP_mean"].astype(str) + " & " + grouped_df["EqOp0_mean"].astype(str) + " & " + grouped_df["EqOp1_mean"].astype(str)

        clr_ratios_b = ["(0.5, 0.5)", "(0.1, 0.1)", "(0.01, 0.01)"]
        clr_ratios_ub = ["(0.5, 0.5)", "(0.66, 0.33)", "(0.06, 0.36)"]

        print("Setting 1")
        for i in clr_ratios_b:
            for idx, row in grouped_df[grouped_df["table_type"] == 4].iterrows():
                if row["sensattr"] == "age" and row["arch"] == arch:
                    if row["clr_ratio"] == i:
                        acc_avg = round((row["0-acc_mean"][0] + row["1-acc_mean"][0])/2, 2)
                        eqodd_avg = round((row["EqOp0_mean"][0] + row["EqOp1_mean"][0])/2, 2)
                        mytable = row["OverleafRow"].maketrans({"(": "", ")": "", ",": ""})
                        str_out = str(acc_avg) + " & " + row["OverleafRow"].translate(mytable) + " & " + str(eqodd_avg)
                        print(row["clr_ratio"] + " & " + str_out + " \\\ \hline")

        print("\n\n")

        print("Setting 2")
        for i in clr_ratios_ub:
            for idx, row in grouped_df[grouped_df["table_type"] == 4].iterrows():
                if row["sensattr"] == "age" and row["arch"] == arch:
                    if row["clr_ratio"] == i:
                        acc_avg = round((row["0-acc_mean"][0] + row["1-acc_mean"][0])/2, 2)
                        eqodd_avg = round((row["EqOp0_mean"][0] + row["EqOp1_mean"][0])/2, 2)
                        mytable = row["OverleafRow"].maketrans({"(": "", ")": "", ",": ""})
                        str_out = str(acc_avg) + " & " + row["OverleafRow"].translate(mytable) + " & " + str(eqodd_avg)
                        print(row["clr_ratio"] + " & " + str_out + " \\\ \hline")
        
        print("\n\n")

get_mnist_tables()
get_adult_tables()