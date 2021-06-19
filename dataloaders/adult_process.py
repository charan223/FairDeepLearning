import numpy as np
import sys

# make noNULL file with: grep -v NULL rawdata_mkmk01.csv | cut -f1,3,4,6- -d, > rawdata_mkmk01_noNULL.csv
EPS = 1e-8

UNPRIVILEGED_ARR = np.array([1, 0])
PRIVILEGED_ARR = np.array([0, 1])
SENSATTR_RULES = {
    "sex": lambda x, thresh1, thresh2: UNPRIVILEGED_ARR
    if x == "Female"
    else PRIVILEGED_ARR,
    "age": lambda x, thresh1, thresh2: UNPRIVILEGED_ARR
    if thresh1 <= int(x) < thresh2
    else PRIVILEGED_ARR,
}


def binarize_sensattr(val, sensattr, thresh1, thresh2):
    return SENSATTR_RULES[sensattr](val, thresh1, thresh2)

def bucket(x, buckets):
    x = float(x)
    n = len(buckets)
    label = n
    for i in range(len(buckets)):
        if x <= buckets[i]:
            label = i
            break
    template = [0.0 for j in range(n + 1)]
    template[label] = 1.0
    return template

def onehot(x, choices):
    if not x in choices:
        print('could not find "{}" in choices'.format(x))
        print(choices)
        raise Exception()
    label = choices.index(x)
    template = [0.0 for j in range(len(choices))]
    template[label] = 1.0
    return template

def continuous(x):
    return [float(x)]

def parse_row(row, headers, headers_use, fns, sensitive, target, thresh1, thresh2):
    new_row_dict = {}
    for i in range(len(row)):
        x = row[i]
        hdr = headers[i]
        new_row_dict[hdr] = fns[hdr](x)
        if hdr == sensitive:
            sens_att = binarize_sensattr(x, hdr, thresh1, thresh2)

    label = new_row_dict[target]
    new_row = []
    for h in headers_use:
        new_row = new_row + new_row_dict[h]
    return new_row, label, sens_att

def whiten(X, mn, std):
    mntile = np.tile(mn, (X.shape[0], 1))
    stdtile = np.maximum(np.tile(std, (X.shape[0], 1)), EPS)
    X = X - mntile
    X = np.divide(X, stdtile)
    return X

def get_adult_data(target, sensitive, clr_ratio):
    thresh1, thresh2 = 0, 0
    if sensitive == "age":
        if clr_ratio == [0.5, 0.5]:
            thresh1, thresh2 = 25, 44
        elif clr_ratio == [0.66, 0.33]:
            thresh1, thresh2 = 38, 60
        elif clr_ratio == [0.33, 0.33]:
            thresh1, thresh2 = 28, 40
        elif clr_ratio == [0.1, 0.1]:
            thresh1, thresh2 = 32, 36
        elif clr_ratio == [0.06, 0.36]:
            thresh1, thresh2 = 0, 30
        elif clr_ratio == [0.01, 0.01]:
            thresh1, thresh2 = 71, 100

    f_in_tr = "./data/adult/adult.data"
    f_in_te = "./data/adult/adult.test"

    REMOVE_MISSING = True
    MISSING_TOKEN = "?"

    headers = "age,workclass,fnlwgt,education,education-num,marital-status,occupation,relationship,race,sex,capital-gain,capital-loss,hours-per-week,native-country,income".split(
        ","
    )
    headers_use = "age,workclass,education,education-num,marital-status,occupation,relationship,race,capital-gain,capital-loss,hours-per-week,native-country".split(
        ","
    )

    options = {
        "age": "buckets",
        "workclass": "Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked",
        "fnlwgt": "continuous",
        "education": "Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool",
        "education-num": "continuous",
        "marital-status": "Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse",
        "occupation": "Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces",
        "relationship": "Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried",
        "race": "White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black",
        "sex": "Female, Male",
        "capital-gain": "continuous",
        "capital-loss": "continuous",
        "hours-per-week": "continuous",
        "native-country": "United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands",
        "income": " <=50K,>50K",
    }

    buckets = {"age": [18, 25, 30, 35, 40, 45, 50, 55, 60, 65]}

    options = {k: [s.strip() for s in sorted(options[k].split(","))] for k in options}

    print(f"OPTIONS = {options}")
    fns = {
        "age": lambda x: bucket(x, buckets["age"]),
        "workclass": lambda x: onehot(x, options["workclass"]),
        "fnlwgt": lambda x: continuous(x),
        "education": lambda x: onehot(x, options["education"]),
        "education-num": lambda x: continuous(x),
        "marital-status": lambda x: onehot(x, options["marital-status"]),
        "occupation": lambda x: onehot(x, options["occupation"]),
        "relationship": lambda x: onehot(x, options["relationship"]),
        "race": lambda x: onehot(x, options["race"]),
        "sex": lambda x: onehot(x, options["sex"]),
        "capital-gain": lambda x: continuous(x),
        "capital-loss": lambda x: continuous(x),
        "hours-per-week": lambda x: continuous(x),
        "native-country": lambda x: onehot(x, options["native-country"]),
        "income": lambda x: onehot(x.strip("."), options["income"]),
    }

    D = {}
    for f, phase in [(f_in_tr, "training"), (f_in_te, "test")]:
        dat = [s.strip().split(",") for s in open(f, "r").readlines()]

        X = []
        Y = []
        A = []
        print(phase)

        for r in dat:
            row = [s.strip() for s in r]
            if MISSING_TOKEN in row and REMOVE_MISSING:
                continue
            if row in ([""], ["|1x3 Cross validator"]):
                continue
            newrow, label, sens_att = parse_row(
                row, headers, headers_use, fns, sensitive, target, thresh1, thresh2
            )
            X.append(newrow)
            Y.append(label)
            A.append(sens_att)

        npX = np.array(X)
        npY = np.array(Y)
        npA = np.array(A)
        npA = np.expand_dims(npA[:, 1], 1)

        D[phase] = {}
        D[phase]["X"] = npX
        D[phase]["Y"] = npY
        D[phase]["A"] = npA

        print(npX.shape)
        print(npY.shape)
        print(npA.shape)

    # should do normalization and centring
    mn = np.mean(D["training"]["X"], axis=0)
    std = np.std(D["training"]["X"], axis=0)
    print(mn, std)
    D["training"]["X"] = whiten(D["training"]["X"], mn, std)
    D["test"]["X"] = whiten(D["test"]["X"], mn, std)

    n = D["training"]["X"].shape[0]
    unshuf = np.arange(n)
    valid_pct = 0.2
    valid_ct = int(n * valid_pct)
    valid_inds = unshuf[:valid_ct]
    train_inds = unshuf[valid_ct:]

    D["training"]["X"] = D["training"]["X"].astype(np.float32).squeeze()
    D["training"]["Y"] = np.argmax(
        D["training"]["Y"].astype(np.float32), axis=1
    ).squeeze()
    D["training"]["A"] = D["training"]["A"].astype(np.float32).squeeze()

    D["test"]["X"] = D["test"]["X"].astype(np.float32).squeeze()
    D["test"]["Y"] = np.argmax(D["test"]["Y"].astype(np.float32), axis=1).squeeze()
    D["test"]["A"] = D["test"]["A"].astype(np.float32).squeeze()

    total_p = sum(D["test"]["A"])
    total_up = D["test"]["A"].shape[0] - sum(D["test"]["A"])

    total_e = sum(D["test"]["Y"])
    total_ie = D["test"]["Y"].shape[0] - sum(D["test"]["Y"])

    priv_elg = sum(D["test"]["Y"] * D["test"]["A"])
    priv_inelg = total_p - priv_elg
    unpriv_elg = total_e - priv_elg
    unpriv_inelg = total_ie - priv_inelg

    min_num = int(min(priv_elg, priv_inelg, unpriv_elg, unpriv_inelg))

    print("Privileged eligible", priv_elg)
    print("Privileged ineligible", priv_inelg)
    print("Unprivileged eligible", unpriv_elg)
    print("Unprivileged ineligible", unpriv_inelg)

    priv_elg_inds = D["test"]["Y"] * D["test"]["A"] > 0
    priv_elg_test_X = D["test"]["X"][priv_elg_inds][0:min_num]
    priv_elg_test_Y = D["test"]["Y"][priv_elg_inds][0:min_num]
    priv_elg_test_A = D["test"]["A"][priv_elg_inds][0:min_num]

    priv_inelg_inds = (1 - D["test"]["Y"]) * D["test"]["A"] > 0
    priv_inelg_test_X = D["test"]["X"][priv_inelg_inds][0:min_num]
    priv_inelg_test_Y = D["test"]["Y"][priv_inelg_inds][0:min_num]
    priv_inelg_test_A = D["test"]["A"][priv_inelg_inds][0:min_num]

    unpriv_elg_inds = D["test"]["Y"] * (1 - D["test"]["A"]) > 0
    unpriv_elg_test_X = D["test"]["X"][unpriv_elg_inds][0:min_num]
    unpriv_elg_test_Y = D["test"]["Y"][unpriv_elg_inds][0:min_num]
    unpriv_elg_test_A = D["test"]["A"][unpriv_elg_inds][0:min_num]

    unpriv_inelg_inds = (1 - D["test"]["Y"]) * (1 - D["test"]["A"]) > 0
    unpriv_inelg_test_X = D["test"]["X"][unpriv_inelg_inds][0:min_num]
    unpriv_inelg_test_Y = D["test"]["Y"][unpriv_inelg_inds][0:min_num]
    unpriv_inelg_test_A = D["test"]["A"][unpriv_inelg_inds][0:min_num]

    print(
        D["test"]["Y"][priv_elg_inds].shape[0],
        D["test"]["Y"][priv_inelg_inds].shape[0],
        D["test"]["Y"][unpriv_elg_inds].shape[0],
        D["test"]["Y"][unpriv_inelg_inds].shape[0],
    )

    D["test"]["X"] = np.concatenate(
        (priv_elg_test_X, priv_inelg_test_X, unpriv_elg_test_X, unpriv_inelg_test_X),
        axis=0,
    )
    D["test"]["Y"] = np.concatenate(
        (priv_elg_test_Y, priv_inelg_test_Y, unpriv_elg_test_Y, unpriv_inelg_test_Y),
        axis=0,
    )
    D["test"]["A"] = np.concatenate(
        (priv_elg_test_A, priv_inelg_test_A, unpriv_elg_test_A, unpriv_inelg_test_A),
        axis=0,
    )

    data_dict = {
        "x_train": D["training"]["X"],
        "x_test": D["test"]["X"],
        "y_train": D["training"]["Y"],
        "y_test": D["test"]["Y"],
        "attr_train": D["training"]["A"],
        "attr_test": D["test"]["A"],
        "train_inds": train_inds,
        "valid_inds": valid_inds,
    }
    return data_dict