import csv
import os


def import_list(file=None):
    mylist = []
    if file is None:
        current_dir = os.path.dirname(__file__)
        with open(os.path.join(current_dir, 'ML_outputs/synergy_list.txt'), "r") as f:
            for line in f:
                mylist.append(line.strip())
    else:
        with open(file, "r") as f:
            for line in f:
                mylist.append(line.strip())
    return mylist


def models_load():
    import joblib
    current_dir = os.path.dirname(__file__)
    with open(os.path.join(current_dir, "ML_outputs/nust_models.pkl"),
              'rb') as input_file:
        nust_models = joblib.load(input_file)
    with open(os.path.join(current_dir, "ML_outputs/ust_models.pkl"),
              'rb') as input_file:
        ust_models = joblib.load(input_file)
    return nust_models["lasso"], ust_models["lasso"]


def model_features():
    nust_features = []
    current_dir = os.path.dirname(__file__)
    with open(os.path.join(current_dir, "ML_outputs/NUST_feature_coefs.csv"), "r") as file:
        reader = csv.DictReader(file)
        for line in reader:
            nust_features.append(line["Feature"].replace("OH-Cleveland", "OH-Cleveland Heights"))
    ust_features = []
    with open(os.path.join(current_dir, "ML_outputs/UST_feature_coefs.csv"), "r") as file:
        reader = csv.DictReader(file)
        for line in reader:
            ust_features.append(line["Feature"].replace("OH-Cleveland", "OH-Cleveland Heights"))
    return nust_features, ust_features