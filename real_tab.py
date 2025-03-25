# import libraries
import tarfile
import os
import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import entropy
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import time
import pandas as pd
import warnings
import string

warnings.filterwarnings("ignore")

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from numpy.random import default_rng

# from treeple.ensemble import ObliqueRandomForestClassifier
from tabpfn import TabPFNClassifier

from ydf import RandomForestLearner

import pickle

# define a function to get X and y given a file


def get_X_y(f, root="./data/", cohort=[], verbose=False):
    df = pd.read_csv(root + f)
    non_features = [
        "Run",
        "Sample",
        "Library",
        "Cancer Status",
        "Tumor type",
        "Stage",
        "Library volume (uL)",
        "Library Volume",
        "UIDs Used",
        "Experiment",
        "P7",
        "P7 Primer",
        "MAF",
    ]
    sample_ids = df["Sample"]
    # if sample is contains "Run" column, remove it
    for i, sample_id in enumerate(sample_ids):
        if "." in sample_id:
            sample_ids[i] = sample_id.split(".")[1]
    target = "Cancer Status"
    y = df[target]
    # convert the labels to 0 and 1
    y = y.replace("Healthy", 0)
    y = y.replace("Cancer", 1)
    # remove the non-feature columns if they exist
    for col in non_features:
        if col in df.columns:
            df = df.drop(col, axis=1)
    nan_cols = df.isnull().all(axis=0).to_numpy()
    # drop the columns with all nan values
    df = df.loc[:, ~nan_cols]
    # if cohort is not None, filter the samples
    if cohort is not None:
        # filter the rows with cohort1 samples
        X = df[sample_ids.isin(cohort)]
        y = y[sample_ids.isin(cohort)]
    else:
        X = df
    if "Wise" in f:
        # replace nans with zero
        X = X.fillna(0)
    # impute the nan values with the mean of the column
    X = X.fillna(X.mean(axis=0))
    # check if there are nan values
    # nan_rows = X.isnull().any(axis=1)
    nan_cols = X.isnull().all(axis=0)
    # remove the columns with all nan values
    X = X.loc[:, ~nan_cols]
    if verbose:
        if nan_cols.sum() > 0:
            print(f)
            print(f"nan_cols: {nan_cols.sum()}")
            print(f"X shape: {X.shape}, y shape: {y.shape}")
        else:
            print(f)
            print(f"X shape: {X.shape}, y shape: {y.shape}")
    # X = X.dropna()
    # y = y.drop(nan_rows.index)

    return X, y


if __name__ == "__main__":
    # Load the real datasets
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DIRECTORY = os.path.join(BASE_DIR, "data")
    SEED = 23
    N_JOBS = 40
    IDX = 41
    IDX_END = 42
    N_ITR = 1
    N_EST = 6000

    filelist = sorted(os.listdir(DIRECTORY))

    sample_list_file = DIRECTORY + "/AllSamples.MIGHT.Passed.samples.txt"
    sample_list = pd.read_csv(sample_list_file, sep=" ", header=None)
    sample_list.columns = ["library", "sample_id", "cohort"]
    sample_list.head()
    # get the sample_ids where cohort is Cohort1
    cohort1 = sample_list[sample_list["cohort"] == "Cohort1"]["sample_id"]
    cohort2 = sample_list[sample_list["cohort"] == "Cohort2"]["sample_id"]
    PON = sample_list[sample_list["cohort"] == "PanelOfNormals"]["sample_id"]

    for j in range(N_ITR):
        for i in range(IDX, IDX_END, 1):
            print(i)
            if i == 25:
                continue
            X, y = get_X_y(filelist[i], root=DIRECTORY, cohort=cohort1)
            FEATURENAME = filelist[i].split(".csv.gz")[0]

            pos_l = []
            y_l = []
            train_time_l = []
            test_time_l = []

            kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=23)
            for train_index, test_index in kf.split(X, y):
                X_train, X_test, y_train, y_test = (
                    X.iloc[train_index],
                    X.iloc[test_index],
                    y.iloc[train_index],
                    y.iloc[test_index],
                )
                y_l.append(y_test)

                # Shuffle the training sets
                p = default_rng(seed=SEED).permutation(len(X_train))
                X_t = X_train.iloc[p]
                y_t = y_train.iloc[p]

                # p = default_rng(seed=SEED).permutation(len(X_train.columns.tolist()))
                # X_t = X_train.iloc[:, p]
                # y_t = y_train

                model = TabPFNClassifier(n_jobs=N_JOBS, ignore_pretraining_limits=True)

                # Train the model
                start_time = time.perf_counter()
                model.fit(X_t, y_t)
                end_time = time.perf_counter()
                train_time_l.append(end_time - start_time)

                # Test the model
                start_time = time.perf_counter()
                pos_l.append(model.predict_proba(X_test))
                end_time = time.perf_counter()
                test_time_l.append(end_time - start_time)

                with open(
                    DIRECTORY
                    + "results/tab_pos_"
                    + FEATURENAME
                    + "_"
                    + str(j+1)
                    + ".pkl",
                    "wb",
                ) as f:
                    pickle.dump(pos_l, f)

                with open(
                    DIRECTORY
                    + "results/tab_train_"
                    + FEATURENAME
                    + "_"
                    + str(j+1)
                    + ".pkl",
                    "wb",
                ) as f:
                    pickle.dump(train_time_l, f)

                with open(
                    DIRECTORY
                    + "results/tab_test_"
                    + FEATURENAME
                    + "_"
                    + str(j+1)
                    + ".pkl",
                    "wb",
                ) as f:
                    pickle.dump(test_time_l, f)

                with open(
                    DIRECTORY + "results/tab_y_" + FEATURENAME + "_" + str(j+1) + ".pkl",
                    "wb",
                ) as f:
                    pickle.dump(y_l, f)
