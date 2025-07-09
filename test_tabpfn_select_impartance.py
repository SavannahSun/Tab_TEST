import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import torch
from tabpfn import TabPFNClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")
import time

# Load data
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


sample_list_file = "data/AllSamples.MIGHT.Passed.samples.txt"
sample_list = pd.read_csv(sample_list_file, sep=" ", header=None)
sample_list.columns = ["library", "sample_id", "cohort"]
sample_list.head()
# get the sample_ids where cohort is Cohort1
cohort1 = sample_list[sample_list["cohort"] == "Cohort1"]["sample_id"]
cohort2 = sample_list[sample_list["cohort"] == "Cohort2"]["sample_id"]
PON = sample_list[sample_list["cohort"] == "PanelOfNormals"]["sample_id"]

X, y = get_X_y("WiseCondorX.Wise-1.csv", root="./data/", cohort=cohort1, verbose=True)
# 确保数据格式正确
print(y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#--------------------------------------------------------------
# 使用 RF 得到特征重要性排序
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

importances = rf.feature_importances_
feature_ranking = X_train.columns[np.argsort(importances)[::-1]]

feature_counts = [10, 20, 30, 40, 50, 100, 150,200, 250, 300, 400, 500, 800, 1000, 1500, 2000, X_train.shape[1]]

for k in feature_counts:
    selected_features = feature_ranking[:k]
    X_tr_k = X_train[selected_features]
    X_te_k = X_test[selected_features]

    print(f"\n Evaluating with Top-{k} features...")

    start_time = time.time()
    clf = TabPFNClassifier(device="cuda" if torch.cuda.is_available() else "cpu", n_jobs=8, ignore_pretraining_limits=True)
    clf.fit(X_tr_k, y_train)

    y_pred = clf.predict(X_te_k)
    y_pred_prob = clf.predict_proba(X_te_k)[:, 1]
    end_time = time.time()

    total_time = end_time - start_time

    # Metrics
    auc_score = roc_auc_score(y_test, y_pred_prob)
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    accuracy = accuracy_score(y_test, y_pred)

    # S@98
    mask = fpr <= 0.02
    S98 = np.max(tpr[mask]) if np.any(mask) else 0

    # Print
    print(f"Top-{k} Features")
    print(f"Inference Time: {total_time:.2f} seconds")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {auc_score:.4f}")
    print(f"S@98: {S98:.4f}")
    print("=" * 50)