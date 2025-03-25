import pandas as pd
import os
import numpy as np
import time
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from tabpfn import TabPFNClassifier
import warnings

warnings.filterwarnings("ignore")


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


# Load the sample list
sample_list_file = "data/AllSamples.MIGHT.Passed.samples.txt"
sample_list = pd.read_csv(sample_list_file, sep=" ", header=None, names=["library", "sample_id", "cohort"])

# Filter the samples based on the cohort
cohort1 = sample_list.loc[sample_list["cohort"] == "Cohort1", "sample_id"]
cohort2 = sample_list[sample_list["cohort"] == "Cohort2"]["sample_id"]

# Load the data
X, y = get_X_y("WiseCondorX.Wise-1.csv", root="./data/", cohort=cohort2, verbose=True)

print(y.value_counts())

# Set the number of splits and runs
N_SPLITS = 5
N_RUNS = 10
SEED_LIST = np.random.randint(0, 10000, N_RUNS)  # Generate 10 random seeds
# Save the results
all_auc, all_S98, all_acc, all_train_times = [], [], [], []

for run, seed in enumerate(SEED_LIST, 1):
    print(f" Running Experiment {run}/{N_RUNS} with Random Seed: {seed}")

    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

    auc_scores, S98_scores, acc_scores, train_times = [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):
        print(f"  Training Fold {fold}/{N_SPLITS}...")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Initialize the TabPFN Classifier
        clf = TabPFNClassifier(device="cuda:0" if torch.cuda.is_available() else "cpu",
                               n_jobs=40, ignore_pretraining_limits=True)

        # Training time
        start_time = time.perf_counter()
        clf.fit(X_train, y_train)
        end_time = time.perf_counter()

        train_time = end_time - start_time
        train_times.append(train_time)

        # Pridiction
        y_pred = clf.predict(X_test)
        y_pred_prob = clf.predict_proba(X_test)[:, 1]

        # AUC score
        auc_score = roc_auc_score(y_test, y_pred_prob)
        auc_scores.append(auc_score)

        # Accuracy score
        acc = accuracy_score(y_test, y_pred)
        acc_scores.append(acc)

        # Sensitivity at 98% specificity
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        mask = fpr <= 0.02
        S98 = np.max(tpr[mask]) if np.any(mask) else 0
        S98_scores.append(S98)

        print(f"  Fold {fold} - Accuracy: {acc:.4f}, AUC: {auc_score:.4f}, S@98: {S98:.4f}, Train Time: {train_time:.2f} sec")
        print("  " + "=" * 50)

    # Save the results
    all_auc.append(np.mean(auc_scores))
    all_S98.append(np.mean(S98_scores))
    all_acc.append(np.mean(acc_scores))
    all_train_times.append(np.mean(train_times))

    print(f"\n Experiment {run} Summary:")
    print(f"   Avg Accuracy: {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}")
    print(f"   Avg AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
    print(f"   Avg S@98: {np.mean(S98_scores):.4f} ± {np.std(S98_scores):.4f}")
    print(f"   Avg Train Time: {np.mean(train_times):.2f} sec")
    print("=" * 70)

# Output the overall results
print("\n Overall Results Across 10 Runs ")
print(f" Overall Avg Accuracy: {np.mean(all_acc):.4f} ± {np.std(all_acc):.4f}")
print(f" Overall Avg AUC: {np.mean(all_auc):.4f} ± {np.std(all_auc):.4f}")
print(f" Overall Avg S@98: {np.mean(all_S98):.4f} ± {np.std(all_S98):.4f}")
print(f" Overall Avg Train Time: {np.mean(all_train_times):.2f} sec")
print("=" * 80)