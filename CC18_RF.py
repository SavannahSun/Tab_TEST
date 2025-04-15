import os
import pandas as pd
import numpy as np
import time
import gc
import warnings

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
random_state = 407  # 407 42 57 425 507 20020425 20020507 1234 123 2025

# ==== 函数：读取数据 ====
def get_X_y(file_path, shuffle_data=True):
    df = pd.read_csv(file_path)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # 对 X 中的 object 类型列做 Label Encoding
    for col in X.columns:
        if X[col].dtype == "object":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    # Label encode y
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    if shuffle_data:
        X, y = shuffle(X, y, random_state=random_state)

    num_classes = len(np.unique(y))
    print(f"📊 数据集 {file_path} 共有 {num_classes} 个类别")

    if num_classes > 10:
        print(f"🚨 跳过类别数超过 10 的数据集: {file_path}")
        return None, None, num_classes

    return X, y, num_classes
# ==== 参数 ====
DATA_DIR = "./data_OpenML_CC18"
RESULTS_CSV = "./result/rf_openml_results_0.csv"
SKIPPED_CSV = "./result/rf_openml_skipped_0.csv"
N_SPLITS = 10

completed_datasets = set()
if os.path.exists(RESULTS_CSV):
    completed_df = pd.read_csv(RESULTS_CSV)
    completed_datasets.update(completed_df["Dataset"].tolist())

dataset_files = sorted(os.listdir(DATA_DIR), key=lambda x: os.path.getctime(os.path.join(DATA_DIR, x)))

for dataset_file in dataset_files:
    if not dataset_file.endswith(".csv") or dataset_file in completed_datasets:
        print(f"🔄 跳过已完成的数据集: {dataset_file}")
        continue

    file_path = os.path.join(DATA_DIR, dataset_file)
    X, y, num_classes = get_X_y(file_path, shuffle_data=True)

    if X is None or y is None:
        skipped_df = pd.DataFrame([[dataset_file, num_classes]], columns=["Dataset", "Num Classes"])
        skipped_df.to_csv(SKIPPED_CSV, mode='a', header=not os.path.exists(SKIPPED_CSV), index=False)
        continue

    print(f"✔ 处理数据集: {dataset_file}, X shape: {X.shape}, y shape: {y.shape}")

    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=random_state)
    auc_scores, acc_scores, train_times = [], [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):
        print(f"🌲 RF Training Fold {fold}/{N_SPLITS} on {dataset_file}...")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = RandomForestClassifier(
            n_estimators=1000,
            max_depth=None,
            random_state=random_state,
            n_jobs=-1
        )

        start_time = time.perf_counter()
        clf.fit(X_train, y_train)
        end_time = time.perf_counter()

        y_pred = clf.predict(X_test)
        y_pred_prob = clf.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        acc_scores.append(acc)

        if num_classes == 2:
            auc_score = roc_auc_score(y_test, y_pred_prob[:, 1])
        else:
            auc_score = roc_auc_score(y_test, y_pred_prob, multi_class="ovr")
        auc_scores.append(auc_score)
        train_times.append(end_time - start_time)

        print(f"✅ Fold {fold} - Accuracy: {acc:.4f}, AUC: {auc_score:.4f}")

    avg_acc = np.mean(acc_scores)
    std_acc = np.std(acc_scores)
    avg_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)
    avg_train_time = np.mean(train_times)

    print(f"📊 {dataset_file} - Avg Accuracy: {avg_acc:.4f} ± {std_acc:.4f}, Avg AUC: {avg_auc:.4f} ± {std_auc:.4f}, Avg Train Time: {avg_train_time:.2f} sec")

    results_df = pd.DataFrame([[dataset_file, X.shape[0], avg_acc, std_acc, avg_auc, std_auc, avg_train_time]],
                              columns=["Dataset", "Samples", "Avg Accuracy", "Std Accuracy", "Avg AUC", "Std AUC", "Avg Train Time"])
    
    results_df.to_csv(RESULTS_CSV, mode='a', header=not os.path.exists(RESULTS_CSV), index=False)

    del X, y
    gc.collect()

print(f"✅ 所有数据处理完毕，结果保存在 {RESULTS_CSV}")