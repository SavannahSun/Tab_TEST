import os
import pandas as pd
import numpy as np
import time
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from tabpfn import TabPFNClassifier
from sklearn.utils import shuffle
import gc
import warnings
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
random_state = 407 #42 57 425 507 20020425 20020507 1234 123 2025


# **定义 get_X_y 函数**
def get_X_y(file_path, shuffle_data=True):
    df = pd.read_csv(file_path)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # 使用 LabelEncoder 转换 y
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    if shuffle_data:
        X, y = shuffle(X, y, random_state=random_state)

    num_classes = len(np.unique(y))
    print(f"📊 数据集 {file_path} 共有 {num_classes} 个类别")
    
    if num_classes > 10:
        print(f"🚨 数据集 {file_path} 共有 {num_classes} 个类别，超过 10，跳过！")
        return None, None, num_classes

    return X, y, num_classes


# **参数**
DATA_DIR = "./data_OpenML_CC18"  # 数据集目录
RESULTS_CSV = "./result/tabpfn_openml_results_9.csv"  # 结果保存文件
SKIPPED_CSV = "./result/tabpfn_openml_skipped_9.csv"  # 被跳过的数据集
N_SPLITS = 10  # 10-Fold 交叉验证

completed_datasets = set()
if os.path.exists(RESULTS_CSV):
    completed_df = pd.read_csv(RESULTS_CSV)
    completed_datasets.update(completed_df["Dataset"].tolist())


# 获取所有数据集**
dataset_files = sorted(os.listdir(DATA_DIR), key=lambda x: os.path.getctime(os.path.join(DATA_DIR, x)))

for dataset_file in dataset_files:
    
    if not dataset_file.endswith(".csv") or dataset_file in completed_datasets:
        print(f"🔄 跳过已完成的数据集: {dataset_file}")
        continue
    
    file_path = os.path.join(DATA_DIR, dataset_file)

    # **加载数据**
    X, y, num_classes = get_X_y(file_path, shuffle_data=True)
    if X is None or y is None:
        # **记录跳过的数据集**
        skipped_df = pd.DataFrame([[dataset_file, num_classes]], columns=["Dataset", "Num Classes"])
        skipped_df.to_csv(SKIPPED_CSV, mode='a', header=not os.path.exists(SKIPPED_CSV), index=False)
        continue

    print(f"✔ 处理数据集: {dataset_file}, X shape: {X.shape}, y shape: {y.shape}")


    # **10-Fold 交叉验证**
    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=random_state)
    auc_scores, acc_scores, train_times = [], [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):
        print(f"🚀 Training Fold {fold}/{N_SPLITS} on {dataset_file}...")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # **初始化 TabPFN**
        clf = TabPFNClassifier(device="cuda:0" if torch.cuda.is_available() else "cpu",
                               n_jobs=40, ignore_pretraining_limits=True)

        # **记录训练时间**
        start_time = time.perf_counter()
        clf.fit(X_train, y_train)
        end_time = time.perf_counter()

        # **预测**
        y_pred = clf.predict(X_test)
        y_pred_prob = clf.predict_proba(X_test)

        # **计算 Accuracy**
        acc = accuracy_score(y_test, y_pred)  # 计算 Accuracy
        acc_scores.append(acc)

        # **计算 AUC**
        if num_classes == 2:
            auc_score = roc_auc_score(y_test, y_pred_prob[:, 1])  # 二分类 AUC
        else:
            auc_score = roc_auc_score(y_test, y_pred_prob, multi_class="ovr")  # 多分类 AUC
        
        auc_scores.append(auc_score)
        train_times.append(end_time - start_time)

        print(f"✅ Fold {fold} - Accuracy: {acc:.4f}, AUC: {auc_score:.4f}")

        # **释放 GPU 内存**
        del clf
        gc.collect()
        torch.cuda.empty_cache()

    # **计算平均值**
    avg_acc = np.mean(acc_scores)
    std_acc = np.std(acc_scores)
    avg_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)
    avg_train_time = np.mean(train_times)

    print(f"📊 {dataset_file} - Avg Accuracy: {avg_acc:.4f} ± {std_acc:.4f}, Avg AUC: {avg_auc:.4f} ± {std_auc:.4f}, Avg Train Time: {avg_train_time:.2f} sec")

    # **存入结果到 CSV**
    results_df = pd.DataFrame([[dataset_file, X.shape[0], avg_acc, std_acc, avg_auc, std_auc, avg_train_time]],
                              columns=["Dataset", "Samples", "Avg Accuracy", "Std Accuracy", "Avg AUC", "Std AUC", "Avg Train Time"])
    
    results_df.to_csv(RESULTS_CSV, mode='a', header=not os.path.exists(RESULTS_CSV), index=False)

    # **释放 CPU 内存**
    del X, y
    gc.collect()

print(f"📁 所有数据处理完毕，结果实时保存至 {RESULTS_CSV}")