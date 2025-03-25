import pandas as pd
import numpy as np
import time
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.utils import shuffle
from tabpfn import TabPFNClassifier
import warnings

warnings.filterwarnings("ignore")


def get_X_y(file_path, shuffle_data=True):
    df = pd.read_csv(file_path)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # 使用 LabelEncoder 转换 y
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    if shuffle_data:
        X, y = shuffle(X, y, random_state=42)

    num_classes = len(np.unique(y))
    print(f"📊 数据集 {file_path} 共有 {num_classes} 个类别")
    
    if num_classes > 10:
        print(f"🚨 数据集 {file_path} 共有 {num_classes} 个类别，超过 10，跳过！")
        return None, None, num_classes

    return X, y, num_classes

# **加载数据**
file_path = "./data_OpenML_CC18/dataset_1486.csv"
X, y, num_classes = get_X_y(file_path, shuffle_data=True)

# **确保数据格式正确**
print(f"✔ 数据加载成功: {file_path}")
print(f"🔹 X shape: {X.shape}, y shape: {y.shape}")
print(f"🎯 标签分布:\n{pd.Series(y).value_counts()}")

# **10-fold 交叉验证**
N_SPLITS = 10
kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

auc_scores,acc_scores, train_times = [], [], []

for fold, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):
    print(f"🚀 Training Fold {fold}/{N_SPLITS}...")

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # **初始化 TabPFN**
    clf = TabPFNClassifier(device="cuda" if torch.cuda.is_available() else "cpu",
                           n_jobs=8, ignore_pretraining_limits=True)

    # **记录训练时间**
    start_time = time.perf_counter()
    clf.fit(X_train, y_train)
    end_time = time.perf_counter()

    train_time = end_time - start_time
    train_times.append(train_time)

    # **预测**
    y_pred = clf.predict(X_test)
    y_pred_prob = clf.predict_proba(X_test)

    # **计算 AUC**
    if num_classes == 2:
            auc_score = roc_auc_score(y_test, y_pred_prob[:, 1])  # 二分类 AUC
    else:
        auc_score = roc_auc_score(y_test, y_pred_prob, multi_class="ovr")  # 多分类 AUC
       

    # **计算 Accuracy**
    acc = accuracy_score(y_test, y_pred)
    acc_scores.append(acc)


    print(f"✅ Fold {fold} - Accuracy: {acc:.4f}, AUC: {auc_score:.4f}, Train Time: {train_time:.2f} sec")
    print("=" * 50)

# **最终结果**
print(f"\n📊 **最终结果:**")
print(f"Avg Accuracy: {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}")
print(f"Avg AUC: {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")
print(f"Avg Train Time: {np.mean(train_times):.2f} sec")