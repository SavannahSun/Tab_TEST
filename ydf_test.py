import warnings
import ydf
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import os

# **忽略 google.protobuf 相关的警告**
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")

DATA_DIR = "./data_OpenML_CC18"

def get_X_y(file_path, shuffle_data=True):
    df = pd.read_csv(file_path)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # **确保 y 是整数**
    label_name = y.name if isinstance(y, pd.Series) else "target"
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)  # 转换成整数
    y = pd.Series(y).fillna(0).astype(int)  # 填充 NaN 并转换为整数

    if shuffle_data:
        X, y = shuffle(X, y, random_state=42)

    num_classes = len(np.unique(y))
    print(f" 数据集 {file_path} 共有 {num_classes} 个类别")

    return X, y, num_classes, label_name

file_path = "./data_OpenML_CC18/dataset_1486.csv"
X, y, num_classes, label_name = get_X_y(file_path, shuffle_data=True)

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
auc_scores = []
acc_scores = []

for train_index, test_index in kf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # **确保 y_train 是整数**
    y_train_series = pd.Series(y_train, name=label_name).fillna(0).astype(int)  
    train_df = pd.concat([X_train, y_train_series], axis=1)  # 组合数据

    # **使用正确的 Task 类型**
    learner = ydf.RandomForestLearner(
        label=label_name, 
        task=ydf.Task.CLASSIFICATION,  
        num_trees=100, 
        num_threads=min(128, os.cpu_count())  
    )
    model = learner.train(train_df)  # 训练模型
    
    # Predict probabilities and labels
    y_prob = model.predict_proba(X_test)  
    y_pred = model.predict(X_test)
    
    # Compute metrics
    auc = roc_auc_score(y_test, y_prob, multi_class="ovr")  
    acc = accuracy_score(y_test, y_pred)
    
    auc_scores.append(auc)
    acc_scores.append(acc)

# Compute mean and std of scores
auc_mean, auc_std = np.mean(auc_scores), np.std(auc_scores)
acc_mean, acc_std = np.mean(acc_scores), np.std(acc_scores)

print(f"AUC: {auc_mean:.4f} ± {auc_std:.4f}")
print(f"Accuracy: {acc_mean:.4f} ± {acc_std:.4f}")