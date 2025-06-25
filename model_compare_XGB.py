import os
import sys
import pandas as pd
import numpy as np
import time
import gc
import warnings

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '3,4'

# ✅ 日志重定向
class Logger:
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "a", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger("xgb_single_run_log.txt")  # 日志写入文件

# ==== 参数设置 ====
SEED = 42
N_SPLITS = 10
DATA_DIR = "./data_OpenML_CC18"
RESULTS_CSV = "./xgb_comparison_single_run.csv"
os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)

# ==== 加载数据 ====
def get_X_y(file_path, seed=SEED):
    df = pd.read_csv(file_path)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    for col in X.columns:
        if X[col].dtype == "object":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    X = X.fillna(X.mean(numeric_only=True))  # 处理 NaN
    y = LabelEncoder().fit_transform(y)
    X, y = shuffle(X, y, random_state=seed)
    return X, y

# ==== 获取文件顺序 ====
def get_dataset_index(filename):
    try:
        return int(filename.split("_")[1].split(".")[0])
    except:
        return float("inf")

dataset_files = sorted(
    [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")],
    key=get_dataset_index
)

# ==== 主循环 ====
for dataset_file in dataset_files:
    file_path = os.path.join(DATA_DIR, dataset_file)
    X, y = get_X_y(file_path, seed=SEED)
    num_classes = len(np.unique(y))

    print(f"\n📂 Dataset: {dataset_file}")
    print(f"    🔸 Samples: {X.shape[0]}, Features: {X.shape[1]}, Classes: {num_classes}")

    acc_list, f1_list = [], []
    start_time = time.perf_counter()

    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):
        print(f"🚀 Fold {fold}/{N_SPLITS}...")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = XGBClassifier(
            tree_method="gpu_hist",
            predictor="gpu_predictor",
            objective="multi:softprob" if num_classes > 2 else "binary:logistic",
            num_class=num_classes if num_classes > 2 else None,
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.1,
            verbosity=0,
            random_state=SEED,
            n_jobs=-1,
            eval_metric='mlogloss' if num_classes > 2 else 'logloss'
        )

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        acc_list.append(acc)
        f1_list.append(f1)
        print(f"    ✅ Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    avg_acc = np.mean(acc_list)
    avg_f1 = np.mean(f1_list)

    print(f"📊 Done {dataset_file}: Accuracy = {avg_acc:.4f}, Macro F1 = {avg_f1:.4f}, Time = {elapsed_time:.2f}s")

    # ✅ 即时写入 CSV 文件
    row = {
        "Dataset": dataset_file,
        "Samples": X.shape[0],
        "Features": X.shape[1],
        "NumClasses": num_classes,
        "XGB_Accuracy": avg_acc,
        "XGB_MacroF1": avg_f1,
        "XGB_Time": round(elapsed_time, 2)
    }

    df_row = pd.DataFrame([row])
    df_row.to_csv(RESULTS_CSV, mode='a', header=not os.path.exists(RESULTS_CSV), index=False)

    del X, y, clf
    gc.collect()

print(f"\n✅ All done! Results saved to {RESULTS_CSV}")