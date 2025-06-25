import os
import re
import sys
import time
import torch
import gc
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils import shuffle

from tabpfn import TabPFNClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from treeple import ObliqueRandomForestClassifier

import warnings
warnings.filterwarnings("ignore")

# === 实时日志记录到 txt ===
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

sys.stdout = Logger("model_comparison_log.txt")

# === CUDA 设置 ===
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
available_cuda_ids = [int(x) for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
cuda_index = 0

# === 参数设置
n_estimators = 1000
SEED = 42
EARLY_STOPPING_ROUNDS = 20
RESULTS_CSV = "model_comparison_accuracy.csv"
DATA_DIR = "./data_OpenML_CC18"

def get_dataset_index(filename):
    match = re.search(r"dataset_(\d+)\.csv", filename)
    return int(match.group(1)) if match else float("inf")

def load_dataset(file_path, seed=SEED):
    df = pd.read_csv(file_path)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    for col in X.columns:
        if X[col].dtype == "object":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    # ✅ 统一补充缺失值
    X = X.fillna(X.mean(numeric_only=True))
    y = LabelEncoder().fit_transform(y)
    X, y = shuffle(X, y, random_state=seed)
    return X.astype(np.float32), y

def evaluate_model(clf, X, y, model_name, seed=SEED):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    acc_list, f1_list = [], []

    for i, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        try:
            if model_name == "CatBoost":
                clf.fit(
                    X_train, y_train,
                    eval_set=(X_test, y_test),
                    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                    verbose=False
                )
            else:
                clf.fit(X_train, y_train)  

            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')

            acc_list.append(acc)
            f1_list.append(f1)
            print(f"    🔹 Fold {i}: Accuracy = {acc:.4f}, Macro F1 = {f1:.4f}")

        except Exception as e:
            print(f"❌ Fold {i} failed for {model_name}: {e}")
            acc_list.append(np.nan)
            f1_list.append(np.nan)

    return np.nanmean(acc_list), np.nanmean(f1_list)

def get_model(name, params, assigned_cuda):
    if name == "TabPFN":
        return TabPFNClassifier(device=f"cuda:{assigned_cuda}", n_jobs=40, ignore_pretraining_limits=True)
    elif name == "CatBoost":
        return CatBoostClassifier(verbose=0, **params)
    elif name == "XGBoost":
        return XGBClassifier(
            tree_method='gpu_hist',
            predictor='gpu_predictor',
            gpu_id=assigned_cuda,
            use_label_encoder=False,
            verbosity=0,
            objective='multi:softprob' if params.get("multi_class", False) else 'binary:logistic',
            num_class=params.get("num_class", None),
            n_estimators=params.get("n_estimators", 1000),
            max_depth=params.get("max_depth", 6),
            learning_rate=params.get("learning_rate", 0.1),
            max_bin=512,
            enable_categorical=False,
            eval_metric='mlogloss' if params.get("multi_class", False) else 'logloss',
            n_jobs=-1
        )
    elif name == "LightGBM":
        return LGBMClassifier(
            device='gpu',
            boosting_type='gbdt',
            objective='multiclass' if num_classes > 2 else 'binary',
            num_class=num_classes if num_classes > 2 else 1,
            n_estimators=n_estimators,
            learning_rate=0.1,
            max_depth=6,
            random_state=SEED,
            n_jobs=40,
            verbose=-1
        )
    elif name == "RF":
        return RandomForestClassifier(**params)
    elif name == "SPORF":
        return ObliqueRandomForestClassifier(**params)
    else:
        raise ValueError(f"Unknown model name: {name}")

models = {
    "TabPFN": {},
    "CatBoost": {"iterations": 1000, "learning_rate": 0.1, "depth": 6},
    # "XGBoost": {"n_estimators": n_estimators, "max_depth": 6, "learning_rate": 0.1},
    # "LightGBM": {"n_estimators": n_estimators, "learning_rate": 0.1, "max_depth": 6},
    "RF": {"n_estimators": n_estimators, "n_jobs": 40, "random_state": SEED},
    "SPORF": {"n_estimators": n_estimators, "n_jobs": 40, "bootstrap": True, "max_features": 0.7},
}

completed = set()
if os.path.exists(RESULTS_CSV):
    try:
        df_prev = pd.read_csv(RESULTS_CSV)
        completed.update(df_prev["Dataset"].tolist())
    except Exception as e:
        print(f"⚠️ Failed to load {RESULTS_CSV}: {e}")

model_times = defaultdict(list)

dataset_files = sorted(os.listdir(DATA_DIR), key=get_dataset_index)
for file in dataset_files:
    if not file.endswith(".csv") or file in completed:
        print(f"⏭️ Skip the completed or invalid files: {file}")
        continue

    file_path = os.path.join(DATA_DIR, file)
    X, y = load_dataset(file_path, seed=SEED)
    num_classes = len(np.unique(y))

    print(f"\n📂 Dataset: {file}")
    print(f"    🔸 Samples: {X.shape[0]}, Features: {X.shape[1]}, Classes: {num_classes}")

    row = {
        "Dataset": file,
        "Samples": X.shape[0],
        "Features": X.shape[1],
        "NumClasses": num_classes
    }

    for model_name, model_params in models.items():
        if model_name == "TabPFN" and (X.shape[0] > 10000 or num_classes > 10):
            print(f"⏭️ Skip {model_name} on {file} due to TabPFN limitation")
            row[f"{model_name}_Accuracy"] = np.nan
            row[f"{model_name}_MacroF1"] = np.nan
            row[f"{model_name}_Time"] = np.nan
            continue

        print(f"🚀 Evaluating {model_name} on {file}...")
        assigned_cuda = available_cuda_ids[cuda_index % len(available_cuda_ids)]
        try:
            start = time.perf_counter()
            clf = get_model(model_name, model_params, assigned_cuda)
            acc, macro_f1 = evaluate_model(clf, X, y, model_name, seed=SEED)
            end = time.perf_counter()
            duration = end - start
            row[f"{model_name}_Accuracy"] = acc
            row[f"{model_name}_MacroF1"] = macro_f1
            row[f"{model_name}_Time"] = duration
            model_times[model_name].append(duration)
            print(f"✅ {model_name} on {file}: Accuracy = {acc:.4f}, Macro F1 = {macro_f1:.4f}, Time = {duration:.2f}s")
        except Exception as e:
            print(f"⚠️ {model_name} failed on {file}: {e}")
            row[f"{model_name}_Accuracy"] = np.nan
            row[f"{model_name}_MacroF1"] = np.nan
            row[f"{model_name}_Time"] = np.nan
        finally:
            del clf
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        cuda_index += 1

    pd.DataFrame([row]).to_csv(RESULTS_CSV, mode="a", index=False, header=not os.path.exists(RESULTS_CSV))
    print(f"📤 The result that has been written to {file}")

# === 平均训练时间统计 ===
print("\n📈 Average Training Time per Model:")
for model, times in model_times.items():
    avg_time = np.mean(times)
    print(f"⏱️ {model}: {avg_time:.2f} seconds over {len(times)} datasets")

print("\n🎉 All data processing has been completed! The result is saved in model_comparison_accuracy.csv")