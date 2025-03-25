import os
import pandas as pd
import numpy as np
import time
import torch
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tabpfn import TabPFNRegressor
from sklearn.utils import shuffle
import gc
import warnings

warnings.filterwarnings("ignore")

# 参数设定
DATA_DIR = "./data_OpenML_CTR23/regression"  # 存放回归数据集的路径
RESULTS_CSV = "tabpfn_ctr23_regression_results_Shuffle.csv"
SKIPPED_CSV = "tabpfn_ctr23_regression_skipped_Shuffle.csv"
N_SPLITS = 10

# 加载已完成数据集
completed_datasets = set()
if os.path.exists(RESULTS_CSV):
    completed_df = pd.read_csv(RESULTS_CSV)
    completed_datasets.update(completed_df["Dataset"].tolist())

def get_X_y(file_path, shuffle_data=True):
    df = pd.read_csv(file_path)

    # 提取特征和目标
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # 检查 y 是否为数值型（回归任务要求）
    if not np.issubdtype(y.dtype, np.number):
        print(f"⚠️ 非数值型目标，跳过: {file_path}")
        return None, None

    # Shuffle 数据，同时保持 X 和 y 的对应关系
    if shuffle_data:
        df_shuffled = shuffle(df, random_state=42)
        X = df_shuffled.iloc[:, :-1]
        y = df_shuffled.iloc[:, -1]

    return X.reset_index(drop=True), y.reset_index(drop=True)

# 遍历数据集
dataset_files = sorted(os.listdir(DATA_DIR), key=lambda x: os.path.getctime(os.path.join(DATA_DIR, x)))

for dataset_file in dataset_files:
    if not dataset_file.endswith(".csv") or dataset_file in completed_datasets:
        print(f"🔄 跳过已完成或无效文件: {dataset_file}")
        continue

    file_path = os.path.join(DATA_DIR, dataset_file)
    X, y = get_X_y(file_path, shuffle_data=True)
    if X is None or y is None:
        pd.DataFrame([[dataset_file, "load_error"]], columns=["Dataset", "Reason"]).to_csv(SKIPPED_CSV, mode="a", header=not os.path.exists(SKIPPED_CSV), index=False)
        continue

    print(f"✔ 正在处理: {dataset_file} - X: {X.shape}, y: {y.shape}")

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    rmse_list, mae_list, r2_list, train_times = [], [], [], []

    try:
        for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
            print(f"🚀 Fold {fold}/{N_SPLITS} on {dataset_file}...")

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = TabPFNRegressor(device="cuda:0" if torch.cuda.is_available() else "cpu",
                                    n_jobs=20, ignore_pretraining_limits=True)

            start_time = time.perf_counter()
            model.fit(X_train, y_train)
            end_time = time.perf_counter()
            train_times.append(end_time - start_time)

            y_pred = model.predict(X_test)

            rmse_list.append(np.sqrt(mean_squared_error(y_test, y_pred)))
            mae_list.append(mean_absolute_error(y_test, y_pred))
            r2_list.append(r2_score(y_test, y_pred))

            print(f"✅ RMSE: {rmse_list[-1]:.4f}, MAE: {mae_list[-1]:.4f}, R²: {r2_list[-1]:.4f}, Time: {train_times[-1]:.2f}s")

            del model
            gc.collect()
            torch.cuda.empty_cache()

        # 汇总结果
        result = pd.DataFrame([[
            dataset_file, X.shape[0],
            np.mean(rmse_list), np.std(rmse_list),
            np.mean(mae_list), np.std(mae_list),
            np.mean(r2_list), np.std(r2_list),
            np.mean(train_times)
        ]], columns=[
            "Dataset", "Samples",
            "Avg RMSE", "Std RMSE",
            "Avg MAE", "Std MAE",
            "Avg R2", "Std R2",
            "Avg Train Time"
        ])
        result.to_csv(RESULTS_CSV, mode="a", header=not os.path.exists(RESULTS_CSV), index=False)

    except Exception as e:
        print(f"❌ 训练失败: {e}")
        pd.DataFrame([[dataset_file, f"fold_error: {str(e)}"]], columns=["Dataset", "Reason"]).to_csv(SKIPPED_CSV, mode="a", header=not os.path.exists(SKIPPED_CSV), index=False)

print("✅ 所有回归任务处理完毕！结果实时保存。")