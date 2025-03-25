import os
import openml
import pandas as pd

# 设置存储目录
base_dir = "./data_OpenML_CTR23"
classification_dir = os.path.join(base_dir, "classification")
regression_dir = os.path.join(base_dir, "regression")

os.makedirs(classification_dir, exist_ok=True)
os.makedirs(regression_dir, exist_ok=True)

benchmark_suite = openml.study.get_suite(353)  # OpenML-CTR23
dataset_ids = benchmark_suite.data

for dataset_id in dataset_ids:
    try:
        dataset = openml.datasets.get_dataset(dataset_id)
        target_column = dataset.default_target_attribute
        df, target_series, _, _ = dataset.get_data(target=target_column)

        if target_series is None or len(target_series) != len(df):
            print(f"⚠️ {dataset_id} skipped: target missing or length mismatch")
            continue

        df[target_column] = target_series.reset_index(drop=True)
        y = df[target_column]

        # 分类 vs 回归判断
        if y.dtype == "object" or pd.api.types.is_bool_dtype(y):
            task_type = "classification"
        elif pd.api.types.is_numeric_dtype(y) and y.nunique() > 10:
            task_type = "regression"
        else:
            task_type = "classification"

        subdir = classification_dir if task_type == "classification" else regression_dir
        file_path = os.path.join(subdir, f"dataset_{dataset_id}.csv")
        df.to_csv(file_path, index=False)

        print(f"✅ Saved dataset {dataset_id} ({task_type}) to {file_path}")
    except Exception as e:
        print(f"❌ Failed to process dataset {dataset_id}: {e}")
        
print("✅ All datasets processed!")