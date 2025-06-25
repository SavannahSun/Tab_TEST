import os
import openml
import pandas as pd

# **1️⃣ 设定存储目录**
save_dir = "./data_OpenML_CC18"
# save_dir = "./data_OpenML_CTR23"
os.makedirs(save_dir, exist_ok=True)  # 确保文件夹存在

# **2️⃣ 获取 OpenML-CC18 数据集列表**
benchmark_suite = openml.study.get_suite(99)  # OpenML-CC18
# benchmark_suite = openml.study.get_suite(353)  # OpenML-CTR23
dataset_ids = benchmark_suite.data

print(f"🔍 发现 {len(dataset_ids)} 个 OpenML-CC18 数据集，开始下载...")
# print(f"🔍 发现 {len(dataset_ids)} 个 OpenML-CTR23 数据集，开始下载...")

# **3️⃣ 遍历所有数据集**
for dataset_id in dataset_ids:
    try:
        # **3.1 获取数据集**
        dataset = openml.datasets.get_dataset(dataset_id)
        target_column = dataset.default_target_attribute  # 获取 label 列名
        file_path = os.path.join(save_dir, f"dataset_{dataset_id}.csv")

        # **3.2 检查是否已下载**
        if os.path.exists(file_path):
            print(f"✔ 数据集 {dataset_id} 已存在，跳过下载")
            continue

        print(f"⏳ 下载数据集 {dataset_id} ({dataset.name})...")

        # **3.3 获取数据**
        df, target_series, _, _ = dataset.get_data(target=target_column)

        # **3.4 处理 target**
        if target_series is None or len(target_series) != len(df):
            print(f"⚠ 数据集 {dataset_id} ({dataset.name}) 跳过: target 为空或长度不匹配")
            continue

        # **3.5 确保 target 变量是 Series 并与 DataFrame 匹配**
        df[target_column] = target_series.reset_index(drop=True)
        columns = [col for col in df.columns if col != target_column] + [target_column]
        df = df[columns]

        # **3.6 保存到 CSV**
        df.to_csv(file_path, index=False)
        print(f"✔ 数据集 {dataset_id} ({dataset.name}) 已保存到 {file_path}")

    except Exception as e:
        print(f"❌ 数据集 {dataset_id} 处理失败: {e}")

print("✅ 所有数据集处理完成！")
