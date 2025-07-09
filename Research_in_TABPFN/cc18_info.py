import openml
import pandas as pd

# 获取 OpenML-CC18 基准套件
suite = openml.study.get_suite("OpenML-CC18")  # 相当于 study id = 99  [oai_citation:0‡openml.github.io](https://openml.github.io/openml-python/main/examples/20_basic/simple_suites_tutorial.html?utm_source=chatgpt.com)

# 准备收集元信息的列表
records = []

for dataset_id in sorted(suite.data):
    ds = openml.datasets.get_dataset(dataset_id, download_data=False)
    meta = ds.qualities  # 含有 NumberOfClasses、NumberOfFeatures、NumberOfInstances 等
    records.append({
        "id": dataset_id,
        "name": ds.name,
        "class_num": int(meta["NumberOfClasses"]),
        "feature_num": int(meta["NumberOfFeatures"]),
        "sample_num": int(meta["NumberOfInstances"]),
        # 可根据需要添加更多字段（参照 ds.qualities 输出）
        "missing_values": meta.get("NumberOfMissingValues", "")
    })

# 构建 DataFrame 并排序导出 CSV
df = pd.DataFrame(records).sort_values("id")
df.to_csv("openml_cc18_datasets.csv", index=False)
print("done")