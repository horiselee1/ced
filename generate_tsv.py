'''
Author: horiselee1 horiselee@gmail.com
Date: 2026-06-03 10:31:20
LastEditors: horiselee1 horiselee@gmail.com
LastEditTime: 2026-06-03 10:31:27
FilePath: \python72\ced-train\generate_tsv.py
Description: 

Copyright (c) 2026 by ${git_name_email}, All Rights Reserved. 
'''
import os
import pandas as pd

DATA_DIR = "./data"
# 明确类别到索引的映射
label_map = {"Bird": 0, "Construction": 1, "Thunder": 2, "zOther": 3}

records = []
for folder_name, label_idx in label_map.items():
    folder_path = os.path.join(DATA_DIR, folder_name)
    if not os.path.exists(folder_path):
        continue
    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            # 保存相对路径
            rel_path = os.path.join(DATA_DIR, folder_name, file)
            records.append({"filename": rel_path, "labels": str(label_idx)})

df = pd.DataFrame(records)
# 导出为标准的无标签头/全路径初始tsv（这里以 \t 隔开）
df.to_csv("my_raw_data.tsv", sep="\t", index=False)
print(f"原始扫描成功，共生成 {len(df)} 条样本。")