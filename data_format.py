import pandas as pd

# ====== 1. 读取数据 ======
input_file = "data\\Credit.csv"   # 修改为你的文件路径
df = pd.read_csv(input_file)


# ====== 5. 构造时间列 ======
# 使用 step 作为时间（如果你想用 index 也可以）
time = df["Time"]
df = df.drop(columns=["Time"])


df = df.drop(columns=["V8","V13","V15","V20","V21","V22","V23","V24","V25","V26","V27","V28"])
print(df.columns)
# ====== 6. 转换为 CATCH 格式 ======
rows = []

# 遍历每个变量（列）
from tqdm import tqdm
for col in tqdm(df.columns):
    values = df[col].values
    
    for t, v in zip(time, values):
        rows.append([t, v, col])

# # ====== 7. 添加 label ======
# for t, y in zip(time, label):
#     rows.append([t, y, "label"])

# ====== 8. 转为 DataFrame ======
catch_df = pd.DataFrame(rows, columns=["date", "data", "cols"])

# ====== 9. 保存 ======
output_file = "Credit.csv"
catch_df.to_csv(output_file, index=False)

print("转换完成，输出文件：", output_file)