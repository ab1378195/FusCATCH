import pandas as pd

# ====== 1. 读取数据 ======
input_file = "data\\raw\\TRBO.csv"   # 修改为你的文件路径
df = pd.read_csv(input_file)


# ====== 5. 构造时间列 ======
# 使用 step 作为时间（如果你想用 index 也可以）
time = df["Date"]
df = df.drop(columns=["Date"])

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
output_file = "TRBO.csv"
catch_df.to_csv(output_file, index=False)

print("转换完成，输出文件：", output_file)