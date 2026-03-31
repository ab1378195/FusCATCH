import pandas as pd

df = pd.read_parquet(r"data\FiFAR\testbed\train_alert\shuffle_1#team_2\train.parquet")
print(df.head())
print(df.columns)