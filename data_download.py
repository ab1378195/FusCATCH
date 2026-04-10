import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time

# 五只股票及其异常区间
stocks = {
    "TRBO": ("2018-03-30", "2021-04-09"),
    "APPB": ("2018-03-25", "2021-04-13"),
    "AEMD": ("2018-01-22", "2021-02-07"),
    "NBDR": ("2018-03-11", "2021-04-03"),
    "GME":  ("2019-01-11", "2022-01-29"),
}

def get_data_with_labels(ticker, anomaly_start, anomaly_end):
    anomaly_start = datetime.strptime(anomaly_start, "%Y-%m-%d")
    anomaly_end = datetime.strptime(anomaly_end, "%Y-%m-%d")

    # # 论文规则
    # start_date = anomaly_start - timedelta(days=730)  # 24个月
    # end_date = anomaly_end + timedelta(days=365)      # 12个月
    start_date = anomaly_start
    end_date = anomaly_end

    print(f"Downloading {ticker} from {start_date.date()} to {end_date.date()}")

    # yf.set_config(proxy="http://127.0.0.1:7890")
    yf.config.network.proxy = {
        "http": "http://127.0.0.1:7890",
        "https": "http://127.0.0.1:7890"
    }
    # 下载数据
    df = yf.download(
        ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        progress=False,
        threads=False,
    )

    if df.empty:
        print(f"Warning: No data for {ticker}")
        return None

    # 重置索引
    df.reset_index(inplace=True)

    # 添加标签
    df["anomaly"] = df["Date"].apply(
        lambda x: 1 if anomaly_start <= x <= anomaly_end else 0
    )

    # 添加ticker列
    df["ticker"] = ticker

    return df


# 主流程
all_data = []

for ticker, (start, end) in stocks.items():
    df = get_data_with_labels(ticker, start, end)
    if df is not None:
        all_data.append(df)
        df.to_csv(f"{ticker}_data.csv", index=False)
    time.sleep(5)

# 合并数据（可选）
# if all_data:
#     combined = pd.concat(all_data, ignore_index=True)
#     combined.to_csv("combined_stock_data.csv", index=False)

print("Done!")