import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

df = pd.read_csv('data/raw/Credit.csv')

# df.drop(columns=['Time'], inplace=True)
# df.drop(columns=['Amount'], inplace=True)
# fraud = df[df['Class']==1]
# normal = df[df['Class']==0]

# correlationFraud = fraud.loc[:, df.columns != 'Class'].corr()
# correlationNormal = normal.loc[:, df.columns != 'Class'].corr()

# mask = np.zeros_like(correlationNormal)
# indices = np.triu_indices_from(correlationNormal)
# mask[indices] = True

# grid_kws = {"width_ratios": (.9, .9, .05), "wspace": 0.2}
# f, (ax1, ax2, cbar_ax) = plt.subplots(1, 3, gridspec_kw=grid_kws, figsize = (14, 9))

# # 正常用户-特征相关性展示
# cmap = sns.diverging_palette(150, 0, s=100, l=50, as_cmap=True)
# ax1 =sns.heatmap(correlationNormal, ax = ax1, vmin = -1, vmax = 1, cmap = cmap, square = False, linewidths = 0.5, mask = mask, cbar = False)
# ax1.set_xticklabels(ax1.get_xticklabels(), size = 16); 
# ax1.set_yticklabels(ax1.get_yticklabels(), size = 16); 
# ax1.set_title('Normal', size = 20)

# # 被欺诈的用户-特征相关性展示
# ax2 = sns.heatmap(correlationFraud, vmin = -1, vmax = 1, cmap = cmap, ax = ax2, square = False, linewidths = 0.5, mask = mask, yticklabels = False, cbar_ax = cbar_ax, cbar_kws={'orientation': 'vertical', 'ticks': [-1, -0.5, 0, 0.5, 1]})
# ax2.set_xticklabels(ax2.get_xticklabels(), size = 16); 
# ax2.set_title('Fraud', size = 20)

# plt.show()


# v_feat_col = ["V4", "V13"]
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# for i, cn in enumerate(df[v_feat_col]):
#     plt.figure(figsize=(16, 4))
#     sns.kdeplot(df[cn][df["Class"] == 1], color='red', label='欺诈')
#     sns.kdeplot(df[cn][df["Class"] == 0], color='green', label='正常')
#     plt.xlabel('数值')
#     plt.ylabel('密度')
#     plt.title('概率密度分布图: ' + str(cn))
#     plt.legend()
#     plt.show()


# f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(16,4))
# bins = 30
# ax1.hist(df["Amount"][df["Class"]== 1], bins = bins)
# ax1.set_title('欺诈交易')
# ax1.set_ylabel('交易数量')

# ax2.hist(df["Amount"][df["Class"] == 0], bins = bins)
# ax2.set_title('正常交易')
# ax2.set_ylabel('交易数量')

# plt.xlabel('金额($)')
# plt.yscale('log')
# plt.show()



import seaborn as sns
import matplotlib.pyplot as plt

df['Hour'] = df['Time'].apply(lambda x: int(divmod(x, 3600)[0]))
# 使用 catplot 替代 factorplot
sns.catplot(x="Hour", data=df, kind="count", height=6, aspect=3)
# plt.title('交易次数在时间上的分布')
plt.xlabel('小时(Hour)')
plt.ylabel('交易次数')
plt.xticks(rotation=0) 
plt.tight_layout()
plt.show()
