import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. 環境設定
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font='Microsoft JhengHei')

# 讀取資料
df = pd.read_csv('main.csv')
parking_df = df[df['車位總價元'] > 0].copy()

output_dir = 'parking_research'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- 針對 Lasso 四大變數進行分佈研究 ---
lasso_features = ['鄉鎮市區', '車位類別', '建物型態', '車位筆棟數']

fig, axes = plt.subplots(2, 2, figsize=(20, 16))

# A. 鄉鎮市區 (Top 15)
top_districts = parking_df['鄉鎮市區'].value_counts().head(15).index
sns.boxplot(data=parking_df[parking_df['鄉鎮市區'].isin(top_districts)], 
            x='鄉鎮市區', y='車位總價元', ax=axes[0,0], palette='viridis')
axes[0,0].set_title('鄉鎮市區與車位價格 (Top 15)')
axes[0,0].tick_params(axis='x', rotation=45)

# B. 車位類別
sns.boxplot(data=parking_df, x='車位類別', y='車位總價元', ax=axes[0,1], palette='Set2')
axes[0,1].set_title('車位類別與車位價格')

# C. 建物型態
sns.boxplot(data=parking_df, x='建物型態', y='車位總價元', ax=axes[1,0], palette='coolwarm')
axes[1,0].set_title('建物型態與車位價格')
axes[1,0].tick_params(axis='x', rotation=45)

# D. 車位筆棟數 (與價格散佈)
sns.regplot(data=parking_df, x='車位筆棟數', y='車位總價元', ax=axes[1,1], 
            scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
axes[1,1].set_title('車位數量與總價關係')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'lasso_features_distribution_research.png'))
plt.close()

print(f"Lasso 變數分佈研究圖表已生成於 {output_dir}")
