import pandas as pd
import numpy as np

# 讀取完整特徵資料集
df = pd.read_csv('main_unbundled_lasso_v3_with_pca_final.csv')

# 定義標的欄位
col = '淨屋單價元坪'

# 計算 99% 門檻
threshold = df[col].quantile(0.99)
outliers = df[df[col] > threshold].copy()

print(f"--- 離群值 (Top 1%) 分析報告 ---")
print(f"離群門檻 (99th Percentile): {threshold:,.2f} 元/坪")
print(f"離群值總筆數: {len(outliers)} 筆")

# 1. 依行政區分佈
print("\n[1. 行政區分佈-前五名]")
print(outliers['鄉鎮市區'].value_counts().head(5))

# 2. 依建物型態分佈
print("\n[2. 建物型態分佈]")
print(outliers['建物型態'].value_counts())

# 3. 具体案例檢視
cols_to_show = ['鄉鎮市區', '土地位置建物門牌', '建物型態', '屋齡', '建物移轉總面積坪', '總價元', '淨屋單價元坪']
print("\n[3. 具體案例 (單價前10名)]")
print(outliers[cols_to_show].sort_values(by=col, ascending=False).head(10))

# 4. 面積分佈探討 (離群值)
print("\n[4. 離群值之面積分佈]")
print(outliers['建物移轉總面積坪'].describe())
