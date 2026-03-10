import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. 環境設定
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False
df = pd.read_csv('main.csv')

# 找出正確欄位
def find_col(possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
        for actual in df.columns:
            if name in actual:
                return actual
    return None

col_year_built = find_col(['建築完成年'])
col_public_ratio = find_col(['公設比'])
col_car_area = find_col(['車位移轉總面積坪'])
col_type = find_col(['建物型態'])

print(f"欄位偵測: 建築年={col_year_built}, 公設比={col_public_ratio}, 車位面積={col_car_area}, 建物型態={col_type}")

# 2. 處理完成年 (只提取前4位年份)
def extract_year(val):
    try:
        s = str(val).split('.')[0]
        if len(s) >= 4:
            year = int(s[:4])
            if 1970 <= year <= 2026:
                return year
    except:
        pass
    return None

df['完成年_整理'] = df[col_year_built].apply(extract_year)

# 3. 分類
def categorize_parking(row):
    if row['車位筆棟數'] == 0:
        return '無車位'
    elif row[col_car_area] == 0:
        return 'A.零面積車位'
    else:
        return 'B.有面積車位'

df['車位分類'] = df.apply(categorize_parking, axis=1)

# 4. 篩選與整理趨勢數據
df_trend = df[(df[col_type].isin(['住宅大樓(11層含以上有電梯)', '華廈(10層含以下有電梯)'])) & 
              (df['完成年_整理'].notnull()) & 
              (df[col_public_ratio] > 0.05) & (df[col_public_ratio] < 0.6)].copy()

# 以 5 年為一個建築時期
df_trend['建築時期'] = (df_trend['完成年_整理'] // 5) * 5

# 5. 繪圖
output_dir = 'parking_historical_research'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

plt.figure(figsize=(12, 7))
# 使用 lineplot 繪製趨勢
sns.lineplot(data=df_trend, x='建築時期', y=col_public_ratio, hue='車位分類', 
             palette='Set1', marker='o', linewidth=2.5)

plt.title('高雄市：歷年建築公設比趨勢 (依車位分類)', fontsize=16)
plt.ylabel('平均公設比', fontsize=12)
plt.xlabel('建築完成年 (每5年)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(title='車位分類', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'historical_public_ratio_trend.png'))

# 6. 輸出數據
stats = df_trend.groupby(['建築時期', '車位分類'])[col_public_ratio].mean().unstack()
print("\n--- 建築年代別平均公設比 ---")
print(stats)
