import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. 環境與資料載入
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False
df = pd.read_csv('main.csv')

# 找出關鍵面積欄位
def find_col(possible_names):
    for name in possible_names:
        for actual in df.columns:
            if name in actual:
                return actual
    return None

col_total = find_col(['建物移轉總面積坪'])
col_main = find_col(['主建物面積'])
col_sub = find_col(['附屬建物面積'])
col_balcony = find_col(['陽台面積'])
col_car_area = find_col(['車位移轉總面積坪'])
col_type = find_col(['建物型態'])

print(f"總面積: {col_total}, 主建物: {col_main}, 附屬: {col_sub}, 陽台: {col_balcony}, 車位面積: {col_car_area}")

# 2. 計算公設與私有面積
# 私有面積 = 主建物 + 附屬 + 陽台
df['私有面積坪'] = df[col_main] + df[col_sub] + df[col_balcony]
# 共有面積 (公設面積) = 總面積 - 私有面積 - 車位面積
df['共有面積坪'] = df[col_total] - df['私有面積坪'] - df[col_car_area]

# 3. 分組
def categorize(row):
    if row['車位筆棟數'] == 0:
        return '無車位'
    elif row[col_car_area] == 0:
        return 'A.零面積車位'
    else:
        return 'B.有面積車位'

df['車位類別分析'] = df.apply(categorize, axis=1)

# 4. 深度驗證：在相同建物型態下，比較「共有面積 + 車位面積」的總和
# 如果 A 組的共有面積 ~= B 組的 (共有+車位), 則證明 A 組是把車位藏在共有面積裡。

# 只看主建物大樓與華廈
main_types = ['住宅大樓(11層含以上有電梯)', '華廈(10層含以下有電梯)']
df_filtered = df[df[col_type].isin(main_types)].copy()

# 為了公平比較，排除私有面積異常大的物件 (如透天混入)
df_filtered = df_filtered[df_filtered['私有面積坪'] < 100]

# 計算「權狀公項相關總面積」 (共有面積 + 車位面積)
df_filtered['廣義公項面積坪'] = df_filtered['共有面積坪'] + df_filtered[col_car_area]

# 5. 繪製對比圖
output_dir = 'parking_area_verification'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

plt.figure(figsize=(14, 8))
sns.boxplot(data=df_filtered, x=col_type, y='廣義公項面積坪', hue='車位類別分析', showfliers=False)
plt.title('驗證：廣義公項面積 (共有面積 + 車位面積) 之對比', fontsize=16)
plt.ylabel('面積 (坪)')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'area_bundling_verification.png'))

# 6. 統計數據輸出
stats = df_filtered.groupby([col_type, '車位類別分析'])[['私有面積坪', '共有面積坪', col_car_area, '廣義公項面積坪']].mean()
print("\n--- 面積組成統計 (平均值) ---")
print(stats)

# 7. 核心問題：若是 Case A, 總面積裡面到底含不含車位價值？
# 如果 A 與 B 的「私有面積」差不多，但 B 有額外的「車位面積」，
# 那麼 A 的「共有面積」應該要比 B 大約一個車位的面積 (約 8-12 坪) 才對。

# 比較 A 組與「無車位組」的差異
plt.figure(figsize=(12, 7))
sns.boxplot(data=df_filtered, x=col_type, y='共有面積坪', hue='車位類別分析', showfliers=False)
plt.title('共有面積 (傳統公設) 之對比', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'common_area_only_contrast.png'))

print("\n分析報告預覽：")
# 計算差異
for t in main_types:
    try:
        mean_no = stats.loc[(t, '無車位'), '共有面積坪']
        mean_zero = stats.loc[(t, 'A.零面積車位'), '共有面積坪']
        diff = mean_zero - mean_no
        print(f"[{t}] 零面積車位組比無車位組多出的共有面積: {diff:.2f} 坪")
    except:
        pass
