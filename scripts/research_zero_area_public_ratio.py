import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 1. 環境設定
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False
df = pd.read_csv('main.csv')

# 2. 定義二分組 (零面積車位 vs. 其他所有物件)
def categorize_binary(row):
    if row['車位筆棟數'] > 0 and row['車位移轉總面積坪'] == 0:
        return '零面積車位 (可能併入公設)'
    else:
        return '其他所有物件 (含無車位或正常登記)'

df['二分分類'] = df.apply(categorize_binary, axis=1)

# 3. 繪製盒鬚圖
plt.figure(figsize=(10, 8))
# 篩選主要的建物型態
filter_cond = df['建物型態'].isin(['住宅大樓(11層含以上有電梯)', '華廈(10層含以下有電梯)'])
df_plot = df[filter_cond].copy()

ax = sns.boxplot(data=df_plot, x='二分分類', y='公設比', palette='Set1', showfliers=False)

plt.title('高雄市：零面積車位物件 vs. 其他物件之公設比對比', fontsize=16)
plt.ylabel('公設比 (%)', fontsize=12)
plt.xlabel('', fontsize=12)

# 4. 計算統計資料
stats_binary = df_plot.groupby('二分分類')['公設比'].agg(['mean', 'median', 'count'])
print("--- 二分分類公設比統計 ---")
print(stats_binary)

for i, group in enumerate(stats_binary.index):
    mean_val = stats_binary.loc[group, 'mean']
    plt.text(i, mean_val, f'平均: {mean_val:.2f}%', 
             ha='center', va='bottom', fontweight='bold', color='black')

output_dir = 'parking_research'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'zero_area_vs_all_public_ratio.png'))
print(f"\n[完成] 二分對比圖表已儲存至 {output_dir}/zero_area_vs_all_public_ratio.png")
