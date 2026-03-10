import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# 1. 環境設定
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid", font='Microsoft JhengHei')

# 2. 讀取資料
df = pd.read_csv('main.csv')

# 3. 過濾出有明確標示車位價格的資料 (基數夠大才準)
parking_df = df[df['車位總價元'] > 0].copy()

output_dir = 'parking_research'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 定義 Lasso 模型將使用的目標欄位
features_cat = ['鄉鎮市區', '車位類別', '建物型態']
features_num = ['車位筆棟數']

results = []

# A. 處理類別型變數 (ANOVA)
for var in features_cat:
    temp = parking_df[[var, '車位總價元']].dropna()
    groups = [group['車位總價元'].values for name, group in temp.groupby(var)]
    
    if len(groups) > 1:
        f_stat, p_val = stats.f_oneway(*groups)
        overall_mean = temp['車位總價元'].mean()
        ss_total = ((temp['車位總價元'] - overall_mean)**2).sum()
        ss_between = 0
        for g in groups:
            ss_between += len(g) * (g.mean() - overall_mean)**2
        eta_sq = ss_between / ss_total if ss_total != 0 else 0
        
        results.append({
            '變數名稱': var,
            '指標內容': f'F-Stat: {f_stat:.2f}',
            '解釋力 (Power)': round(eta_sq, 4),
            'P-value': f"{p_val:.4e}"
        })

# B. 處理數值型變數 (Pearson)
for var in features_num:
    temp = parking_df[[var, '車位總價元']].dropna()
    corr, p_val = stats.pearsonr(temp[var], temp['車位總價元'])
    results.append({
        '變數名稱': var,
        '指標內容': f'Pearson R: {corr:.4f}',
        '解釋力 (Power)': round(corr**2, 4),
        'P-value': f"{p_val:.4e}"
    })

# 4. 展現結果
results_df = pd.DataFrame(results).sort_values(by='解釋力 (Power)', ascending=False)
results_df.to_csv(os.path.join(output_dir, 'lasso_features_correlation.csv'), index=False, encoding='utf-8-sig')

# 5. 視覺化解釋力
plt.figure(figsize=(10, 6))
sns.barplot(data=results_df, x='解釋力 (Power)', y='變數名稱', palette='magma')
plt.title('Lasso 模型預定變數之解釋力分析', fontsize=16)
plt.xlabel('解釋力 (Power, 越高代表對預測車位價越重要)')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'lasso_features_power.png'))
plt.close()

print("Lasso 預定變數相關性分析已完成。")
print(results_df[['變數名稱', '指標內容', '解釋力 (Power)', 'P-value']].to_string())
