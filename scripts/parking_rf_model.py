import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error
import os
import warnings

# 1. 環境與資料載入
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False
df = pd.read_csv('main.csv')

output_dir = 'parking_rf_model_results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 2. 資料清理：穩健過濾 (Robust Filtering)
# 僅針對有標價的車位進行 1%~99% 的過濾，移除極端離群值
train_full = df[df['車位總價元'] > 0].copy()
q_low = train_full['車位總價元'].quantile(0.01)
q_high = train_full['車位總價元'].quantile(0.99)
train_clean = train_full[(train_full['車位總價元'] >= q_low) & (train_full['車位總價元'] <= q_high)].copy()

print(f"穩健過濾完成：移除標價低於 {q_low:,.0f} 或高於 {q_high:,.0f} 的極端值")
print(f"訓練樣本數：{len(train_clean)} (原始：{len(train_full)})")

# 3. 特徵工程
cat_features = ['鄉鎮市區', '車位類別', '建物型態']
num_features = ['車位筆棟數', '屋齡']
target = '車位總價元'

# 填補數值型缺失值 (屋齡可能為空)
train_clean[num_features] = train_clean[num_features].fillna(train_clean[num_features].median())

# 編碼類別型變數
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_cat = encoder.fit_transform(train_clean[cat_features])
X = np.hstack([X_cat, train_clean[num_features].values])
y = train_clean[target].values

# 4. 訓練隨機森林模型
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n正在訓練優化後的隨機森林模型 (Random Forest)...")
model = RandomForestRegressor(n_estimators=150, max_depth=20, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 5. 模型評估
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"\n模型評估結果：")
print(f"R-squared: {r2:.4f}")
print(f"平均絕對誤差 (MAE): {mae:,.0f} 元")

# 6. 特徵重要性分析
feature_names = list(encoder.get_feature_names_out(cat_features)) + num_features
importances = pd.DataFrame({
    'Feature': feature_names,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False).head(20)

plt.figure(figsize=(12, 8))
sns.barplot(data=importances, x='Importance', y='Feature', palette='viridis')
plt.title('隨機森林模型：前 20 大特徵重要性', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'rf_feature_importance.png'))
plt.close()

# 7. 全量資料預測與拆算
print("\n正在執行全量資料的車位價格推估...")
# 準備全量資料特徵
df_filled = df.copy()
df_filled[num_features] = df_filled[num_features].fillna(df_filled[num_features].median())
X_all_cat = encoder.transform(df_filled[cat_features])
X_all = np.hstack([X_all_cat, df_filled[num_features].values])

# 預測車位價值
df['RF預測車位價'] = model.predict(X_all)

# 如果原始資料已有標價，則維持原標價
df.loc[df['車位總價元'] > 0, 'RF預測車位價'] = df['車位總價元']

# 8. 計算「淨屋單價」
# 確保預測車位價不超過總價 (避免出現負房屋價值)
df['RF預測車位價'] = df[['RF預測車位價', '總價元']].min(axis=1)

df['建物淨面積坪'] = df['建物移轉總面積坪'] - df['車位移轉總面積坪']
# 處理異常面積 (某些物件可能只有車位或登記錯誤)
df.loc[df['建物淨面積坪'] <= 0, '建物淨面積坪'] = np.nan

df['淨屋單價元坪'] = (df['總價元'] - df['RF預測車位價']) / df['建物淨面積坪']

# 修正極端負值或異常值 (由模型誤差產生)
df.loc[df['淨屋單價元坪'] < 0, '淨屋單價元坪'] = 0

# 9. 產出結果
df.to_csv('main_unbundled_rf.csv', index=False, encoding='utf-8-sig')
print(f"\n[完成] 優化後的修正數據已儲存至 main_unbundled_rf.csv (已修正負值)")
print(f"包含新欄位：RF預測車位價, 淨屋單價元坪")
