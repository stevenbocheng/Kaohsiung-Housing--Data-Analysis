import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. 環境與預作準備
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False
df = pd.read_csv('main_unbundled_lasso_v3_with_pca_final.csv')

# 2. 定義特徵與目標變數
target = '淨屋單價元坪'
# 排除無效座標或極端值
df = df.dropna(subset=['PC1_整體大眾運輸依賴度', target])
# 剔除極端的離群值 (例如價格前 1% 或後 1%)
lower_bound = df[target].quantile(0.01)
upper_bound = df[target].quantile(0.99)
df = df[(df[target] >= lower_bound) & (df[target] <= upper_bound)]

features_num = [
    '屋齡', '公設比', '主建物率', '土地持分率', '交易年', 
    'PC1_整體大眾運輸依賴度', 'PC2_北高雄產業樞紐軸度'
]

features_cat = [
    '鄉鎮市區', '建物型態'
]

# 選擇欄位並處理空值
X = df[features_num + features_cat].copy()
y = df[target]

# 填補數值型空值 (用中位數)
for col in features_num:
    X[col] = X[col].fillna(X[col].median())

# One-Hot Encoding 給類別變數
X_encoded = pd.get_dummies(X, columns=features_cat, drop_first=True)

# 3. 切分訓練集與測試集
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# 4. 訓練 XGBoost 模型
print("開始訓練 XGBoost 模型...")
# 使用樹狀模型 (XGBoost不需要對特徵做標準化)
model = XGBRegressor(
    n_estimators=300, 
    learning_rate=0.1, 
    max_depth=8, 
    random_state=42, 
    n_jobs=-1
)

model.fit(X_train, y_train)

# 5. 評估成效
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = root_mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"\n--- 模型表現 ---")
print(f"R-squared: {r2:.4f}")
print(f"RMSE: {rmse:.2f} 元/坪")
print(f"MAE: {mae:.2f} 元/坪")

# 6. 特徵重要性分析 (Feature Importance)
feature_importances = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': model.feature_importances_
})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# 取前 20 名最重要的特徵畫圖
top_n = 20
plt.figure(figsize=(12, 8))
sns.barplot(data=feature_importances.head(top_n), x='Importance', y='Feature', palette='viridis')
plt.title(f'XGBoost 特徵重要性排行 (Top {top_n})', fontsize=18)
plt.xlabel('影響力權重 (Gain)')
plt.ylabel('特徵')
plt.tight_layout()
os.makedirs('model_results', exist_ok=True)
plt.savefig('model_results/xgboost_feature_importance.png')
print("\n特徵重要性圖表已儲存至 model_results/xgboost_feature_importance.png")

# 印出排名
# 印出排名
print("\n--- PCA 地理特徵目前的排名 ---")
pca_features = [f for f in X_train.columns if 'PC' in f]
for f in pca_features:
    rank = feature_importances[feature_importances['Feature'] == f].index[0]
    rank_pos = list(feature_importances['Feature']).index(f) + 1
    weight = feature_importances[feature_importances['Feature'] == f]['Importance'].values[0]
    print(f"{f}: 第 {rank_pos} 名 (Weight: {weight:.4f})")
