import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error
import os
import joblib

# 1. 環境與資料載入
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False
df = pd.read_csv('main.csv')

output_dir = 'parking_model_results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 2. 資料準備：僅使用有標價的 24% 資料進行訓練
train_data = df[df['車位總價元'] > 0].copy()

# 特徵選擇
cat_features = ['鄉鎮市區', '車位類別', '建物型態']
num_features = ['車位筆棟數']
target = '車位總價元'

# 處理類別型變數 (One-Hot Encoding)
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_cat = encoder.fit_transform(train_data[cat_features])
cat_feature_names = encoder.get_feature_names_out(cat_features)

# 合併數值型
X = np.hstack([X_cat, train_data[num_features].values])
feature_names = list(cat_feature_names) + num_features
y = train_data[target].values

# 3. 拆分訓練與測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 訓練 LassoCV 模型 (自動尋找最佳 alpha)
print("正在訓練 Lasso 回歸模型...")
model = LassoCV(cv=5, random_state=42, max_iter=10000)
model.fit(X_train, y_train)

# 5. 模型評估
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"\n模型評估結果:")
print(f"R-squared: {r2:.4f}")
print(f"平均絕對誤差 (MAE): {mae:,.0f} 元")

# 6. 係數分析 (哪些因素最影響價格)
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': model.coef_
})
coef_df['Abs_Coef'] = coef_df['Coefficient'].abs()
top_coefs = coef_df.sort_values(by='Abs_Coef', ascending=False).head(20)

plt.figure(figsize=(12, 8))
sns.barplot(data=top_coefs, x='Coefficient', y='Feature', palette='RdBu_r')
plt.title('Lasso 模型係數分析 (前 20 大影響因子)', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'lasso_coefficients.png'))
plt.close()

# 7. 預測與實際對照圖
plt.figure(figsize=(10, 10))
plt.scatter(y_test, y_pred, alpha=0.3, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('實際車位價')
plt.ylabel('預測車位價')
plt.title(f'車位價格預測：實際 vs 預測 (R2={r2:.2f})')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'predicted_vs_actual_parking.png'))
plt.close()

# 8. 執行全量預測 (對那 76% 的包裹資料)
print("\n正在對全量資料執行車位價格拆算...")
# 準備全量特徵
X_all_cat = encoder.transform(df[cat_features])
X_all = np.hstack([X_all_cat, df[num_features].values])

# 預測
df['Lasso預測車位價'] = model.predict(X_all)

# 如果原始資料已經有標價，則使用原始標價
df.loc[df['車位總價元'] > 0, 'Lasso預測車位價'] = df['車位總價元']

# 9. 計算「淨屋單價」
# 確保分母不為 0
df['建物淨面積坪'] = df['建物移轉總面積坪'] - df['車位移轉總面積坪']
# 避免除以 0 或是面積為負的異常值
df.loc[df['建物淨面積坪'] <= 0, '建物淨面積坪'] = np.nan

df['淨屋單價元坪'] = (df['總價元'] - df['Lasso預測車位價']) / df['建物淨面積坪']

# 10. 產出結果
df.to_csv('main_unbundled.csv', index=False, encoding='utf-8-sig')
print(f"\n成果已儲存至 main_unbundled.csv")
print(f"包含新欄位: Lasso預測車位價, 淨屋單價元坪")
