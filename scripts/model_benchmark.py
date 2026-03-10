import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('main.csv')
p_data = df[df['車位總價元'] > 0].copy()

# 1. 處理極端離群值 (去除上下 1%)
q_low = p_data['車位總價元'].quantile(0.01)
q_high = p_data['車位總價元'].quantile(0.99)
p_clean = p_data[(p_data['車位總價元'] >= q_low) & (p_data['車位總價元'] <= q_high)].copy()

print(f"原始資料筆數: {len(p_data)}, 清理後 (1%~99%): {len(p_clean)}")

# 特徵準備
# 注意：Lasso 模型需要比較穩定的特徵，我們加入屋齡
cat_features = ['鄉鎮市區', '車位類別', '建物型態']
num_features = ['車位筆棟數', '屋齡'] 
target = '車位總價元'

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
X_cat = encoder.fit_transform(p_clean[cat_features])
X = np.hstack([X_cat, p_clean[num_features].fillna(p_clean[num_features].median()).values])
y = p_clean[target].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 模型比較
models = {
    'LassoCV (Cleaned)': LassoCV(cv=5, random_state=42),
    'RidgeCV (Cleaned)': RidgeCV(cv=5),
    'RandomForest (100 trees)': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
}

print("\n--- 模型效能比較 ---")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"{name:25s} | R2: {r2_score(y_test, y_pred):.4f} | MAE: {mean_absolute_error(y_test, y_pred):,.0f}")
