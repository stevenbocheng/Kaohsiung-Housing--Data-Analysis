import pandas as pd
import numpy as np
import joblib
import json
import os
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

def train():
    source_path = "backup_archive/main_unbundled_lasso_v3_with_pca.csv"
    if not os.path.exists(source_path):
        print(f"找不到原始資料: {source_path}")
        return

    print("載入資料 (使用索引定位)...")
    df_raw = pd.read_csv(source_path)
    
    # 索引映射
    idx_map = {
        '鄉鎮市區': 0,
        '交易年月': 4,
        '建物型態': 6,
        '主建物面積': 7,
        '建物移轉總面積坪': 9,
        '有無管理組織': 14,
        '屋齡': 31,
        '公設比_主建物比': 32,
        '土地持分率': 34,
        '淨單價元坪': 39,
        'Street': 40,
        '最小TRA距離_公尺': 43,
        '最小TSMC距離_公尺': 44,
        '最小MRT距離_公尺': 45,
        '最小HSR距離_公尺': 46,
        '最小LRT距離_公尺': 47,
        '最小大型公園量體距離_公尺': 48,
        'PC1_整體大眾運輸依賴度': 49,
        'PC2_北高雄產業樞紐軸度': 50
    }

    # 提取並重命名
    df = df_raw.iloc[:, list(idx_map.values())].copy()
    df.columns = list(idx_map.keys())

    # 有些欄位可能是 ' ' 或其他字串，強制轉為數值
    numeric_features = [
        '建物移轉總面積坪', '主建物面積', '屋齡', '公設比_主建物比', '土地持分率',
        'PC1_整體大眾運輸依賴度', 'PC2_北高雄產業樞紐軸度',
        '最小TRA距離_公尺', '最小TSMC距離_公尺', '最小MRT距離_公尺', 
        '最小HSR距離_公尺', '最小LRT距離_公尺', '最小大型公園量體距離_公尺'
    ]
    
    for col in numeric_features + ['淨單價元坪']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 移除目標值為 0 的資料
    df = df[df['淨單價元坪'] > 0]

    # 行情指標
    dist_stats = df.groupby('鄉鎮市區')['淨單價元坪'].mean().to_dict()
    df['District_MA180_Past'] = df['鄉鎮市區'].map(dist_stats)
    df['MA30_Momentum'] = df['District_MA180_Past'] * 1.02
    df['MA90_Momentum'] = df['District_MA180_Past'] * 1.01
    df['MA180_Momentum'] = df['District_MA180_Past']
    
    features = [
        '鄉鎮市區', '建物型態', '建物移轉總面積坪', '主建物面積', '屋齡', 
        '有無管理組織', 'Street', '公設比_主建物比', '土地持分率',
        'PC1_整體大眾運輸依賴度', 'PC2_北高雄產業樞紐軸度', 'District_MA180_Past',
        'MA30_Momentum', 'MA90_Momentum', 'MA180_Momentum',
        '最小TRA距離_公尺', '最小TSMC距離_公尺', '最小MRT距離_公尺', 
        '最小HSR距離_公尺', '最小LRT距離_公尺', '最小大型公園量體距離_公尺'
    ]
    
    target = '淨單價元坪'
    categorical_features = ['鄉鎮市區', '建物型態', 'Street', '有無管理組織']
    
    # 強健分割
    counts = df['建物型態'].value_counts()
    if len(counts) < 2:
        print("樣本不足以進行分割！")
        return
        
    house_label = counts.index[1]
    df_house = df[df['建物型態'] == house_label].copy()
    df_apt = df[df['建物型態'] != house_label].copy()

    for col in categorical_features:
        df_apt[col] = df_apt[col].astype(str)
        df_house[col] = df_house[col].astype(str)

    os.makedirs("models", exist_ok=True)

    # --- 1. CatBoost ---
    print("訓練 CatBoost (Apartment)...")
    model_apt = CatBoostRegressor(iterations=300, depth=6, learning_rate=0.1, 
                                 cat_features=categorical_features, verbose=False)
    model_apt.fit(df_apt[features], df_apt[target])
    joblib.dump(model_apt, "models/catboost_apartment_model.pkl")

    # --- 2. LightGBM ---
    print("訓練 LightGBM (House)...")
    for col in categorical_features:
        df_house[col] = df_house[col].astype('category')
    model_house = LGBMRegressor(n_estimators=300, max_depth=6, num_leaves=31, random_state=42)
    model_house.fit(df_house[features], df_house[target])
    joblib.dump(model_house, "models/lgbm_house_model.pkl")

    # 匯出特徵清單
    with open("models/apartment_features.json", "w", encoding="utf-8") as f:
        json.dump(features, f, ensure_ascii=False)
    with open("models/house_features.json", "w", encoding="utf-8") as f:
        json.dump(features, f, ensure_ascii=False)

    print("✅ 全部任務完成！")

if __name__ == "__main__":
    train()
