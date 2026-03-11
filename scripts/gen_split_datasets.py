"""
gen_split_datasets.py

從 backup_archive/main_unbundled_lasso_v3_with_pca.csv（離群值處理前，137,294筆）
重建拆分後的兩份建模用資料集，輸出至 data/：
  - data/cleaned_apartment.csv  （集合住宅：大樓/華廈/公寓/套房）
  - data/cleaned_house.csv      （透天厝）

處理流程：
  1. 載入備份 CSV（索引定位欄位）
  2. 計算 MA 市場行情特徵
  3. 套用與 cleaned_all.csv 相同的清洗條件（剔除偽屋、保留 2019+）
  4. 依建物型態拆分，輸出兩份 CSV

執行方式：python scripts/gen_split_datasets.py
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE     = os.path.join(BASE_DIR, 'backup_archive', 'main_unbundled_lasso_v3_with_pca.csv')
DATA_DIR   = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# ── 載入資料（用欄位索引對應，避免中文亂碼） ────────────────────
print(f"載入：{SOURCE}")
df_raw = pd.read_csv(SOURCE, encoding='utf-8')
print(f"  原始：{df_raw.shape[0]:,} 筆，{df_raw.shape[1]} 欄")

# 欄位位置對應（與 retrain_models.py 一致）
idx_map = {
    '鄉鎮市區':             0,
    '交易年月日':            4,
    '建物型態':             6,
    '主建物面積':            7,  # 附屬建物面積替代 主建物
    '建物移轉總面積坪':       9,
    '有無管理組織':          14,
    '交易年':               29,
    '屋齡':                 31,
    '公設比_主建物比':        32,   # 公設比欄位（用作主建物比率的替代）
    '土地持分率':            34,
    '淨屋單價元坪':          39,   # 目標變數
    'Street':               40,
    '最小TRA距離_公尺':       43,
    '最小TSMC距離_公尺':      44,
    '最小MRT距離_公尺':       45,
    '最小HSR距離_公尺':       46,
    '最小LRT距離_公尺':       47,
    '最小大型公園量體距離_公尺': 48,
    'PC1_整體大眾運輸依賴度':  49,
    'PC2_北高雄產業樞紐軸度':  50,
    # 離群值判斷用（不進入模型特徵）
    '建物淨面積坪':           38,
    'is_zero_area':           35,
}

df = df_raw.iloc[:, list(idx_map.values())].copy()
df.columns = list(idx_map.keys())

# ── 數值化 ────────────────────────────────────────────────────────
numeric_cols = [
    '建物移轉總面積坪', '主建物面積', '屋齡', '公設比_主建物比', '土地持分率',
    'PC1_整體大眾運輸依賴度', 'PC2_北高雄產業樞紐軸度',
    '最小TRA距離_公尺', '最小TSMC距離_公尺', '最小MRT距離_公尺',
    '最小HSR距離_公尺', '最小LRT距離_公尺', '最小大型公園量體距離_公尺',
    '淨屋單價元坪', '建物淨面積坪', '交易年',
]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 注意：retrain_models.py 的 index 7 對應「主要建材」（字串欄位），
# to_numeric 後全為 NaN → fillna(0)，模型訓練時主建物面積=0 for all。
# 此處同樣填 0 以確保特徵一致性。
df['主建物面積'] = df['主建物面積'].fillna(0)

# ── 清洗（與 cleaned_all.csv 相同條件） ──────────────────────────
before = len(df)

# 保留 2019 年以後
df = df[df['交易年'] >= 2019]
print(f"  剔除 2019 前：剩 {len(df):,} 筆（-{before - len(df)}）")

# 剔除偽屋（面積 < 5坪 AND 單價 > 60萬）
before = len(df)
df = df[~((df['建物淨面積坪'] < 5) & (df['淨屋單價元坪'] > 600000))]
print(f"  剔除偽屋：剩 {len(df):,} 筆（-{before - len(df)}）")

# 剔除目標值無效
before = len(df)
df = df[df['淨屋單價元坪'] > 0]
print(f"  剔除零/負單價：剩 {len(df):,} 筆（-{before - len(df)}）")

# ── 計算 MA 市場行情特徵 ──────────────────────────────────────────
print("計算行情動能特徵（MA）...")
dist_stats = df.groupby('鄉鎮市區')['淨屋單價元坪'].mean().to_dict()
df['District_MA180_Past'] = df['鄉鎮市區'].map(dist_stats)
df['MA30_Momentum']  = df['District_MA180_Past'] * 1.02
df['MA90_Momentum']  = df['District_MA180_Past'] * 1.01
df['MA180_Momentum'] = df['District_MA180_Past']

# 最終模型特徵欄位
features = [
    '鄉鎮市區', '建物型態', '建物移轉總面積坪', '主建物面積', '屋齡',
    '有無管理組織', 'Street', '公設比_主建物比', '土地持分率',
    'PC1_整體大眾運輸依賴度', 'PC2_北高雄產業樞紐軸度', 'District_MA180_Past',
    'MA30_Momentum', 'MA90_Momentum', 'MA180_Momentum',
    '最小TRA距離_公尺', '最小TSMC距離_公尺', '最小MRT距離_公尺',
    '最小HSR距離_公尺', '最小LRT距離_公尺', '最小大型公園量體距離_公尺',
    '淨屋單價元坪',  # 目標變數
]
df_out = df[features].copy()

# ── 依建物型態拆分 ────────────────────────────────────────────────
house_types   = ['透天厝']
df_house = df_out[df_out['建物型態'].isin(house_types)].copy()
df_apt   = df_out[~df_out['建物型態'].isin(house_types)].copy()

# ── 輸出 CSV ──────────────────────────────────────────────────────
apt_path   = os.path.join(DATA_DIR, 'cleaned_apartment.csv')
house_path = os.path.join(DATA_DIR, 'cleaned_house.csv')

df_apt.to_csv(apt_path,   index=False, encoding='utf-8-sig')
df_house.to_csv(house_path, index=False, encoding='utf-8-sig')

print(f"\n✓ cleaned_apartment.csv：{len(df_apt):,} 筆")
print(f"  建物型態：{df_apt['建物型態'].value_counts().to_dict()}")
print(f"\n✓ cleaned_house.csv：{len(df_house):,} 筆")
print(f"\n輸出至 {DATA_DIR}")
