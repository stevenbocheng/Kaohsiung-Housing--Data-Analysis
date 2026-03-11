"""
gen_shap_charts.py

從已訓練模型重新生成 SHAP 可解釋性圖表，輸出至 visuals/shaps/。
需先執行 gen_split_datasets.py 生成 data/cleaned_apartment.csv 和 data/cleaned_house.csv。

執行方式：python scripts/gen_shap_charts.py

輸出圖表：
  - visuals/shaps/shap_summary_apartment.png  — 集合住宅 CatBoost SHAP 摘要圖
  - visuals/shaps/shap_summary_house.png      — 透天厝 LightGBM SHAP 摘要圖
  - visuals/shaps/shap_bar_apartment.png      — 集合住宅 SHAP 特徵重要性（條形圖）
  - visuals/shaps/shap_bar_house.png          — 透天厝 SHAP 特徵重要性（條形圖）
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import joblib
import json
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(BASE_DIR, 'data')
MODEL_DIR  = os.path.join(BASE_DIR, 'models')
OUT_DIR    = os.path.join(BASE_DIR, 'visuals', 'shaps')
os.makedirs(OUT_DIR, exist_ok=True)

# 特徵顯示名稱（映射簡化標籤）
FEAT_LABELS = {
    '鄉鎮市區':                 '行政區',
    '建物型態':                 '建物型態',
    '建物移轉總面積坪':          '建物總面積（坪）',
    '主建物面積':               '主建物面積',
    '屋齡':                     '屋齡',
    '有無管理組織':              '有無管理組織',
    'Street':                   '街道',
    '公設比_主建物比':           '公設比',
    '土地持分率':               '土地持分率',
    'PC1_整體大眾運輸依賴度':    'PC1（交通依賴度）',
    'PC2_北高雄產業樞紐軸度':    'PC2（北高雄產業軸）',
    'District_MA180_Past':      '行政區均價（180日）',
    'MA30_Momentum':            '短期行情動能（30日）',
    'MA90_Momentum':            '中期行情動能（90日）',
    'MA180_Momentum':           '長期行情動能（180日）',
    '最小TRA距離_公尺':          '最近台鐵站距離',
    '最小TSMC距離_公尺':         '台積電廠距離',
    '最小MRT距離_公尺':          '最近捷運站距離',
    '最小HSR距離_公尺':          '最近高鐵站距離',
    '最小LRT距離_公尺':          '最近輕軌站距離',
    '最小大型公園量體距離_公尺':  '最近大型公園距離',
}

categorical_features = ['鄉鎮市區', '建物型態', 'Street', '有無管理組織']
SAMPLE_N = 3000   # SHAP 計算抽樣數（控制速度）


def load_model_and_data(model_name, csv_name, use_category=False):
    """
    use_category=True  → LightGBM：分類欄位轉為 category dtype（與訓練一致）
    use_category=False → CatBoost：分類欄位為字串即可
    """
    model_path = os.path.join(MODEL_DIR, model_name)
    data_path  = os.path.join(DATA_DIR,  csv_name)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型不存在：{model_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"資料不存在：{data_path}\n"
            f"請先執行：python scripts/gen_split_datasets.py"
        )

    model = joblib.load(model_path)

    with open(os.path.join(MODEL_DIR, 'apartment_features.json'), encoding='utf-8') as f:
        features = json.load(f)
    features = [f for f in features if f != '淨屋單價元坪' and f != '淨單價元坪']

    df = pd.read_csv(data_path, encoding='utf-8-sig')

    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"  [警告] 缺少特徵欄位：{missing}")
        features = [f for f in features if f in df.columns]

    X = df[features].copy()
    # 數值欄位 NaN 填 0（與 retrain_models.py 一致）
    for col in X.select_dtypes(include='number').columns:
        X[col] = X[col].fillna(0)

    for col in categorical_features:
        if col not in X.columns:
            continue
        if use_category:
            # LightGBM：需與訓練時相同的 category dtype
            X[col] = X[col].astype('category')
        else:
            X[col] = X[col].astype(str)

    return model, X, features


def shorten_labels(features):
    return [FEAT_LABELS.get(f, f) for f in features]


def plot_shap(model, X, features, out_prefix, model_label, is_catboost=False):
    np.random.seed(42)
    n = min(SAMPLE_N, len(X))
    X_sample = X.sample(n, random_state=42).reset_index(drop=True)

    print(f"  計算 SHAP 值（n={n}）...")
    if is_catboost:
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_sample)
    short_labels = shorten_labels(features)

    # ── 圖1：SHAP Beeswarm Summary Plot ───────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=short_labels,
        show=False, plot_size=None,
        max_display=20,
    )
    plt.title(f'{model_label} — SHAP 特徵影響摘要\n（紅=高特徵值推高價格，藍=低特徵值壓低價格）',
              fontsize=12, fontweight='bold', pad=10)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, f'{out_prefix}_summary.png')
    plt.savefig(out, bbox_inches='tight', dpi=130)
    plt.close()
    print(f"  ✓ {out}")

    # ── 圖2：SHAP Bar Plot（平均重要性） ──────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=short_labels,
        plot_type='bar',
        show=False, plot_size=None,
        max_display=20,
    )
    plt.title(f'{model_label} — SHAP 特徵重要性（平均 |SHAP 值|）',
              fontsize=12, fontweight='bold', pad=10)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, f'{out_prefix}_bar.png')
    plt.savefig(out, bbox_inches='tight', dpi=130)
    plt.close()
    print(f"  ✓ {out}")


# ─────────────────────────────────────────
# 主程式
# ─────────────────────────────────────────
if __name__ == '__main__':
    print("\n[1/2] CatBoost — 集合住宅 SHAP 圖...")
    try:
        model_apt, X_apt, feats_apt = load_model_and_data(
            'catboost_apartment_model.pkl', 'cleaned_apartment.csv', use_category=False
        )
        plot_shap(model_apt, X_apt, feats_apt,
                  'shap_apartment', 'CatBoost 集合住宅模型', is_catboost=True)
    except Exception as e:
        print(f"  [錯誤] {e}")

    print("\n[2/2] LightGBM — 透天厝 SHAP 圖...")
    try:
        model_house, X_house, feats_house = load_model_and_data(
            'lgbm_house_model.pkl', 'cleaned_house.csv', use_category=True
        )
        plot_shap(model_house, X_house, feats_house,
                  'shap_house', 'LightGBM 透天厝模型', is_catboost=False)
    except Exception as e:
        print(f"  [錯誤] {e}")

    print("\n完成！圖表輸出至 visuals/shaps/")
