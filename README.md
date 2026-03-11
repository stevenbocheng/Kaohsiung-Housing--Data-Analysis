# 高雄房價數據分析 & 儀表板

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://kaohsiung-housing--data-analysis-kiogievjdtcmtownemeud8.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

基於機器學習的高雄房價預測與分析儀表板。透過內政部實價登錄數據（2019–2026），結合地理空間分析、車位拆算（Lasso 回歸）與 GBDT 演算法，提供精準的房價估算與市場趨勢洞察。

---

## 核心功能

| 頁面 | 功能說明 |
|------|---------|
| **即時估價** | 輸入物件條件，AI 預測淨屋單價與總價，並附 SHAP 瀑布圖解釋預測依據 |
| **市場行情地圖** | 高雄 31 個行政區的互動式地理面量圖，支援平均值、中位數、成交筆數（全部／離群值）切換 |
| **EDA 數據藝廊** | 完整資料處理流程：車位拆算研究（Lasso 回歸補值去除）、離群值處理、台積電效應地區分析 |
| **技術與模型說明** | 資料清洗流水線、特徵工程原理、模型競賽結果、SHAP 可解釋性 |

---

## 模型效能

採用**分流建模**策略，針對集合住宅（大樓／華廈／公寓）與透天厝分別訓練最優模型：

| 建物類型 | 演算法 | MAPE | MAE（元/坪） | R² |
|----------|--------|------|-------------|-----|
| 集合住宅（99,989 筆） | CatBoost | **11.90%** | 34,893 | 0.6252 |
| 透天厝（37,255 筆） | LightGBM | **14.92%** | 55,902 | 0.5132 |

### 模型限制說明

- **R² 偏低（0.52～0.62）的原因**：台灣房市受人脈、急售、繼承等人為因素影響，非線性程度高，即便業界主流模型 R² 也普遍落在 0.5～0.7 區間，此數值屬於合理範圍。
- **MAPE 解讀**：MAPE 11～15% 代表平均預測誤差約每坪 3～5 萬元，適用於輔助決策參考，不建議直接作為報價依據。
- **資料範圍限制**：模型僅涵蓋高雄市 2019–2026 年的住宅交易，不適用於其他城市或非住宅用途物件。

---

## 特徵工程亮點

### 車位拆算（Lasso 回歸）
台灣實價登錄的交易總價通常包含車位，導致「單價」計算失真。本專案使用 Lasso 回歸預測每筆交易的車位價格，再從總價扣除，還原真實的**淨屋單價**。

研究過程中並發現「零面積車位」現象（有車位但面積為 0），追查後確認為早期建案將車位面積納入公設登記所致，並設計了對應的處理策略。

### PCA 地理降維
將 5 個交通設施距離特徵（捷運、輕軌、火車、高鐵、台積電楠梓廠）透過 PCA 壓縮為兩個主成分：
- **PC1**（解釋 76.7% 變異）：整體大眾運輸依賴度
- **PC2**（解釋 13.4% 變異）：北高雄產業樞紐軸度

### 市場動能指標
引入行政區 30／90／180 日移動平均價格，讓模型具備「行情感知」能力，捕捉區域漲跌趨勢。

---

## 專案架構

```
kaohsiung-house-price-analysis/
├── app/
│   ├── main.py                    # Streamlit 主程式（四頁式導航）
│   ├── market_context.json        # 31 個行政區市場行情快照
│   ├── kaohsiung_districts.json   # 行政區 GeoJSON 邊界（地圖用）
│   └── street_coords_cache.json   # 街道坐標快取（加速定位）
├── data/
│   ├── cleaned_all.csv            # 清洗後資料集（137,244 筆，51 欄，EDA 用）
│   └── map_data.csv               # 市場行情地圖精簡資料（含離群值，3 欄）
├── models/
│   ├── catboost_apartment_model.pkl   # 集合住宅模型
│   ├── lgbm_house_model.pkl           # 透天厝模型
│   ├── apartment_features.json        # 集合住宅特徵清單
│   └── house_features.json            # 透天厝特徵清單
├── visuals/
│   ├── eda/                       # 探索性分析圖表（50+ 張）
│   ├── shaps/                     # SHAP 特徵解釋圖表
│   └── reports/                   # 離群值分析報告圖
├── scripts/                       # 資料處理與模型訓練腳本（28 個）
│   ├── retrain_models.py          # 重新訓練 CatBoost / LightGBM 模型
│   ├── gen_market_context.py      # 重建行政區市場行情快照
│   ├── rebuild_data_assets.py     # 同步 cleaned_all.csv 與快照
│   ├── gen_missing_eda_charts.py  # 重新生成 EDA 圖表
│   ├── gen_parking_research_charts.py  # 車位研究圖表
│   ├── gen_shap_charts.py         # SHAP 特徵解釋圖表
│   └── gen_split_datasets.py      # 資料集切分（含離群值版本）
├── backup_archive/                # 本地原始中間檔（不納入 git，83MB+）
│   └── main_unbundled_lasso_v3_with_pca.csv  # 含離群值完整資料集（來源）
├── requirements.txt
└── README.md
```

---

## 資料說明

| 項目 | 內容 |
|------|------|
| 來源 | 內政部不動產交易實價查詢服務網（H 類住宅交易） |
| 範圍 | 高雄市 2019 年 1 月 – 2026 年 |
| 筆數 | 137,244 筆（清洗後） |
| 更新 | 下載新資料後執行 `rebuild_data_assets.py` 重新產出所有資產 |

> **注意**：`backup_archive/` 目錄因超過 GitHub 大小限制（83MB+），已排除於版本控制。市場行情地圖使用的精簡版 `data/map_data.csv`（僅含鄉鎮市區、淨屋單價、建物型態，約 7.6MB）已納入 git 追蹤。

---

## 快速開始

### 本地執行

```bash
git clone https://github.com/stevenbocheng/Kaohsiung-Housing--Data-Analysis
cd Kaohsiung-Housing--Data-Analysis
pip install -r requirements.txt
streamlit run app/main.py
```

### Streamlit Cloud 部署

1. Fork 此倉庫至您的 GitHub
2. 至 [Streamlit Cloud](https://share.streamlit.io/) 建立新 App
3. 主程式設定為 `app/main.py`，Python 版本選 3.10+

### 模型與資料更新流程

```bash
# 1. 重新訓練模型（需要 backup_archive/ 中的原始資料）
python scripts/retrain_models.py

# 2. 更新市場行情快照
python scripts/gen_market_context.py

# 3. 同步資料資產（含重新產出 data/map_data.csv）
python scripts/rebuild_data_assets.py

# 4. 重新生成 EDA 圖表
python scripts/gen_missing_eda_charts.py
```

---

## 技術棧

| 類別 | 工具 |
|------|------|
| Web 框架 | Streamlit ≥ 1.35.0 |
| ML 模型 | CatBoost、LightGBM、XGBoost |
| 可解釋性 | SHAP |
| 地圖視覺化 | Folium + streamlit-folium |
| 互動圖表 | Plotly |
| 資料處理 | Pandas、NumPy、scikit-learn |
| 地理編碼 | ArcGIS REST API |

---

*資料來源：內政部不動產交易實價查詢服務網。本專案僅供學術研究與數據展示，實際成交請以市場概況為準。*
