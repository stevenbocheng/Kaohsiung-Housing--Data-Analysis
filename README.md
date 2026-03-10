# 🏙️ 高雄房價數據分析&儀錶板 (Kaohsiung House Price Analysis)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)

這是一個基於機器學習的高雄房價預測與分析儀表板。透過內政部實價登錄數據，結合地理空間分析 (POI) 與先進的 GBDT 演算法，提供精準的房價估算與市場趨勢洞察。

## 🚀 核心功能

- **🤖 AI 智能估價**：提供「集合住宅」與「透天厝」兩類專屬模型，MAPE 表現優異。
- **🗺️ 市場行情地圖**：互動式地理面量圖，支援平均價、中位數及預設豪宅分布切換。
- **📊 EDA 數據藝廊**：深度解析數據背後的邏輯，包含相關性熱力圖、車位拆算研究及台積電效應專題。
- **🧠 技術與模型說明**：透明的資料清洗流水線與 SHAP 模型可解釋性展示。

## 🛠️ 技術對戰與特徵工程

- **模型對決**: CatBoost (集合住宅組) 與 LightGBM (透天組) 最終決選。
- **特徵工程**:
  - **PCA 降維**: 將 5 大 POI 距離轉化為空間位置指標。
  - **Lasso 拆算**: 自動剝離含車位物件的車位價，求得淨屋單價。
  - **Momentum**: 引入行政區歷史行情動能。

## 📂 專案架構

```text
kaohsiung-house-price-analysis/
├── app/                    # 應用程式核心 (main.py, config)
├── data/                   # 核心數據 (cleaned_all.csv)
├── models/                 # 訓練好的實體模型 (.pkl)
├── visuals/                # 分類存放的視覺化圖表
│   ├── eda/                # 數據探索圖表
│   ├── shaps/              # 模型解釋性圖表
│   └── reports/            # 整合性分析報告
├── scripts/                # 開發與處理腳本
├── requirements.txt        # 套件依賴清單
└── README.md               # 專案說明文件
```

## 📦 部署指引

1. **GitHub 上傳**: 請確保包含上述所有核心目錄 (backup_archive 已自動排除)。
2. **Streamlit Cloud**: 連結您的 GitHub 倉庫，主程式設定為 `app/main.py` 即可。

---

*Disclaimer: 本專案僅供學術研究與數據展示，實際成交請以內政部實價登錄與市場概況為準。*
