import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import shap
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
import plotly.express as px

# 1. 初始化與資源載入
st.set_page_config(page_title="高雄房價數據戰情室", layout="wide", initial_sidebar_state="expanded")

# 修正中文顯示
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

@st.cache_resource
def load_assets():
    # 模型
    models = {
        "apt": joblib.load("models/catboost_apartment_model.pkl"),
        "house": joblib.load("models/lgbm_house_model.pkl")
    }
    # 特徵清單
    with open("models/apartment_features.json", "r", encoding="utf-8") as f:
        apt_feats = json.load(f)
    with open("models/house_features.json", "r", encoding="utf-8") as f:
        house_feats = json.load(f)
    
    # 座標快照 (用於預測地圖定位)
    with open("app/street_coords_cache.json", "r", encoding="utf-8") as f:
        coords = json.load(f)

    # 地段特徵快照 (MA180, PC1, PC2)
    with open("app/market_context.json", "r", encoding="utf-8") as f:
        market_ctx = json.load(f)
    
    return models, {"apt": apt_feats, "house": house_feats}, coords, market_ctx

@st.cache_data
def load_data():
    # 使用已經過 Step 5 清洗後的完整資料集 (已排除 2019以前 與 偽屋)
    df = pd.read_csv("data/cleaned_all.csv")
    return df

@st.cache_data
def load_map_data():
    # 市場行情地圖使用含離群值的精簡欄位檔（鄉鎮市區、淨屋單價元坪、建物型態）
    df = pd.read_csv("data/map_data.csv", low_memory=False)
    return df

# 注意：需確保路徑正確或檔案已移動
try:
    models, features, coords, market_ctx = load_assets()
    df_all = load_data()
    df_map = load_map_data()
except Exception as e:
    st.error(f"資源載入失敗，請檢查目錄結構: {e}")
    st.info("請確認是否已執行 gen_market_context.py 產出 market_context.json")
    st.stop()

# --- 側邊欄導航 ---
st.sidebar.title("🗂️ 選單")
page = st.sidebar.radio("切換頁面", ["即時估價", "市場行情地圖", "EDA 數據藝廊", "技術與模型說明"])
# --- 共通函數：行政區中心點 ---
def get_district_center(dist_name):
    # 簡單從快照中抓取含該區域名稱的第一筆座標
    for k, v in coords.items():
        if dist_name in k:
            return v['lat'], v['lon']
    return 22.6273, 120.3014 # 預設高雄中心

# =================================================================
# Page 1: AI 預測與預算導航
# =================================================================
if page == "即時估價":
    st.title("即時估價 & 預算導航")
    
    tab_predict, tab_budget = st.tabs(["精準估價", "預算找房"])
    
    # --- 共通函數：執行預測 ---
    def run_prediction(h_type, dist, age, area, p_ratio):
        target_key = "apt" if "集合住宅" in h_type else "house"
        model = models[target_key]
        feats = features[target_key]
        ctx = market_ctx.get(dist, {})

        p_safe = min(float(p_ratio), 99.0)   # 防止分母為 0

        input_dict = {
            '交易年': 2026, '交易月': 3, '屋齡': age, '建物移轉總面積坪': area,
            '主建物面積': 0.0,   # 訓練時 col[7] 為主要建材(字串)，強制轉數值後全為 NaN→fillna(0)
            '公設比_主建物比': p_safe / (100.0 - p_safe),
            '土地持分率': 0.15 if target_key=="apt" else 1.0,
            '有無管理組織': '有',
            'Street': '不明',
            'PC1_整體大眾運輸依賴度': ctx.get('PC1', 0.0),
            'PC2_北高雄產業樞紐軸度': ctx.get('PC2', 0.0),
            'District_MA180_Past': ctx.get('District_MA180_Past', 250000),
            'MA30_Momentum': ctx.get('MA30_Momentum', 255000),
            'MA90_Momentum': ctx.get('MA90_Momentum', 252500),
            'MA180_Momentum': ctx.get('MA180_Momentum', 250000),
            '鄉鎮市區': dist, '建物用途大類': '住家用', '主要建材': '鋼筋混凝土造',
            '土地分區大類': '住宅區', '建物型態': '住宅大樓(11層含以上有電梯)' if target_key=="apt" else '透天厝',
            '最小TRA距離_公尺': ctx.get('最小TRA距離_公尺', 1000.0),
            '最小TSMC距離_公尺': ctx.get('最小TSMC距離_公尺', 5000.0),
            '最小MRT距離_公尺': ctx.get('最小MRT距離_公尺', 800.0),
            '最小HSR距離_公尺': ctx.get('最小HSR距離_公尺', 3000.0),
            '最小LRT距離_公尺': ctx.get('最小LRT距離_公尺', 2000.0),
            '最小大型公園量體距離_公尺': ctx.get('最小大型公園量體距離_公尺', 500.0)
        }
        X_df = pd.DataFrame([input_dict])[feats]

        # CatBoost 需要字串，LightGBM 需要 category dtype
        cat_cols = ['鄉鎮市區', '建物型態', '有無管理組織', 'Street']
        for c in cat_cols:
            if c in X_df.columns:
                if target_key == "house":
                    X_df[c] = X_df[c].astype('category')
                else:
                    X_df[c] = X_df[c].astype(str)

        # 模型直接預測原始單價（元/坪），訓練時 target = '淨單價元坪'，無 log 轉換
        pred = float(model.predict(X_df)[0])

        # 防護：若結果不合理則回傳 None
        if np.isinf(pred) or np.isnan(pred) or pred <= 0 or pred > 2_000_000:
            return None, model, X_df
        return pred, model, X_df

    with tab_predict:
        st.subheader("物件條件輸入")
        col_input, col_out = st.columns([1, 1.2])
        with col_input:
            h_type_p = st.selectbox("建物類型", ["集合住宅 (大樓/華廈/公寓)", "透天厝"], key="p_type")
            dist_p = st.selectbox("行政區", sorted(market_ctx.keys()), key="p_dist")
            age_p = st.slider("物件屋齡 (年)", 0, 60, 10, key="p_age")
            area_p = st.number_input("物件坪數", 5.0, 200.0, 35.0, key="p_area")
            use_custom_p = st.checkbox("自定義公設比 (預設 32%)")
            p_ratio_p = st.slider("公設比 (%)", 0.0, 50.0, 32.0) if use_custom_p else 32.0
            
            # st.markdown("---")
            btn_pressed = st.button("開始估價", type="primary", use_container_width=True)
            # st.info(f"系統偵測該區行情基準: {market_ctx.get(dist_p, {}).get('District_MA180_Past', 0)/10000:,.1f} 萬元/坪")
            
        with col_out:
            st.markdown("#### ⚡ 即時估價結果")
            if btn_pressed:
                pred_val, model, X_df = run_prediction(h_type_p, dist_p, age_p, area_p, p_ratio_p)

                if pred_val is None:
                    st.error("預測失敗：輸入參數超出合理範圍，請調整後重試。")
                else:
                    # 計算誤差區間 (以 12% 為例)
                    lower_p, upper_p = pred_val * 0.88, pred_val * 1.12

                    res_col1, res_col2 = st.columns(2)
                    with res_col1:
                        st.metric("AI 預測單價", f"{pred_val/10000:,.2f} 萬元/坪", delta=None)
                        st.markdown(f"###### 💡 市場合理區間: {lower_p/10000:,.2f} ~ {upper_p/10000:,.2f} 萬元/坪")
                    with res_col2:
                        st.metric("預估成交總價", f"{(pred_val * area_p / 10000):,.1f} 萬元")
                
                    with st.expander("🔍 查看預算拆解 (SHAP)"):
                        explainer = shap.TreeExplainer(model)
                        shap_v = explainer(X_df)

                        # 顯示 Base Value (訓練集平均預測值，元/坪)
                        base_val = float(explainer.expected_value)
                        st.write(f"📊 全高雄 `{h_type_p}` 平均基準價: **{base_val/10000:,.2f}** 萬元/坪")

                        fig, ax = plt.subplots(figsize=(20, 8))
                        shap.plots.waterfall(shap_v[0], max_display=10, show=False)
                        st.pyplot(fig)
            else:
                st.info("請在左側輸入條件後，點擊「開始智能估價」按鈕。")

    with tab_budget:
        st.subheader("💰 預算找房助理 (區間搜尋)")
        st.markdown("請設定您的預算條件，系統將根據各區 AI 預測行情進行匹配。")
        
        budget_col1, budget_col2 = st.columns(2)
        my_budget_total = budget_col1.number_input("您的【全屋總價】預算上限 (萬元)", 100, 10000, 1500)
        my_budget_unit = budget_col2.number_input("您的【每坪單價】預算上限 (萬元/坪)", 1.0, 150.0, 45.0) * 10000
        
        # 改為範圍選擇
        st.markdown("---")
        st.subheader("🔍 搜尋條件細分")
        col_r1, col_r2 = st.columns(2)
        age_range = col_r1.slider("期望屋齡範圍 (年)", 0, 60, (5, 30), key="b_age_range")
        area_range = col_r2.slider("期望坪數範圍 (建物總坪數)", 5, 200, (20, 45))
        ratio_range = st.slider("期望公設比範圍 (%)", 0, 50, (25, 35))
        
        h_type_b = st.selectbox("偏好建物類型", ["集合住宅 (大樓/華廈/公寓)", "透天厝"], key="b_type")
        
        # 行政區多選
        all_districts = sorted(market_ctx.keys())
        target_districts = st.multiselect("欲搜尋的行政區 (留空則搜尋全高雄)", all_districts)
        
        if st.button("🔍 執行區間行情掃描"):
            search_list = target_districts if target_districts else all_districts
            results = []
            with st.spinner("各區模擬計算中..."):
                for dist in search_list:
                    # 計算該範圍內最便宜與最貴的組合
                    # 最便宜：最大年資、最小坪數 (總價最低)
                    p_low, _, _ = run_prediction(h_type_b, dist, age_range[1], area_range[0], ratio_range[1])
                    total_low = (p_low * area_range[0]) / 10000
                    
                    # 最貴：最小年資、最大坪數
                    p_high, _, _ = run_prediction(h_type_b, dist, age_range[0], area_range[1], ratio_range[0])
                    total_high = (p_high * area_range[1]) / 10000
                    
                    # 篩選條件：最便宜入手價需低於「總價預算」且單價需低於「單價預算」
                    if total_low <= my_budget_total and p_low <= my_budget_unit:
                        # 潛力指數 (Opportunity Score): MA30 > MA180 代表起漲
                        ctx = market_ctx.get(dist, {})
                        opp_score = "🔥 起漲區" if ctx.get('MA30_Momentum', 0) > ctx.get('MA180_Momentum', 0) else "穩定"
                        
                        results.append({
                            "行政區": dist,
                            "最低入手單價 (萬/坪)": round(p_low/10000, 2),
                            "最低入手總價 (萬)": round(total_low),
                            "最高規格總價 (萬)": round(total_high),
                            "預算狀態": "✅ 完全符合" if total_high <= my_budget_total else "🟡 部分符合",
                            "市場趨勢": opp_score
                        })
            
            if results:
                res_df = pd.DataFrame(results).sort_values("最低入手總價 (萬)")
                st.success(f"找到 {len(results)} 個行政區內有符合您預算區間的機會！")
                st.dataframe(res_df.style.format({
                    "最低入手單價 (萬/坪)": "{:,.2f} 萬",
                    "最低入手總價 (萬)": "{:,.0f} 萬",
                    "最高規格總價 (萬)": "{:,.0f} 萬"
                }), width="stretch")
            else:
                st.warning("⚠️ 沒找到符合的區域，建議調高預算或放寬坪數/屋齡限制。")

# =================================================================
# Page 2: 市場行情地圖
# =================================================================
elif page == "市場行情地圖":
    st.title("🗺️ 高雄房價地理熱度圖")
    
    m_col1, m_col2 = st.columns([1, 4])
    with m_col1:
        st.subheader("🛠️ 顯示設定")
        stat_mode = st.radio("計算指標", ["平均值 (Mean)", "中位數 (Median)", "成交筆數"])
        if "成交筆數" in stat_mode:
            count_scope = st.radio("資料範圍", ["全部", "離群值 (>60萬/坪)"], horizontal=True)
            show_outliers = count_scope == "離群值 (>60萬/坪)"
        else:
            show_outliers = st.toggle("☢️ 僅顯示離群值 (>60萬/坪)", value=False)
        st.info("💡 數值單位統一為「萬元/坪」。")

    # 數據準備（使用含離群值的 lasso v3 中間檔）
    map_df = df_map.copy()
    if show_outliers:
        map_df = map_df[map_df['淨屋單價元坪'] > 600000]
    
    # 計算各區統計 (包含平均、中位、筆數、建物型態佔比)
    def get_district_stats(df):
        stats = df.groupby('鄉鎮市區')['淨屋單價元坪'].agg(['mean', 'median', 'count']).reset_index()
        
        # 建物型態比例
        type_pivot = pd.crosstab(df['鄉鎮市區'], df['建物型態'], normalize='index') * 100
        # 簡化型態名稱以利顯示
        type_pivot.columns = [c.split('(')[0] for c in type_pivot.columns]
        
        stats = stats.merge(type_pivot, left_on='鄉鎮市區', right_index=True)
        return stats

    dist_stats = get_district_stats(map_df)
    dist_stats.columns = ['district', 'mean_price', 'median_price', 'total_count'] + list(dist_stats.columns[4:])
    
    # 載入 GeoJSON 並將數據注入 properties
    with open("app/kaohsiung_districts.json", "r", encoding="utf-8") as f:
        geo_data = json.load(f)
    
    # 建立查找表
    stats_lookup = dist_stats.set_index('district').to_dict('index')
    
    for feature in geo_data['features']:
        name = feature['properties']['名稱'].replace('高雄市', '')
        feature['properties']['名稱'] = name
        if name in stats_lookup:
            s = stats_lookup[name]
            feature['properties']['mean_p'] = f"{s['mean_price']/10000:,.1f} 萬"
            feature['properties']['median_p'] = f"{s['median_price']/10000:,.1f} 萬"
            feature['properties']['count'] = f"{s['total_count']:,.0f}"
            
            # 建立建物比例文字
            type_info = []
            for col in dist_stats.columns[4:]:
                if s[col] > 5: # 只顯示佔比大於 5% 的
                    type_info.append(f"{col}: {s[col]:.1f}%")
            feature['properties']['type_dist'] = " | ".join(type_info)
        else:
            feature['properties']['mean_p'] = "N/A"
            feature['properties']['median_p'] = "N/A"
            feature['properties']['count'] = "0"
            feature['properties']['type_dist'] = "無資料"

    with m_col2:
        m_choropleth = folium.Map(location=[22.65, 120.35], zoom_start=11, tiles="cartodbpositron")
        
        if "平均值" in stat_mode:
            target_val = 'mean_price'
            legend_label = "平均 淨屋單價 (元/坪)"
        elif "中位數" in stat_mode:
            target_val = 'median_price'
            legend_label = "中位數 淨屋單價 (元/坪)"
        else:
            target_val = 'total_count'
            legend_label = "成交筆數 (筆)"
        
        # 面量圖底層
        folium.Choropleth(
            geo_data=geo_data,
            name="choropleth",
            data=dist_stats,
            columns=["district", target_val],
            key_on="feature.properties.名稱",
            fill_color="YlOrRd",
            fill_opacity=0.7,
            line_opacity=0.4,
            legend_name=legend_label,
            highlight=True
        ).add_to(m_choropleth)
        
        # 互動層：超級工具提示
        folium.GeoJson(
            geo_data,
            style_function=lambda x: {'fillColor': '#ffffff00', 'color': '#00000000'},
            tooltip=folium.GeoJsonTooltip(
                fields=['名稱', 'mean_p', 'median_p', 'count', 'type_dist'],
                aliases=['行政區:', '平均單價:', '中位數:', '成交筆數:', '建物分佈:'],
                localize=True,
                sticky=True,
                labels=True,
                style="""
                    background-color: #F0F2F6;
                    border: 2px solid black;
                    border-radius: 3px;
                    box-shadow: 3px;
                """,
                max_width=400,
            )
        ).add_to(m_choropleth)

        st_folium(m_choropleth, width=1100, height=600, key="main_map")
    
    # --- 動態統計圖表 (New Row) ---
    st.markdown("---")
    c_left, c_right = st.columns(2)
    
    with c_left:
        st.subheader("行政區成交量排行 (Top 10)")
        top10_df = dist_stats.nlargest(10, 'total_count')
        fig_bar = px.bar(
            top10_df, x='total_count', y='district', 
            orientation='h',
            text='total_count',
            color='total_count',
            color_continuous_scale='YlOrRd',
            labels={'total_count': '成交筆數', 'district': '行政區'}
        )
        fig_bar.update_layout(showlegend=False, height=400, margin=dict(t=30, b=0, l=0, r=0))
        fig_bar.update_traces(textposition='outside')
        st.plotly_chart(fig_bar, use_container_width=True)

    with c_right:
        st.subheader("物件結構比例分析")
        type_counts = map_df['建物型態'].value_counts().reset_index()
        type_counts.columns = ['型態', '數量']
        fig_pie = px.pie(
            type_counts, values='數量', names='型態',
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.RdBu_r
        )
        fig_pie.update_layout(height=400, margin=dict(t=30, b=0, l=0, r=0))
        st.plotly_chart(fig_pie, use_container_width=True)



# =================================================================
# Page 3: EDA 數據藝廊
# =================================================================
elif page == "EDA 數據藝廊":
    st.title("深入探索數據背後的真相")
    
    eda_tabs = st.tabs(["1. 初步探索", "2. 特徵分佈", "3. 車位拆算", "4. 離群值處理", "5. 市場動態"])
    
    # Section 1: Preliminary Analysis
    with eda_tabs[0]:
        st.subheader("初步了解資料")
        st.write(
            "資料來源為**內政部不動產交易實價登錄**，涵蓋 2019–2026 年高雄市各行政區的住宅交易記錄，"
            "共 **137,244 筆**。原始資料經過清洗與特徵工程後，包含建物面積、屋齡、公設比、"
            "地理位置、交通距離等 51 個欄位。以下為主要數值特徵的統計摘要。"
        )

        # 簡單統計表格
        cont_cols = ['建物移轉總面積坪', '屋齡', '淨屋單價元坪', '公設比', '主建物率', '土地持分率']
        valid_cols = [c for c in cont_cols if c in df_all.columns]
        if valid_cols:
            stats_df = df_all[valid_cols].agg(['max', 'min', 'mean', 'std', 'median']).T
            stats_df.columns = ['最大值', '最小值', '平均數', '標準差', '中位數']
            st.dataframe(stats_df.style.format("{:,.2f}"), use_container_width=True)
        else:
            st.warning("查無連續型變數統計資料")

    # Section 2: Correlations & Distributions
    with eda_tabs[1]:
        st.subheader("相關性與維度分佈")
        st.write(
            "透過相關性分析，我們可以了解各特徵之間的關聯程度，"
            "協助篩選對預測單價最有幫助的特徵。"
            "左圖為連續型變數之間的 Pearson 相關係數，右圖為類別型變數的 Cramér's V 關聯強度。"
        )
        col_corr1, col_corr2 = st.columns(2)
        with col_corr1:
            st.image("visuals/eda/heatmap_continuous.png", caption="連續型變數相關熱力圖")
        with col_corr2:
            st.image("visuals/eda/heatmap_categorical.png", caption="類別型變數相關係數 (Cramer's V)")

        st.markdown("---")
        st.write("#### 類別變數分佈")
        with st.expander("展開查看類別變數統計圖"):
            c1, c2 = st.columns(2)
            c1.image("visuals/eda/bar_鄉鎮市區.png")
            c2.image("visuals/eda/bar_建物型態.png")
            c1.image("visuals/eda/bar_土地分區大類.png")
            c2.image("visuals/eda/bar_建物用途大類.png")
            c1.image("visuals/eda/bar_電梯.png")
            c2.image("visuals/eda/bar_有無管理組織.png")

        st.markdown("---")
        st.write("#### 連續變數分布特性")
        st.image("visuals/eda/group1_district_age.png", caption="各行政區屋齡分佈 (箱型圖)")
        st.image("visuals/eda/group4_type_net_price.png", caption="建物型態與單價關係")

    # Section 3: Parking
    with eda_tabs[2]:
        st.subheader("資料處理：目標值建立")

        # ── 步驟一：問題說明 ──────────────────────────────────────
        st.markdown("#### 步驟一：為何需要車位拆算？")
        st.write(
            "台灣房子通常將**車位包含在總價**中一起交易。"
            "由於我們的目標是計算「**每坪多少錢（淨屋單價）**」，"
            "因此必須先將車位金額從總價中拆除，才能得到真實的房屋單價。"
        )

        st.markdown("---")
        # ── 步驟二：發現零面積車位 ───────────────────────────────
        st.markdown("#### 步驟二：發現零面積車位問題")
        st.write(
            "在處理過程中，我們發現部分資料**有車位但車位面積填寫為 0**。"
            "我們推測這是因為早期建案將車位面積併入公設（共有部分）登記，"
            "導致公設比偏高，而非真的沒有車位。"
        )
        st.image(
            "visuals/eda/historical_public_ratio_trend.png",
            caption="歷年建築公設比趨勢（依建築完成年代，5年一組）：零面積車位組公設比持續高於有面積組，"
                    "反映早期建案將車位面積納入公設的歷史慣例。"
        )

        c_type, c_ratio = st.columns(2)
        c_type.image(
            "visuals/eda/parking_type_distribution.png",
            caption="各建物型態中零面積車位佔比：大樓與華廈的比例遠高於透天厝"
        )
        c_ratio.image(
            "visuals/eda/parking_public_ratio_comparison.png",
            caption="公設比三組對比：零面積車位組（橘）平均公設比顯著高於有面積組（藍）"
        )

        st.markdown("---")
        # ── 步驟三：屋齡是關鍵因素 ──────────────────────────────
        st.markdown("#### 步驟三：屋齡是關鍵因素")
        st.write(
            "進一步分析顯示，**屋齡越高，零面積車位的比例越高**。"
            "這佐證了「以前的建案習慣將車位面積納入公設登記」的推論。"
            "右圖也可以看到，近年交易中零面積車位的比例逐年下降，反映登記規範趨於完整。"
        )
        c3, c4 = st.columns(2)
        c3.image("visuals/eda/5_age_distribution.png",
                 caption="屋齡分布對比：零面積（紅）vs 有面積（藍）車位")
        c4.image("visuals/eda/4_yearly_trend.png",
                 caption="歷年零面積車位佔比（有車位案件中）")
        st.success(
            "**結論**：零面積車位集中於屋齡較高的老舊建物，且公設比顯著偏高，"
            "確認車位面積被早期建案「灌入公設」。\n\n"
            "**處理策略**：對零面積車位，僅扣除車位**價格**，不扣除車位面積（避免雙重扣除）。"
        )

        st.markdown("---")
        # ── 步驟四：Lasso 拆價模型 ──────────────────────────────
        st.markdown("#### 步驟四：Lasso 回歸模型補值與拆除")
        st.write(
            "針對上述問題，我們使用 **Lasso 回歸模型**預測每筆交易的車位價格，"
            "再從總價中扣除，得到淨屋總價，最後除以建物淨面積計算出淨屋單價。\n\n"
            "公式：**淨屋單價 = (總價 − Lasso 預測車位價) ÷ 建物淨面積**"
        )
        st.image(
            "visuals/eda/price_correction_kde.png",
            caption="修正前（含車位單價）vs 修正後（淨屋單價）KDE 對比：拆除車位後峰值左移、右尾縮短，更精確反映純住宅單價"
        )

        st.markdown("---")
        # ── 步驟五：修正後結果 ───────────────────────────────────
        st.markdown("#### 步驟五：修正後的淨屋單價分布")
        st.image(
            "visuals/eda/boxplot_net_price_overall.png",
            caption="清洗後全體淨屋單價箱型圖（已排除離群值 > 60 萬/坪）"
        )

    # Section 4: Outliers
    with eda_tabs[3]:
        st.subheader("離群值處理")
        st.error(
            "完成車位拆算後，發現資料中仍存在**不合理的極端值**，"
            "可能來自實價登錄的登記偏差或特殊交易，需進行篩除以確保模型穩健。"
        )

        st.markdown("#### 單價分布與門檻設定")
        st.image("visuals/eda/1_price_distribution_with_threshold.png",
                 caption="淨屋單價分布：橘線為偽屋篩選門檻（60萬/坪），虛線為99th百分位")

        st.markdown("#### 偽屋（Fake House）篩選")
        st.write(
            "進一步發現部分物件**面積極小卻單價極高**，不符合正常住宅交易行為，"
            "研判為登記錯誤或非典型交易，採用**雙重條件**篩除："
        )
        st.code("篩選條件：建物淨面積 < 5 坪  AND  淨屋單價 > 60 萬/坪", language=None)
        st.image("visuals/eda/2_fake_house_scatter.png",
                 caption="偽屋散點圖：紅色標記為篩除對象（面積<5坪且單價>60萬）")

    
        st.info(
            "共篩除 **27 筆**偽屋（佔全體 0.02%），保留 137,244 筆有效資料。\n\n"
            "清洗後，依**建物型態**分流：集合住宅（大樓/華廈/公寓）與透天厝分別建立獨立模型。"
        )


    # Section 5: Market Trends
    with eda_tabs[4]:
        st.subheader("市場動態：台積電效應專題")
        st.markdown(
            "近年高雄房市受**台積電楠梓設廠**消息影響，北高雄各行政區房價出現明顯分化。"
            "以下以 18 個行政區的**淨屋單價中位數**為基準，比較 2019 年與 2026 年的漲幅，"
            "橘色為台積電周邊區域（楠梓、橋頭、左營、岡山、仁武）。"
        )

        # ── 總覽排名圖 ────────────────────────────────────────
        st.markdown("#### 各行政區漲幅排名（2019→2026）")
        ranking_path = "visuals/eda/district_growth_ranking.png"
        if os.path.exists(ranking_path):
            st.image(
                ranking_path,
                caption="各行政區淨屋單價中位數漲幅排名。橘色 = 台積電周邊；數字為 2019→2026 萬元/坪中位數。"
            )

        st.info(
            "**漲幅解讀**：\n"
            "- 台積電周邊（楠梓、橋頭等）普遍位居漲幅前段，反映產業進駐的直接帶動效應\n"
            "- 部分南高雄傳統核心區（三民、苓雅）漲幅相對溫和，但基期單價仍處高位\n"
            "- 漲幅後段的行政區多為郊區或農業區，交易量稀少導致中位數波動較大"
        )

        # ── 各區年度折線小多圖 ───────────────────────────────
        st.markdown("#### 各行政區逐年走勢（依漲幅排序）")
        st.caption("橘色折線為台積電周邊區域，方便對比受產業利多帶動的區域與其他區域的差異。")
        growth_parts = [
            "district_trends_v5_growth_part1.png",
            "district_trends_v5_growth_part2.png",
            "district_trends_v5_growth_part3.png",
        ]
        for p_img in growth_parts:
            path = f"visuals/eda/{p_img}"
            if os.path.exists(path):
                st.image(path)

        st.success(
            "**觀察重點**：\n"
            "- 楠梓、橋頭等北高雄區域在設廠消息明朗後，房價成長斜率明顯陡峭化\n"
            "- 同期左營、岡山等鄰近區域也出現顯著漲幅，顯示外溢效應\n"
            "- 傳統核心區（三民、苓雅）漲幅相對溫和，但基期單價仍屬高位"
        )

# =================================================================
# Page 4: 技術說明
# =================================================================
else:
    st.title("技術與模型說明 ")
    
    tech_tabs = st.tabs(["數據清洗流程", "核心特徵工程", "機器學習建模", "模型可解釋性"])
    
    # Tab 1: Cleaning Flow
    with tech_tabs[0]:
        st.subheader("數據清洗與預處理流水線")
        st.write("從原始實價登錄 CSV 到入模資料集的轉化過程：")
        
        flow_cols = st.columns(4)
        steps = [
            (" Step 1: 資料收集", "資料來自內政部:::不動產交易實價查詢服務網(https://lvr.land.moi.gov.tw/)，合併 2020-2026 全高雄原始季檔資料。"),
            (" Step 2: 資料清理與特徵工程", "交易標的篩選、建物型態篩選、土地分區整併、民國年轉換、面積轉坪、比率計算、異常值過濾"),
            (" Step 3: 目標變數衍算與車位拆價", "利用 Lasso 回歸進行車位價格剝離，以求取最真實的「淨屋單價」。"),
            (" Step 4: 地理擴充", "API 獲取座標，計算房子距離最近的關鍵地標(包含捷運站、輕軌站、火車站、高鐵站、台積電楠梓廠)距離並進行 PCA 降維。")
        ]
        for i, (title, desc) in enumerate(steps):
            with flow_cols[i % 4]:
                st.info(f"{title}\n\n{desc}")
                
        st.write("---")
        st.markdown("### 資料量流失漏斗")
        funnel_data = {
            "階段": [
                "原始實價登錄資料",
                "篩選高雄住宅交易",
                "Lasso 車位拆算 + 地理擴充",
                "偽屋篩除（面積<5坪且單價>60萬）",
                "集合住宅（入模）",
                "透天厝（入模）",
            ],
            "筆數": ["~200,000+", "168,843", "137,294", "137,244", "99,989", "37,255"],
            "說明": [
                "內政部實價登錄原始季檔合併",
                "限定住家用途、高雄行政區、2019年後",
                "Lasso 拆算車位價 → 地理 API → PCA",
                "雙重條件篩除極端異常點",
                "大樓 / 華廈 / 公寓 / 套房",
                "獨棟 / 連棟 / 農舍等透天型態",
            ]
        }
        st.table(pd.DataFrame(funnel_data))

        st.write("---")
        st.markdown("""
        ### 🛡️ 資料品質防護網 (Quality Guard)
        - **偽屋過濾**：排除 `面積 < 5坪` 且 `單價 > 60萬` 的異常登記點。
        - **時間聚焦**：僅採納 2019 年以後的資料，確保市場趨勢的代表性。
        - **分流建模**：將「集合住宅」與「透天厝」分流，因其價格驅動因素完全不同。
        """)

    # Tab 2: Feature Engineering
    with tech_tabs[1]:
        st.subheader("核心特徵工程技術")

        st.markdown("#### 1. 地理空間降維（PCA）")
        st.write(
            "原始地理特徵為 5 個 POI 距離（捷運、台鐵、輕軌、高鐵、台積電楠梓廠）。"
            "由於各交通設施距離高度共線（離捷運近通常也離台鐵近），"
            "直接投入模型會引入多重共線性。"
            "透過 **PCA 主成分分析**將 5 維壓縮為 2 個主成分，保留 **90.1%** 的原始資訊："
        )
        pca_col1, pca_col2 = st.columns(2)
        pca_col1.metric("PC1 解釋變異", "76.7%", "整體大眾運輸依賴度")
        pca_col2.metric("PC2 解釋變異", "13.4%", "北高雄產業樞紐軸度（台積電 vs 輕軌）")

        st.markdown("---")
        st.markdown("#### 2. 在地化市場動能（Momentum）")
        st.write(
            "引入行政區層級的**移動平均行情指標**，讓模型能感知「目前這個區域的市場溫度」。"
        )
        st.markdown("""
        | 特徵名稱 | 說明 |
        |----------|------|
        | `District_MA180_Past` | 行政區過去 180 日中位成交單價（基準行情） |
        | `MA30_Momentum` | 短期 30 日行情動能（景氣加速） |
        | `MA90_Momentum` | 中期 90 日行情動能 |
        | `MA180_Momentum` | 長期 180 日行情動能（趨勢基準） |
        """)
        st.info(
            "**為何重要**：加入動能特徵後，集合住宅模型的 MAPE 從 ~15% 改善至 11.9%，"
            "說明「目前行情」是預測單價最關鍵的背景資訊之一。"
        )

        st.markdown("---")
        st.markdown("#### 3. 衍算指標（Derived Features）")
        st.write(
            "除直接欄位外，本專案額外計算多項**比率型特徵**，透過非線性轉換放大特徵間的差異性，"
            "提升模型對住宅結構品質的感知能力。"
        )

        # 注入 CSS：讓 KaTeX 分數容器有足夠的上下空間，避免中文字被裁切
        st.markdown(
            "<style>.katex-display{padding:10px 0 6px 0 !important;"
            "overflow:visible !important;}</style>",
            unsafe_allow_html=True,
        )

        def _latex_block(formula: str):
            """使用 st.markdown $$ 語法讓 Streamlit 內建 KaTeX 渲染，搭配上方 CSS 防止裁切。"""
            st.markdown(f"$${formula}$$")

        st.markdown("**公設比（Common Area Ratio）**")
        _latex_block(r"\text{公設比} = \frac{\text{建物移轉總面積} - \text{主建物面積} - \text{附屬建物面積}}{\text{建物移轉總面積}} \times 100\%")
        st.caption("反映公共設施佔比；越高代表實際居住面積越小。")

        st.markdown("**公設比主建物比（入模特徵）**")
        _latex_block(r"\text{公設比\_主建物比} = \frac{\text{公設比}}{1 - \text{公設比}}")
        st.caption("對公設比做 Odds 轉換，放大高公設比端的差異，對模型更有區辨力。")

        st.markdown("**土地持分率**")
        _latex_block(r"\text{土地持分率} = \frac{\text{土地移轉總面積}}{\text{建物移轉總面積}}")
        st.caption("集合住宅通常 0.02–0.30；透天厝接近 1.0（地主自建）。此特徵是透天厝定價最重要的驅動力。")

        st.markdown("**屋齡**")
        _latex_block(r"\text{屋齡} = \text{交易年（西元）} - \text{建築完成年（西元）}")
        st.caption("民國年於前處理階段轉換為西元年。屋齡每增加 1 年，單價平均折舊顯著。")
        st.markdown("---")
        st.markdown("#### 4. 特徵清單總覽（21 個入模特徵）")
        feat_table = pd.DataFrame({
            "類別": ["地理空間", "地理空間", "市場動能", "市場動能", "市場動能", "市場動能",
                     "建物結構", "建物結構", "建物結構", "建物結構", "建物結構", "建物結構",
                     "地標距離", "地標距離", "地標距離", "地標距離", "地標距離", "地標距離",
                     "類別型", "類別型", "類別型"],
            "特徵名稱": [
                "PC1_整體大眾運輸依賴度", "PC2_北高雄產業樞紐軸度",
                "District_MA180_Past", "MA30_Momentum", "MA90_Momentum", "MA180_Momentum",
                "建物移轉總面積坪", "主建物面積", "屋齡", "公設比_主建物比", "土地持分率", "有無管理組織",
                "最小MRT距離_公尺", "最小LRT距離_公尺", "最小TRA距離_公尺", "最小HSR距離_公尺",
                "最小TSMC距離_公尺", "最小大型公園量體距離_公尺",
                "鄉鎮市區", "建物型態", "Street",
            ],
            "說明": [
                "捷運/台鐵/輕軌/高鐵/台積電距離 PCA 第1主成分",
                "台積電 vs 輕軌交通 PCA 第2主成分",
                "行政區過去180日中位成交單價", "短期30日行情動能", "中期90日行情動能", "長期180日行情動能",
                "建物總坪數（含公設）", "主建物坪數（訓練時為0，詳見說明）", "建築完成至成交年份差",
                "公設比 Odds 轉換值", "土地面積 ÷ 建物面積", "是否有管理委員會",
                "最近捷運站距離（公尺）", "最近輕軌站距離", "最近台鐵站距離",
                "最近高鐵站距離", "最近台積電楠梓廠距離", "最近大型公園距離",
                "鄉鎮市區名稱（31 類）", "建物型態（大樓/華廈/公寓等）", "街道名稱（高基數分類）",
            ]
        })
        st.dataframe(feat_table, use_container_width=True, hide_index=True)

    # Tab 3: Model Battle
    with tech_tabs[2]:
        st.subheader("機器學習建模與競賽（Model Battle）")
        st.write(
            "採用**分流建模**策略：集合住宅（大樓/華廈/公寓/套房）與透天厝的定價邏輯根本不同，"
            "透天的土地價值占比遠高於建物本身，若混合訓練會導致模型在兩種截然不同的價格結構間妥協。"
            "分別針對兩類建物，進行 GBDT 三大演算法的對抗測試，"
            "以**時間序列分割（最後 20% 作驗證集）**模擬「預測未來交易」的實際應用場景。"
        )
        st.markdown("---")
        st.markdown("#### 使用的三大演算法")
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            with st.expander("CatBoost", expanded=True):
                st.markdown(
                    "**類型**：梯度提升決策樹（GBDT）\n\n"
                    "**核心優勢**：\n"
                    "- 原生支援類別型特徵，無需 Label Encoding\n"
                    "- Ordered Boosting 機制避免 target leakage\n"
                    "- 對高基數分類欄位（街道、行政區）特別穩健\n\n"
                    "**本案應用**：集合住宅模型（99,989 筆）"
                )
        with mc2:
            with st.expander("LightGBM", expanded=True):
                st.markdown(
                    "**類型**：梯度提升決策樹（GBDT）\n\n"
                    "**核心優勢**：\n"
                    "- Leaf-wise 分裂策略，每次分裂選損失最大的葉節點\n"
                    "- 訓練速度快、記憶體效率高\n"
                    "- 支援 category dtype，直接處理類別特徵\n\n"
                    "**本案應用**：透天厝模型（37,255 筆）"
                )
        with mc3:
            with st.expander("XGBoost", expanded=True):
                st.markdown(
                    "**類型**：梯度提升決策樹（GBDT）\n\n"
                    "**核心優勢**：\n"
                    "- Level-wise 分裂，正則化（L1/L2）穩健\n"
                    "- 業界廣泛使用的成熟基準模型\n"
                    "- 超參數調整文獻豐富\n\n"
                    "**本案應用**：兩組模型競賽的對照基準"
                )

        st.markdown("---")
        col_m1, col_m2 = st.columns(2)

        with col_m1:
            st.markdown("#### 集合住宅模型組（99,989 筆）")
            apt_results = pd.DataFrame({
                "模型": ["CatBoost", "LightGBM", "XGBoost"],
                "MAPE": ["11.90%", "12.11%", "12.83%"],
                "MAE（元/坪）": ["34,893", "35,843", "39,370"],
                "R²": ["0.6252", "0.6068", "0.5209"],
                "評選": ["選擇此模型", "", ""]
            })
            st.table(apt_results)
            st.caption(
                "決選原因：CatBoost 原生支援類別型特徵（行政區、街道），"
                "對高基數分類特徵的處理優於其他兩者，MAPE 最低且穩定。"
            )

        with col_m2:
            st.markdown("#### 透天厝模型組（37,255 筆）")
            house_results = pd.DataFrame({
                "模型": ["LightGBM", "CatBoost", "XGBoost"],
                "MAPE": ["14.92%", "14.26%", "16.41%"],
                "MAE（元/坪）": ["55,902", "58,181", "69,151"],
                "R²": ["0.5132", "0.2399", "0.0370"],
                "評選": ["選擇此模型", "", ""]
            })
            st.table(house_results)
            st.caption(
                "決選原因：LightGBM 的 R² 達 0.51，顯著優於 CatBoost（0.24）。"
                "透天受土地面積影響大，LightGBM 的葉節點分裂策略對此類高變異數據的魯棒性較佳。"
            )

        st.markdown("---")
        st.info(
            "**R² 偏低（0.51～0.63）的說明**：台灣房市受人脈、急售、繼承等主觀因素影響，"
            "非線性程度極高，即便業界主流模型的 R² 也普遍落在 0.5～0.7 區間，屬合理範圍。"
            "MAPE 11～15% 代表平均每坪預測誤差約 3～5 萬元，適合作為輔助決策參考，"
            "不建議直接作為報價依據。"
        )

    # Tab 4: XAI
    with tech_tabs[3]:
        st.subheader("模型可解釋性（SHAP）")
        st.write(
            "採用 **SHAP（SHapley Additive exPlanations）** 分解模型的預測邏輯，"
            "讓每一筆預測結果都可以追溯到「哪個特徵推高/壓低了這個估價」。"
            "下方的 Beeswarm 圖中，每個點代表一筆資料：**橫軸**為 SHAP 值（正 = 推高價格，負 = 壓低），"
            "**顏色**代表該特徵的原始數值（紅 = 高值，藍 = 低值）。"
        )

        st.markdown("---")
        st.markdown("#### 集合住宅模型（CatBoost）")
        c_apt1, c_apt2 = st.columns(2)
        with c_apt1:
            st.markdown("**關鍵影響因子**")
            st.markdown("""
            1. **行政區 / 街道**：鼓山、苓雅、左營等高價區顯著推高預測值
            2. **行政區均價（District_MA180）**：當前區域行情是最強的行情錨定特徵
            3. **屋齡**：折舊效應明顯，新屋比舊屋每坪可多數萬元
            4. **PC1（大眾運輸依賴度）**：離捷運越近，單價越高
            5. **建物總面積**：大坪數物件的邊際單價遞減
            """)
        with c_apt2:
            if os.path.exists("visuals/shaps/shap_apartment_bar.png"):
                st.image("visuals/shaps/shap_apartment_bar.png",
                         caption="特徵重要性排序（平均 |SHAP 值|）")
        if os.path.exists("visuals/shaps/shap_apartment_summary.png"):
            st.image("visuals/shaps/shap_apartment_summary.png",
                     caption="CatBoost 集合住宅 — SHAP Beeswarm（每點為一筆交易，紅=特徵值高，藍=特徵值低）")

        st.markdown("---")
        st.markdown("#### 透天厝模型（LightGBM）")
        c_house1, c_house2 = st.columns(2)
        with c_house1:
            st.markdown("**關鍵影響因子**")
            st.markdown("""
            1. **土地持分率**：透天厝的「土地 >> 建物」特性，土地佔比是最核心的價格驅動力
            2. **行政區 / 街道**：精華地段溢價效應同樣顯著
            3. **行政區均價（District_MA180）**：行情參考同樣重要
            4. **建物總面積**：與集合住宅相比，面積的邊際效應方向相同但幅度不同
            5. **PC1（大眾運輸依賴度）**：交通便利性對透天有門檻溢價效應
            """)
        with c_house2:
            if os.path.exists("visuals/shaps/shap_house_bar.png"):
                st.image("visuals/shaps/shap_house_bar.png",
                         caption="特徵重要性排序（平均 |SHAP 值|）")
        if os.path.exists("visuals/shaps/shap_house_summary.png"):
            st.image("visuals/shaps/shap_house_summary.png",
                     caption="LightGBM 透天厝 — SHAP Beeswarm")
