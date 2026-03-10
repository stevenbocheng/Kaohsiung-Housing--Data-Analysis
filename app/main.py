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

# 注意：需確保路徑正確或檔案已移動
try:
    models, features, coords, market_ctx = load_assets()
    df_all = load_data()
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
        
        input_dict = {
            '交易年': 2026, '交易月': 3, '屋齡': age, '建物移轉總面積坪': area,
            '主建物面積': area * (1 - p_ratio/100),
            '公設比_主建物比': p_ratio / (100 - p_ratio) if p_ratio < 100 else 0,
            '土地持分率': 0.15 if target_key=="apt" else 1.0,
            '有無管理組織': '有',
            'Street': '不明',
            'PC1_整體大眾運輸依賴度': ctx.get('PC1', 0.0), 
            'PC2_北高雄產業樞紐軸度': ctx.get('PC2', 0.0),
            'District_MA180_Past': ctx.get('District_MA180_Past', 250000), 
            'MA30_Momentum': ctx.get('MA30_Momentum', 250000), 
            'MA90_Momentum': ctx.get('MA90_Momentum', 250000), 
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
        for c in ['鄉鎮市區', '建物用途大類', '主要建材', '土地分區大類', '建物型態', '有無管理組織', 'Street']:
            if c in X_df.columns: X_df[c] = X_df[c].astype('category')
        
        pred_log = model.predict(X_df)[0]
        return np.expm1(pred_log), model, X_df

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
                
                # 計算誤差區間 (以 12% 為例)
                lower_p, upper_p = pred_val * 0.88, pred_val * 1.12
                
                #st.success("✅ 估價運算完成")
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.metric("AI 預測單價", f"{pred_val/10000:,.2f} 萬元/坪", delta=None)
                    st.markdown(f"###### 💡 市場合理區間: {lower_p/10000:,.2f} ~ {upper_p/10000:,.2f} 萬元/坪")
                with res_col2:
                    st.metric("預估成交總價", f"{(pred_val * area_p / 10000):,.1f} 萬元")
                
                with st.expander("🔍 查看預算拆解 (SHAP)"):
                    explainer = shap.TreeExplainer(model)
                    shap_v = explainer(X_df)
                    
                    # 顯示 Base Value (預期平均值)
                    base_val = np.expm1(explainer.expected_value)
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
        stat_mode = st.radio("計算指標", ["平均值 (Mean)", "中位數 (Median)", "豪宅成交筆數 (筆)"])
        show_outliers = st.toggle("☢️ 僅顯示離群值 (>60萬/坪)", value=False or "豪宅" in stat_mode)
        st.info("💡 數值單位統一為「萬元/坪」。")

    # 數據準備
    map_df = df_all.copy()
    if show_outliers or "豪宅" in stat_mode:
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
            legend_label = "豪宅成交筆數 (筆)"
        
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
        st.markdown("**資料來源**：內政部實價登錄 (2019 - 2026)")
        
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
        st.write("####連續變數分布特性")
        st.image("visuals/eda/group1_district_age.png", caption="各行政區屋齡分佈 (箱型圖)")
        st.image("visuals/eda/group4_type_net_price.png", caption="建物型態與單價關係")

    # Section 3: Parking
    with eda_tabs[2]:
        st.subheader("目標值處理：車位拆算深度分析")
        st.info("**核心挑戰**：台灣房價登錄通常包含車位，導致「單價」計算失真。我們的目標是透過模型拆算，還原真實的「淨屋單價」。")
        
        st.markdown("#### 零面積車位現象研究")
        st.write("我們發現早期建物常有「有車位但登記面積為 0」的情況，這在模型中被標註為 `is_zero_area` 特徵。")
        c3, c4 = st.columns(2)
        c3.image("visuals/eda/6_age_public_ratio_correlation.png", caption="屋齡與公設比關係")
        c4.image("visuals/eda/trend_yearly_net_price.png", caption="歷年單價趨勢")
        
        st.image("visuals/eda/historical_public_ratio_trend.png", caption="歷史公設比變化")
        st.success("**分析結論**：零面積車位高度集中於老舊建物，且這些物件的公設比顯著較高（圖中紅藍對比），證實車位面積被併入公設。")
        
        st.markdown("#### 離群值與資料清洗報告")
        c_p1, c_p2 = st.columns(2)
        with c_p1:
            st.image("visuals/eda/price_correction_kde.png", caption="修正前後分佈對比")
        with c_p2:
            st.image("visuals/eda/trend_yearly_net_price.png", caption="修正後單價趨勢")
        

    # Section 4: Outliers
    with eda_tabs[3]:
        st.subheader("離群值處理與偽屋定義")
        st.error("房價資料中存在不合理的極端值 (如實價登錄報錯或特殊交易)，需進行過濾以確保模型穩健。")
        st.image("visuals/eda/price_correction_kde.png", caption="單價分布與修正前對比")
        
        st.write("我們篩選了疑似「假住宅」或「極端行情」的物件。")
        st.image("visuals/reports/kaohsiung_outlier_clean_report.png", caption="離群值分析結果")
        st.write("這類資料多為登記偏差，剔除後能讓模型更專注於大眾市場規律。隨後我們將資料按「建物型態」分流，分別建立大樓與透天模型。")

    # Section 5: Market Trends
    with eda_tabs[4]:
        st.subheader("市場行情動能 (台積電效應專題)")
        st.markdown("近期高雄房價受產業入駐 (如楠梓台積電) 影響顯著，房價漲幅與區域熱度呈現強烈正相關。")
        
        growth_parts = [
            "district_faceted_trends_part1.png", "district_faceted_trends_part2.png", 
            "district_faceted_trends_part3.png", "district_faceted_trends_part4.png"
        ]
        for p_img in growth_parts:
            st.image(f"visuals/eda/{p_img}")
        
        st.success("觀察發現：楠梓及周邊區域在設廠消息確認後，房價成長斜率明顯陡峭化，且成交價位帶有集體上移的趨勢。")

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
                st.info(f"**{title}**\n\n{desc}")
                
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
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            #### 1. 空間降維 (PCA)
            將 5 種交通距離特徵 (MRT, TRA, LRT, HSR, TSMC) 降維。
            - **PC1 (76.7%)**: 整體大眾運輸依賴度。
            - **PC2 (13.4%)**: 北高雄產業樞紐軸度 (台積電近 vs 輕軌近)。
            """)
        with c2:
            st.markdown("""
            #### 💹 2. 在地化市場動能 (Momentum)
            - **District_MA180**: 計算該行政區過去半年同型態物件的平均成交價。
            - **作用**: 讓模型具備「行情感知」能力。
            """)
        # st.image("visuals/shaps/apartment_catboost_shap_summary.png", caption="集合住宅模型特徵影響力 (SHAP Summary)")
        st.info("💡 SHAP 全局解釋圖表目前產出中，請參考下方各特徵說明。")

    # Tab 3: Model Battle
    with tech_tabs[2]:
        st.subheader("機器學習建模與競賽 (Model Battle)")
        st.write("我們針對兩類建物分別進行了三大 GBDT 演算法的對抗實驗：")
        
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            st.write("#### 集合住宅模型組")
            apt_results = pd.DataFrame({
                "模型": ["CatBoost", "LightGBM", "XGBoost"],
                "MAPE (誤差)": ["11.90%", "12.11%", "12.83%"],
                "MAE (坪差)": ["34,893", "35,843", "39,370"],
                "R² (解釋力)": ["0.6252", "0.6068", "0.5209"],
                "評選": ["決選模型", "", ""]
            })
            st.table(apt_results)
            st.caption("決選原因：CatBoost 擅長處理類別型特徵（行政區），表現最優且穩定。")

        with col_m2:
            st.write("#### 透天厝模型組")
            house_results = pd.DataFrame({
                "模型": ["LightGBM", "CatBoost", "XGBoost"],
                "MAPE (誤差)": ["14.92%", "14.26%", "16.41%"],
                "MAE (坪差)": ["55,902", "58,181", "69,151"],
                "R² (解釋力)": ["0.5132", "0.2399", "0.0370"],
                "評選": ["決選模型", "", ""]
            })
            st.table(house_results)
            st.caption("決選原因：LightGBM 在透天極端值的魯棒性較佳，R² 表現顯著優於其他方案。")

    # Tab 4: XAI
    with tech_tabs[3]:
        st.subheader("模型可解釋性")
        st.write("我們拒絕黑盒子。透過 SHAP 值的拆解，我們可以看見模型是如何「思考」價格的：")
        
        c3, c4 = st.columns(2)
        with c3:
            st.markdown("**Apartment - 關鍵影響因子**")
            st.write("1. **行政區/地帶**: 美術館、農16 等高價位區。")
            st.write("2. **區域動能**: 最新行情對於新成交有極強拉動。")
            st.write("3. **屋齡**: 折舊效應明顯。")
        with c4:
            st.markdown("**House - 關鍵影響因子**")
            st.write("1. **土地持分率**: 土地價值是透天價格的本體。")
            st.write("2. **建物總面積**: 總坪數與單價呈現顯著負相關(邊際效應)。")
            st.write("3. **大眾運輸依賴度**: 交通便利性對透天仍有門檻效應。")
            
        st.divider()
        col_s1, col_s2 = st.columns(2)
        # col_s1.image("visuals/shaps/apartment_catboost_shap_summary.png", caption="Apartment 全局解釋")
        # col_s2.image("visuals/shaps/house_lgbm_shap_summary.png", caption="House 全局解釋")
        col_s1.info("集合住宅模型解釋圖待更新")
        col_s2.info("透天厝模型解釋圖待更新")
