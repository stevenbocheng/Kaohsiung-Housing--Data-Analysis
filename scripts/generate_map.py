import pandas as pd
import folium
import json
import requests
import os

# 1. 下載全台鄉鎮 GeoJSON (此檔案較大，約 47MB)
url = "https://raw.githubusercontent.com/titaneric/Taiwan-GeoJSON/master/%E5%8F%B0%E7%81%A3%E9%84%89%E9%8E%AE.json"
geojson_path = "taiwan_towns.json"
kaohsiung_geojson_path = "kaohsiung_districts.json"

if not os.path.exists(kaohsiung_geojson_path):
    print("正在下載全台地圖資料...")
    r = requests.get(url)
    all_data = r.json()
    
    # 2. 過濾出高雄市的行政區
    # 在此 GeoJSON 中，縣市名稱可能在 'COUNTYNAME' 欄位
    kaohsiung_features = [
        f for f in all_data['features'] 
        if f['properties'].get('COUNTYNAME') == '高雄市'
    ]
    
    kaohsiung_geojson = {
        "type": "FeatureCollection",
        "features": kaohsiung_features
    }
    
    with open(kaohsiung_geojson_path, 'w', encoding='utf-8') as f:
        json.dump(kaohsiung_geojson, f, ensure_ascii=False)
    print(f"高雄地圖資料已過濾並儲存至 {kaohsiung_geojson_path}")
else:
    print(f"已存在高雄地圖資料: {kaohsiung_geojson_path}")

# 3. 讀取房價資料並計算各區中位數
df = pd.read_csv('main.csv')
# 只取「土地1建物1車位0」的基本型態來比較房價可能更精準
# 但 user 的需求是「區點上價錢」，我們用所有資料的中位數
district_stats = df.groupby('鄉鎮市區')['單價元坪'].median().reset_index()

# 4. 建立地圖
# 高雄中心點約在 22.7, 120.4
m = folium.Map(location=[22.7, 120.4], zoom_start=10)

# 5. 加入 Choropleth 圖層
with open(kaohsiung_geojson_path, 'r', encoding='utf-8') as f:
    geo_data = json.load(f)

# 偵錯：確認欄位名稱
# 根據 debug 結果，欄位名稱為 '名稱'
print("GeoJSON 屬性預覽:", geo_data['features'][0]['properties'].keys())
print("第一個行政區名稱:", geo_data['features'][0]['properties'].get('名稱'))

# 建立一個 dictionary 方便 tooltip 快速Lookup
district_price_dict = district_stats.set_index('鄉鎮市區')['單價元坪'].to_dict()

# 6. 在 GeoJSON 中注入價格資訊
for feature in geo_data['features']:
    town = feature['properties'].get('名稱', '')
    price = district_price_dict.get(town, "無資料")
    if isinstance(price, (int, float)):
        price_str = f"{price:,.0f} 元/坪"
    else:
        # 有些情況，GeoJSON 縣市名稱格式可能不同，嘗試模糊匹配或是去除空白
        price_str = "無資料"
    
    feature['properties']['price_info'] = f"{town}: {price_str}"

folium.Choropleth(
    geo_data=geo_data,
    name='choropleth',
    data=district_stats,
    columns=['鄉鎮市區', '單價元坪'],
    key_on='feature.properties.名稱', 
    fill_color='YlOrRd',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='房價中位數 (元/坪)',
    highlight=True
).add_to(m)

# 加入浮動標籤圖層
folium.GeoJson(
    geo_data,
    style_function=lambda x: {'fillColor': 'transparent', 'color':'black', 'weight': 0.5},
    highlight_function=lambda x: {'fillColor': '#000000', 'fillOpacity': 0.3},
    tooltip=folium.GeoJsonTooltip(
        fields=['price_info'],
        aliases=['價格資訊:'],
        localize=True,
        sticky=False,
        labels=True,
        style="""
            background-color: #F0EFEF;
            border: 2px solid black;
            border-radius: 3px;
            box-shadow: 3px;
        """,
        max_width=800,
    )
).add_to(m)

# 7. 儲存地圖
m.save('kaohsiung_price_map.html')
print("地圖已成功生成：kaohsiung_price_map.html")
