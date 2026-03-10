import requests
import json
import os

url = "https://raw.githubusercontent.com/titaneric/Taiwan-GeoJSON/master/%E5%8F%B0%E7%81%A3%E9%84%89%E9%8E%AE.json"
kaohsiung_geojson_path = "kaohsiung_districts.json"

print("正在重新下載地圖資料...")
r = requests.get(url)
all_data = r.json()

# 檢查一下前幾筆資料的 properties 長怎樣
print("屬性範例 (第一筆):", all_data['features'][0]['properties'])

# 嘗試找出高雄市。注意：有些資料會用 '高雄市'，有些可能是 'Kaohsiung City'
# 我們搜尋所有 COUNTYNAME 或 TOWNNAME 欄位
target_county = "高雄市"
kaohsiung_features = []

for f in all_data['features']:
    props = f['properties']
    # 檢查所有可能的縣市欄位
    if any(target_county in str(v) for v in props.values()):
        kaohsiung_features.append(f)

print(f"找到 {len(kaohsiung_features)} 個屬於高雄的行政區。")

if len(kaohsiung_features) > 0:
    kaohsiung_geojson = {
        "type": "FeatureCollection",
        "features": kaohsiung_features
    }
    with open(kaohsiung_geojson_path, 'w', encoding='utf-8') as f:
        json.dump(kaohsiung_geojson, f, ensure_ascii=False)
    print("高雄地圖資料已成功儲存。")
else:
    print("錯誤：找不到任何高雄市的資料！")
