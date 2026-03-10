import pandas as pd
from geopy.geocoders import ArcGIS
import time

poi_list = [
    # 紅線 R Line
    "高雄捷運岡山高醫站", "高雄捷運岡山車站",
    "高雄捷運南岡山站", "高雄捷運橋頭火車站", "高雄捷運橋頭糖廠站", "高雄捷運青埔站",
    "高雄捷運都會公園站", "高雄捷運後勁站", "高雄捷運楠梓加工區站", "高雄捷運世運站",
    "高雄捷運左營站", "高雄捷運生態園區站", "高雄捷運巨蛋站", "高雄捷運凹子底站",
    "高雄捷運後驛站", "高雄捷運高雄車站", "高雄捷運美麗島站", "高雄捷運中央公園站",
    "高雄捷運三多商圈站", "高雄捷運獅甲站", "高雄捷運凱旋站", "高雄捷運前鎮高中站",
    "高雄捷運草衙站", "高雄捷運高雄國際機場站", "高雄捷運小港站",
    # 橘線 O Line
    "高雄捷運哈瑪星站", "高雄捷運西子灣站", "高雄捷運鹽埕埔站", "高雄捷運市議會站",
    "高雄捷運信義國小站", "高雄捷運文化中心站", "高雄捷運五塊厝站", "高雄捷運技擊館站", 
    "高雄捷運衛武營站", "高雄捷運鳳山西站", "高雄捷運鳳山站", "高雄捷運大東站", 
    "高雄捷運鳳山國中站", "高雄捷運大寮站",
    # 高鐵 & 主要台鐵
    "高鐵左營站", "台鐵高雄車站", "台鐵鳳山車站", "台鐵美術館車站", "台鐵科工館車站",
    "台鐵正義車站", "台鐵三塊厝車站", "台鐵民族車站", "台鐵鼓山車站", "台鐵左營(舊城)車站",
    "台鐵新左營車站", "台鐵楠梓車站", "台鐵橋頭車站", "台鐵岡山車站", "台鐵大湖車站", "台鐵路竹車站",
    "台鐵後莊車站", "台鐵九曲堂車站",
    # 產業重點
    "台積電楠梓產業園區"
]

geolocator = ArcGIS(user_agent="kaohsiung_poi_fetcher")
data = []

print("Start mapping POIs...")
for poi in poi_list:
    try:
        location = geolocator.geocode(poi, timeout=10)
        if location:
            data.append({"POI": poi, "lat": location.latitude, "lon": location.longitude})
            print(f"[Success] {poi}")
        else:
            print(f"[Failed] {poi}")
    except Exception as e:
        print(f"[Error] {poi}: {e}")
    time.sleep(1)

df_poi = pd.DataFrame(data)
df_poi.to_csv("kaohsiung_poi_coords.csv", index=False)
print(f"Finished collecting {len(df_poi)} POIs.")
