import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import time
import json
import os

def clean_address(addr):
    """
    處理實價登錄去識別化地址，如 '高雄市鼓山區明誠四路1~30號' -> '高雄市鼓山區明誠四路15號'
    """
    if not isinstance(addr, str): return None
    if '~' in addr:
        import re
        parts = re.split(r'~|及', addr)
        # 尋找數字範圍
        nums = [int(s) for s in re.findall(r'\d+', addr)]
        if len(nums) >= 2:
            mid = (nums[0] + nums[1]) // 2
            # 將 1~30 替換成中間值
            addr = re.sub(r'\d+[~及]\d+', str(mid), addr)
    return addr

def geocode_pipeline(sample_size=100):
    cache_file = 'address_coords_cache.json'
    
    # 載入快取
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache = json.load(f)
    else:
        cache = {}

    # 讀取資料
    df = pd.read_csv('main_unbundled_lasso_v3.csv')
    addr_col = [c for c in df.columns if '土地位置' in c][0]
    
    # 取不重複地址樣品
    unique_addrs = df[addr_col].unique()
    sample_addrs = unique_addrs[:sample_size]
    
    geolocator = Nominatim(user_agent="kaohsiung_housing_analysis_v1")
    
    results = []
    new_count = 0
    
    print(f"開始處理 {sample_size} 筆地址...")
    
    for i, raw_addr in enumerate(sample_addrs):
        if raw_addr in cache:
            results.append(cache[raw_addr])
            continue
            
        clean_addr = clean_address(raw_addr)
        try:
            # 加上高雄提高準確度
            query = clean_addr if "高雄" in clean_addr else f"高雄市{clean_addr}"
            location = geolocator.geocode(query, timeout=10)
            
            if location:
                coord = {"lat": location.latitude, "lon": location.longitude}
                cache[raw_addr] = coord
                results.append(coord)
                new_count += 1
            else:
                cache[raw_addr] = None
                results.append(None)
                
        except Exception as e:
            print(f"處理 {raw_addr} 時發生錯誤: {e}")
            results.append(None)
            
        # 禮貌爬蟲：間隔 1 秒防止被鎖
        time.sleep(1)
        if (i+1) % 10 == 0:
            print(f"已完成 {i+1}/{sample_size}...")

    # 更新快取檔案
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=4)
        
    print(f"\n測試結束。")
    print(f"新抓取座標: {new_count} 筆")
    print(f"抓取成功率: {len([r for r in results if r]) / sample_size * 100:.1f}%")
    print(f"快取檔案已更新: {cache_file}")

if __name__ == "__main__":
    geocode_pipeline(100)
