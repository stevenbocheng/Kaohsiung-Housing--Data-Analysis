import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
import time
import json
import os
import re

def normalize_address(addr):
    if not isinstance(addr, str): return None
    
    # 轉半形
    addr = addr.translate(str.maketrans('０１２３４５６７８９', '0123456789'))
    
    # 處理範圍
    if '~' in addr or '及' in addr:
        nums = [int(s) for s in re.findall(r'\d+', addr)]
        if len(nums) >= 2:
            mid = (nums[0] + nums[1]) // 2
            addr = re.sub(r'\d+[~及]\d+', str(mid), addr)
            
    # 移除樓層
    addr = re.sub(r'\d+樓.*', '', addr)
    addr = re.sub(r'[一二三四五六七八九十百]+樓.*', '', addr)
    
    # 標準化號碼後的括號內容 (如：(11樓))
    addr = re.sub(r'\(.*\)', '', addr)
    
    # 只取到 "號" 為止
    match = re.search(r'(.*?\d+號)', addr)
    if match:
        addr = match.group(1)
        
    return addr

def geocode_pipeline(sample_size=20):
    cache_file = 'address_coords_cache_v2.json'
    
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache = json.load(f)
    else:
        cache = {}

    df = pd.read_csv('main_unbundled_lasso_v3.csv')
    addr_col = [c for c in df.columns if '土地位置' in c][0]
    
    unique_addrs = df[addr_col].unique()
    sample_addrs = unique_addrs[:sample_size]
    
    geolocator = Nominatim(user_agent="kaohsiung_v3_research")
    
    success = 0
    print(f"Start processing {sample_size} addresses...")
    
    for i, raw_addr in enumerate(sample_addrs):
        clean_addr = normalize_address(raw_addr)
        
        # 移除表情符號以避免 Windows cp950 編碼錯誤
        msg = f"[{i+1}/{sample_size}] Input: {clean_addr}"
        print(msg.encode('unicode_escape').decode('ascii')) # 安全列印
        
        try:
            # 確保有「高雄市」字樣
            query = clean_addr if "高雄" in clean_addr else f"高雄市{clean_addr}"
            location = geolocator.geocode(query, timeout=10)
            if location:
                print(f"   Success: {location.latitude}, {location.longitude}")
                cache[raw_addr] = {"lat": location.latitude, "lon": location.longitude}
                success += 1
            else:
                print("   Failed: Not Found")
                cache[raw_addr] = None
        except Exception as e:
            print(f"   Error: {e}")
            
        time.sleep(1.5)

    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=4)
        
    print(f"\nCompleted. Success rate: {success/sample_size*100:.1f}%")

if __name__ == "__main__":
    geocode_pipeline(20)
