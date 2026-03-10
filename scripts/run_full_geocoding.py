import pandas as pd
import numpy as np
from geopy.geocoders import ArcGIS
import time
import json
import os
import re
import datetime

def normalize_address(addr):
    if not isinstance(addr, str): return None
    addr = addr.translate(str.maketrans('０１２３４５６７８９', '0123456789'))
    if '~' in addr or '及' in addr:
        nums = [int(s) for s in re.findall(r'\d+', addr)]
        if len(nums) >= 2:
            mid = (nums[0] + nums[1]) // 2
            addr = re.sub(r'\d+[~及]\d+', str(mid), addr)
    addr = re.sub(r'\d+樓.*', '', addr)
    addr = re.sub(r'[一二三四五六七八九十百]+樓.*', '', addr)
    addr = re.sub(r'\(.*\)', '', addr)
    match = re.search(r'(.*?\d+號)', addr)
    if match:
        addr = match.group(1)
    return addr

def run_geocoding_full():
    input_file = 'main_unbundled_lasso_v3.csv'
    cache_file = 'geocoding_cache_full.json'
    
    df = pd.read_csv(input_file)
    addr_col = [c for c in df.columns if '土地位置' in c][0]
    unique_addrs = df[addr_col].unique()
    
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache = json.load(f)
    else:
        cache = {}
        
    todo_addrs = [a for a in unique_addrs if a not in cache]
    total_todo = len(todo_addrs)
    print(f"Total: {len(unique_addrs)} | Cached: {len(cache)} | To do: {total_todo}")
    
    if not todo_addrs:
        print("All done.")
        return

    geolocator = ArcGIS(user_agent="kaohsiung_housing_full_v1")
    
    batch_size = 50
    start_time = time.time()
    
    print("Start...\n")
    try:
        for i, raw_addr in enumerate(todo_addrs):
            if i > 0 and i % batch_size == 0:
                elapsed = time.time() - start_time
                avg_time_per_record = elapsed / i
                eta_seconds = avg_time_per_record * (total_todo - i)
                eta_td = datetime.timedelta(seconds=int(eta_seconds))
                
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(cache, f, ensure_ascii=False, indent=4)
                    
                print(f"[Progress {i}/{total_todo}] Saved cache.")
                print(f"  > Past {batch_size} records took: {elapsed - (avg_time_per_record * (i - batch_size)):.1f} s")
                print(f"  > Avg time per record: {avg_time_per_record:.2f} s")
                print(f"  > ETA for remaining {total_todo - i} records: {eta_td}")
                print("-" * 40)
                break # 測試時跑到 50 筆就中斷，以便快速回報給使用者

            clean_addr = normalize_address(raw_addr)
            query = clean_addr if "高雄" in clean_addr else f"高雄市{clean_addr}"
            
            try:
                location = geolocator.geocode(query, timeout=10)
                if location:
                    cache[raw_addr] = {"lat": location.latitude, "lon": location.longitude}
                else:
                    cache[raw_addr] = None
            except Exception as e:
                print(f"[!] Error processing {raw_addr}: {e}")
                time.sleep(5)
                
            time.sleep(0.3)

    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving...")
    finally:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=4)
        print("Finished.")

if __name__ == "__main__":
    run_geocoding_full()
