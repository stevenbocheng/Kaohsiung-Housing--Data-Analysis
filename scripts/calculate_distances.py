import pandas as pd
import numpy as np

def calculate_all_distances():
    print("載入物件座標與 POI 資料...")
    df_houses = pd.read_csv('main_unbundled_lasso_v3_with_coords.csv')
    df_poi = pd.read_csv('kaohsiung_poi_coords.csv').dropna(subset=['lat', 'lon'])
    
    valid_mask = df_houses['lat'].notna() & df_houses['lon'].notna()
    df_valid = df_houses[valid_mask].copy()
    
    print(f"共有 {len(df_valid)} 筆物件具備座標，開始計算最短距離...")
    
    poi_coords = df_poi[['lat', 'lon']].values
    house_lats = df_valid['lat'].values
    house_lons = df_valid['lon'].values
    
    min_distances = np.zeros(len(df_valid))
    
    batch_size = 10000
    for i in range(0, len(df_valid), batch_size):
        end = min(i + batch_size, len(df_valid))
        batch_lats_rad = np.radians(house_lats[i:end].reshape(-1, 1))
        batch_lons_rad = np.radians(house_lons[i:end].reshape(-1, 1))
        
        poi_lats_rad = np.radians(poi_coords[:, 0].reshape(1, -1))
        poi_lons_rad = np.radians(poi_coords[:, 1].reshape(1, -1))
        
        dlat = poi_lats_rad - batch_lats_rad
        dlon = poi_lons_rad - batch_lons_rad
        
        a = np.sin(dlat/2)**2 + np.cos(batch_lats_rad) * np.cos(poi_lats_rad) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        distances_m = 6371000 * c
        
        min_distances[i:end] = distances_m.min(axis=1)
        print(f"  > 已完成 {end} 筆運算...")
        
    df_houses['最短大眾運輸距離_公尺'] = np.nan
    df_houses.loc[valid_mask, '最短大眾運輸距離_公尺'] = min_distances
    
    output_file = 'main_unbundled_lasso_v3_with_features.csv'
    df_houses.to_csv(output_file, index=False)
    
    print(f"\n計算完成，已儲存至 {output_file}")
    print(df_houses[['土地位置建物門牌', '最短大眾運輸距離_公尺']].dropna().head())

if __name__ == "__main__":
    calculate_all_distances()
