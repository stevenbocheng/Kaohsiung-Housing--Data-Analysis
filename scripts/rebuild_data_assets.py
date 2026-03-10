import pandas as pd
import json
import os

def rebuild():
    source = "backup_archive/main_unbundled_lasso_v3_with_pca.csv"
    if not os.path.exists(source):
        print(f"找不到原始資料: {source}")
        return

    print("載入原始資料...")
    df = pd.read_csv(source)
    
    # 1. 恢復 data/cleaned_all.csv
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/cleaned_all.csv", index=False)
    print("已恢復 data/cleaned_all.csv")
    
    # 2. 重建 app/street_coords_cache.json
    # 索引: 40 (Street), 41 (lat), 42 (lon)
    print("重建座標快照...")
    # 注意：如果原本是用 Street 欄位做 key，就保持一致
    street_col = df.columns[40]
    lat_col = df.columns[41]
    lon_col = df.columns[42]
    
    coords_df = df[[street_col, lat_col, lon_col]].drop_duplicates(subset=[street_col])
    coords_dict = {}
    for _, row in coords_df.iterrows():
        street = str(row[street_col])
        lat = float(row[lat_col])
        lon = float(row[lon_col])
        if street and street != 'nan':
            coords_dict[street] = [lat, lon]

    os.makedirs("app", exist_ok=True)
    with open("app/street_coords_cache.json", "w", encoding="utf-8") as f:
        json.dump(coords_dict, f, ensure_ascii=False, indent=4)
    print("已重建 app/street_coords_cache.json")

if __name__ == "__main__":
    rebuild()
