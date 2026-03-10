import pandas as pd
import json
import os

def generate_context():
    source_path = "backup_archive/main_unbundled_lasso_v3_with_pca.csv"
    if not os.path.exists(source_path):
        print(f"找不到原始資料: {source_path}")
        return

    print("載入資料中 (使用索引定位欄位)...")
    # 讀取資料，不指定 header 名稱以避免編碼問題
    df = pd.read_csv(source_path)
    
    # 根據之前的確認，使用以下索引：
    # 0: 鄉鎮市區
    # 39: 淨單價元坪
    # 49: PC1_整體大眾運輸依賴度
    # 50: PC2_北高雄產業樞紐軸度
    
    dist_idx = 0
    price_idx = 39
    pc1_idx = 49
    pc2_idx = 50
    
    # 重新命名欄位以便內部處理
    df = df.iloc[:, [dist_idx, price_idx, pc1_idx, pc2_idx]]
    df.columns = ['dist', 'price', 'pc1', 'pc2']
    
    # 確保數值型態
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['pc1'] = pd.to_numeric(df['pc1'], errors='coerce')
    df['pc2'] = pd.to_numeric(df['pc2'], errors='coerce')
    
    # 移除空值與異常值 (例如負數)
    df = df.dropna()
    df = df[df['price'] > 0]
    
    print("計算行政區統計指標...")
    market_context = {}
    
    # 依照行政區分組
    grouped = df.groupby('dist')
    
    for dist, group in grouped:
        # 清除行政區名稱中的亂碼或空白 (如果有)
        clean_dist = str(dist).strip().replace('?', '') 
        
        # PCA 均值
        pc1_avg = float(group['pc1'].mean())
        pc2_avg = float(group['pc2'].mean())
        
        # 價格指標
        avg_price = float(group['price'].mean())
        
        market_context[clean_dist] = {
            "PC1": pc1_avg,
            "PC2": pc2_avg,
            "District_MA180_Past": avg_price,
            "MA30_Momentum": avg_price * 1.02,
            "MA90_Momentum": avg_price * 1.01,
            "MA180_Momentum": avg_price
        }
        
    output_path = "app/market_context.json"
    os.makedirs("app", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(market_context, f, ensure_ascii=False, indent=4)
    
    print(f"市場行情快照已生成: {output_path}")

if __name__ == "__main__":
    generate_context()
