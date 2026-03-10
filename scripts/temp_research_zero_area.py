import pandas as pd
import numpy as np

df = pd.read_csv('main.csv')

# 定義「零面積但有位子」的情况
mask_zero_area = (df['車位筆棟數'] > 0) & (df['車位移轉總面積坪'] == 0)
subset_zero = df[mask_zero_area]

# 定義「有面積且有位子」的情况
mask_has_area = (df['車位筆棟數'] > 0) & (df['車位移轉總面積坪'] > 0)
subset_has_area = df[mask_has_area]

print(f"總交易筆數: {len(df)}")
print(f"--- 零面積車位 ({len(subset_zero)} 筆, 佔比 {len(subset_zero)/len(df):.2%}) ---")
print(f"  其中有標價 (扣除用): {len(subset_zero[subset_zero['車位總價元'] > 0])}")
print(f"  其中無標價 (需要補值): {len(subset_zero[subset_zero['車位總價元'] == 0])}")

print(f"\n--- 有面積車位 ({len(subset_has_area)} 筆, 佔比 {len(subset_has_area)/len(df):.2%}) ---")
print(f"  其中有標價 (扣除用): {len(subset_has_area[subset_has_area['車位總價元'] > 0])}")
print(f"  其中無標價 (需要補值): {len(subset_has_area[subset_has_area['車位總價元'] == 0])}")

# 比較價格分佈
print("\n[價格分佈比較 (車位總價元 > 0)]")
price_zero = subset_zero[subset_zero['車位總價元'] > 0]['車位總價元']
price_has = subset_has_area[subset_has_area['車位總價元'] > 0]['車位總價元']

if not price_zero.empty and not price_has.empty:
    print(f"零面積車位平均價: {price_zero.mean():,.0f}")
    print(f"有面積車位平均價: {price_has.mean():,.0f}")
    print(f"零面積車位中位數: {price_zero.median():,.0f}")
    print(f"有面積車位中位數: {price_has.median():,.0f}")
else:
    print("無法比較價格分佈 (缺少數據)")

# 看看建物型態分佈
print("\n[零面積車位的建物型態分佈]")
print(subset_zero['建物型態'].value_counts())
