"""
gen_missing_eda_charts.py
重新生成 EDA 數據藝廊所需的缺失圖片，儲存至 visuals/eda/
"""

import os
import sys
# 強制 stdout 使用 utf-8，避免 Windows cp950 亂碼
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────
# 設定路徑
# ─────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "cleaned_all.csv")
OUT_DIR   = os.path.join(BASE_DIR, "visuals", "eda")
os.makedirs(OUT_DIR, exist_ok=True)

# 統一字型（避免中文亂碼）
plt.rcParams['font.family']       = ['Microsoft JhengHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi']        = 120

print("載入資料中…")
df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"  共 {len(df):,} 筆")


# ─────────────────────────────────────────
# 圖1：4_yearly_trend.png
# 歷年零面積車位比例折線圖
# ─────────────────────────────────────────
def plot_yearly_trend():
    has_parking = df[(df['車位筆棟數'] > 0) & (df['交易年'] >= 2019) & (df['交易年'] <= 2026)].copy()
    grp = has_parking.groupby('交易年')['is_zero_area'].agg(['sum', 'count'])
    grp['pct'] = grp['sum'] / grp['count'] * 100
    grp = grp.sort_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(grp.index, grp['pct'], marker='o', color='#E05C3B', linewidth=2.5, markersize=7)
    ax.fill_between(grp.index, grp['pct'], alpha=0.15, color='#E05C3B')
    for yr, row in grp.iterrows():
        ax.annotate(f"{row['pct']:.1f}%", (yr, row['pct']),
                    textcoords="offset points", xytext=(0, 10),
                    ha='center', fontsize=9, color='#666')
    ax.set_title('歷年零面積車位佔比趨勢\n（有車位案件中，登記面積為 0 的比例）',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('交易年', fontsize=11)
    ax.set_ylabel('零面積車位佔比 (%)', fontsize=11)
    ax.set_xticks(grp.index)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, '4_yearly_trend.png')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {out}")


# ─────────────────────────────────────────
# 圖2：5_age_distribution.png
# 零面積 vs 有面積車位屋齡 KDE 對比
# ─────────────────────────────────────────
def plot_age_distribution():
    has_parking = df[df['車位筆棟數'] > 0].copy()
    zero = has_parking[has_parking['is_zero_area'] == 1]['屋齡'].dropna()
    normal = has_parking[has_parking['is_zero_area'] == 0]['屋齡'].dropna()

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.kdeplot(zero,   ax=ax, fill=True,  color='#E05C3B', alpha=0.45,
                label=f'零面積車位（n={len(zero):,}）', linewidth=2)
    sns.kdeplot(normal, ax=ax, fill=True,  color='#3B82C4', alpha=0.35,
                label=f'有面積車位（n={len(normal):,}）', linewidth=2)
    ax.axvline(zero.median(),   linestyle='--', color='#E05C3B', alpha=0.8,
               label=f'零面積中位屋齡 {zero.median():.0f} 年')
    ax.axvline(normal.median(), linestyle='--', color='#3B82C4', alpha=0.8,
               label=f'有面積中位屋齡 {normal.median():.0f} 年')
    ax.set_title('零面積 vs 有面積車位 — 屋齡分布對比', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('屋齡（年）', fontsize=11)
    ax.set_ylabel('密度', fontsize=11)
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, '5_age_distribution.png')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {out}")


# ─────────────────────────────────────────
# 圖3：boxplot_net_price_overall.png
# 歷年淨屋單價箱型圖
# ─────────────────────────────────────────
def plot_boxplot_net_price():
    col = '淨屋單價元坪'
    prices = df[(df[col] > 0) & (df[col] / 10000 <= 60)][col] / 10000

    q1, med, q3 = prices.quantile(0.25), prices.median(), prices.quantile(0.75)
    iqr = q3 - q1

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(prices, vert=False, patch_artist=True, widths=0.5,
        boxprops=dict(facecolor='#93C4E8', edgecolor='#1E3A5F', linewidth=1.8),
        medianprops=dict(color='#E05C3B', linewidth=3),
        whiskerprops=dict(color='#1E3A5F', linewidth=1.5),
        capprops=dict(color='#1E3A5F', linewidth=1.5),
        flierprops=dict(marker='o', markerfacecolor='#aaa', markersize=3,
                        alpha=0.3, linestyle='none'))

    ax.axvline(med, color='#E05C3B', linewidth=1.5, linestyle='--', alpha=0.6)
    ax.axvline(q1,  color='#3B82C4', linewidth=1,   linestyle=':',  alpha=0.5)
    ax.axvline(q3,  color='#3B82C4', linewidth=1,   linestyle=':',  alpha=0.5)

    y_top = 1.35
    ax.text(q1,  y_top, f'Q1\n{q1:.1f}',       ha='center', va='bottom', fontsize=10, color='#3B82C4')
    ax.text(med, y_top, f'中位數\n{med:.1f}',   ha='center', va='bottom', fontsize=10, color='#E05C3B', fontweight='bold')
    ax.text(q3,  y_top, f'Q3\n{q3:.1f}',       ha='center', va='bottom', fontsize=10, color='#3B82C4')

    ax.annotate('', xy=(q3, 1.0), xytext=(q1, 1.0),
                arrowprops=dict(arrowstyle='<->', color='#555', lw=1.5))
    ax.text((q1 + q3) / 2, 0.92, f'IQR = {iqr:.1f} 萬', ha='center', fontsize=9.5, color='#555')

    ax.set_yticks([])
    ax.set_xlabel('淨屋單價（萬元／坪）', fontsize=12)
    ax.set_title('淨屋單價整體分布（箱型圖，已排除離群值 > 60 萬/坪）',
                 fontsize=13, fontweight='bold', pad=14)
    ax.set_xlim(0, 62)
    ax.xaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    sns.despine(ax=ax, left=True)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'boxplot_net_price_overall.png')
    fig.savefig(out, bbox_inches='tight', dpi=130)
    plt.close(fig)
    print(f"  ✓ {out}")


# ─────────────────────────────────────────
# 圖4：1_price_distribution_with_threshold.png
# 淨屋單價分布 KDE + 門檻線
# ─────────────────────────────────────────
def plot_price_distribution_threshold():
    import numpy as np
    from scipy.stats import gaussian_kde

    col = '淨屋單價元坪'
    prices = df[df[col] > 0][col] / 10000  # 萬元/坪

    threshold = 60          # 偽屋門檻 萬/坪
    x_max     = 120         # 顯示範圍上限

    # 用 scipy KDE 取得平滑曲線
    kde = gaussian_kde(prices, bw_method=0.08)
    x   = np.linspace(0, x_max, 800)
    y   = kde(x)

    fig, ax = plt.subplots(figsize=(12, 5))

    # 左側主分布（正常區段）：藍色填滿
    mask_left  = x <= threshold
    mask_right = x >= threshold
    ax.fill_between(x[mask_left],  y[mask_left],  color='#3B82C4', alpha=0.35, zorder=2)
    ax.fill_between(x[mask_right], y[mask_right], color='#E05C3B', alpha=0.25, zorder=2)
    ax.plot(x, y, color='#1E3A5F', linewidth=2, zorder=3)

    peak_x = x[np.argmax(y)]
    peak_y = y.max()

    # 完整紅色虛線貫穿全圖
    ax.axvline(threshold, color='#E05C3B', linewidth=2.2, linestyle='--', zorder=5)

    # 門檻標籤（頂部）
    ax.text(threshold + 1.2, peak_y * 1.08,
            '← 離群值分界點\n   60 萬 / 坪',
            fontsize=10.5, color='#C0392B', va='top', fontweight='bold')

    # 右側佔比
    pct_above = (prices > threshold).mean() * 100
    ax.text(threshold + 4, peak_y * 0.25,
            f'右側佔 {pct_above:.1f}%\n共 {(prices > threshold).sum():,} 筆',
            fontsize=9.5, color='#C0392B', alpha=0.9)

    # 峰值標注
    ax.annotate(f'峰值約 {peak_x:.0f} 萬/坪',
        xy=(peak_x, peak_y), xytext=(peak_x - 16, peak_y * 0.90),
        fontsize=10, color='#1E3A5F',
        arrowprops=dict(arrowstyle='->', color='#1E3A5F', lw=1.2))

    ax.text(threshold * 0.42, peak_y * 0.13, '正常交易區間',
            fontsize=10, color='#3B82C4', ha='center', alpha=0.85)
    ax.text(threshold + (x_max - threshold) * 0.38, peak_y * 0.13, '離群 / 偽屋區',
            fontsize=10, color='#E05C3B', ha='center', alpha=0.85)

    ax.set_xlim(0, x_max)
    ax.set_ylim(0, peak_y * 1.25)
    ax.set_title('淨屋單價分布與離群值門檻', fontsize=14, fontweight='bold', pad=14)
    ax.set_xlabel('淨屋單價（萬元／坪）', fontsize=12)
    ax.set_ylabel('密度', fontsize=12)
    ax.yaxis.grid(True, linestyle='--', alpha=0.35)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, '1_price_distribution_with_threshold.png')
    fig.savefig(out, bbox_inches='tight', dpi=130)
    plt.close(fig)
    print(f"  ✓ {out}")


# ─────────────────────────────────────────
# 圖5：2_fake_house_scatter.png
# 面積 vs 單價散點圖（標記偽屋邊界）
# ─────────────────────────────────────────
def plot_fake_house_scatter():
    col_area  = '建物淨面積坪'
    col_price = '淨屋單價元坪'

    # 若無建物淨面積坪，用建物移轉總面積坪代替
    if col_area not in df.columns:
        col_area = '建物移轉總面積坪'

    data = df[(df[col_area] > 0) & (df[col_price] > 0)].copy()
    data['price_wan'] = data[col_price] / 10000
    # 只繪製 area < 80坪、price < 150萬 的範圍（顯示全貌）
    vis = data[(data[col_area] < 80) & (data['price_wan'] < 150)].copy()

    fake_mask = (vis[col_area] < 5) & (vis['price_wan'] > 60)
    normal    = vis[~fake_mask].sample(min(30000, (~fake_mask).sum()), random_state=42)
    fake      = vis[fake_mask]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.scatter(normal[col_area], normal['price_wan'],
               s=5, color='#3B82C4', alpha=0.15, label=f'一般資料（取樣，n={len(normal):,}）')
    ax.scatter(fake[col_area], fake['price_wan'],
               s=40, color='#E05C3B', alpha=0.8, zorder=5,
               label=f'偽屋（面積<5坪 且 單價>60萬，n={len(fake)}）')

    # 篩選框
    rect = plt.Rectangle((0, 60), 5, 90, linewidth=2,
                          edgecolor='#E05C3B', facecolor='#E05C3B', alpha=0.08, linestyle='--')
    ax.add_patch(rect)
    ax.axvline(5,  color='#E05C3B', linestyle='--', linewidth=1.2, alpha=0.7)
    ax.axhline(60, color='#E05C3B', linestyle='--', linewidth=1.2, alpha=0.7)
    ax.text(5.3, 130, '雙重篩選區域\n面積 < 5坪\nAND\n單價 > 60 萬/坪',
            fontsize=9, color='#E05C3B', va='top')

    ax.set_title('偽屋散點分布圖', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('建物淨面積（坪）', fontsize=11)
    ax.set_ylabel('淨屋單價（萬元／坪）', fontsize=11)
    ax.legend(fontsize=9, loc='upper right')
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, '2_fake_house_scatter.png')
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ {out}")


# ─────────────────────────────────────────
# 圖6：district_trends_v5_growth_part1~4.png
# 行政區年度中位數折線小多圖，依漲幅排序
# ─────────────────────────────────────────
def plot_district_growth():
    col = '淨屋單價元坪'
    # 只使用 2019–2026 的資料（專案分析範圍）
    data = df[(df[col] > 0) & (df['交易年'] >= 2019) & (df['交易年'] <= 2026)].copy()
    data['price_wan'] = data[col] / 10000

    # 計算各行政區 × 交易年 中位數
    grp = (data.groupby(['鄉鎮市區', '交易年'])['price_wan']
               .median().reset_index())
    grp.columns = ['district', 'year', 'median_price']

    years = sorted(grp['year'].dropna().unique())
    min_yr, max_yr = years[0], years[-1]

    # 只保留在 min_yr 和 max_yr 都有資料的行政區
    districts_min = set(grp[grp['year'] == min_yr]['district'])
    districts_max = set(grp[grp['year'] == max_yr]['district'])
    valid = sorted(districts_min & districts_max)

    # 計算漲幅
    def growth(d):
        p_start = grp[(grp['district'] == d) & (grp['year'] == min_yr)]['median_price'].values[0]
        p_end   = grp[(grp['district'] == d) & (grp['year'] == max_yr)]['median_price'].values[0]
        return (p_end - p_start) / p_start * 100, p_end

    stats = {d: growth(d) for d in valid}
    sorted_districts = sorted(valid, key=lambda d: stats[d][0], reverse=True)

    # 特別標色的行政區（台積電周邊）
    highlight = {'楠梓區', '橋頭區', '左營區', '岡山區', '仁武區'}

    def make_parts(districts_list, part_num):
        n = len(districts_list)
        ncols = 4
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.2),
                                 constrained_layout=True)
        axes = axes.flatten()

        for i, d in enumerate(districts_list):
            ax = axes[i]
            sub = grp[grp['district'] == d].sort_values('year')
            color = '#E05C3B' if d in highlight else '#3B82C4'
            ax.plot(sub['year'], sub['median_price'],
                    marker='o', color=color, linewidth=2.5, markersize=5)
            ax.fill_between(sub['year'], sub['median_price'], alpha=0.12, color=color)

            pct, latest = stats[d]
            title_color = '#C0392B' if d in highlight else '#1E3A5F'
            ax.set_title(f"{d}\n漲幅 {pct:+.1f}%  ▸  {latest:.1f}萬/坪",
                         fontsize=9.5, color=title_color, fontweight='bold', pad=6)
            ax.set_xticks(years[::2])
            ax.set_xticklabels([str(int(y)) for y in years[::2]], fontsize=7, rotation=30)
            ax.yaxis.set_tick_params(labelsize=8)
            ax.yaxis.grid(True, linestyle='--', alpha=0.4)
            ax.set_axisbelow(True)
            sns.despine(ax=ax)

        # 隱藏多餘格
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        # 整體標題
        rank_start = (part_num - 1) * 8 + 1
        rank_end   = rank_start + n - 1
        highlight_note = '  [橘色 = 台積電周邊區域]' if any(d in highlight for d in districts_list) else ''
        fig.suptitle(f'行政區房價趨勢（按漲幅排序，第 {rank_start}–{rank_end} 名）{highlight_note}',
                     fontsize=13, fontweight='bold', y=1.01)

        out = os.path.join(OUT_DIR, f'district_trends_v5_growth_part{part_num}.png')
        fig.savefig(out, bbox_inches='tight', dpi=130)
        plt.close(fig)
        print(f"  ✓ {out}")

    # 拆成 4 張（每張最多 8 個行政區）
    chunk = 8
    for i in range(4):
        chunk_districts = sorted_districts[i * chunk:(i + 1) * chunk]
        if chunk_districts:
            make_parts(chunk_districts, i + 1)


# ─────────────────────────────────────────
# 執行所有圖表生成
# ─────────────────────────────────────────
if __name__ == '__main__':
    print("\n=== 開始生成缺失 EDA 圖表 ===\n")

    print("圖1：歷年零面積車位比例趨勢…")
    plot_yearly_trend()

    print("圖2：屋齡分布對比…")
    plot_age_distribution()

    print("圖3：歷年淨屋單價箱型圖…")
    plot_boxplot_net_price()

    print("圖4：單價分布 + 門檻線…")
    plot_price_distribution_threshold()

    print("圖5：偽屋散點圖…")
    plot_fake_house_scatter()

    print("圖6：行政區漲幅小多圖（4張）…")
    plot_district_growth()

    print("\n=== 全部完成 ===")
    print(f"圖片儲存於：{OUT_DIR}")
