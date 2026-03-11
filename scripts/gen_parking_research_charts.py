"""
gen_parking_research_charts.py

依據備份分析報告，使用 backup_archive/main_unbundled_lasso_v3.csv
重新生成車位深度研究圖表，輸出至 visuals/eda/。

此腳本取代原本指向已不存在 main.csv 的舊研究腳本：
  - scripts/research_historical_public_ratio.py
  - scripts/research_zero_area_public_ratio.py

執行方式：python scripts/gen_parking_research_charts.py

輸出圖表：
  - parking_type_distribution.png    — 各建物型態零面積車位佔比
  - parking_public_ratio_comparison.png — 公設比三組箱型圖
  - historical_public_ratio_trend.png   — 歷年（建築完成年）公設比趨勢（覆蓋舊版）
  - price_correction_kde.png           — Lasso 拆車位前後 KDE 對比（覆蓋舊版）
"""

import os
import sys

# 強制 stdout 使用 utf-8，避免 Windows cp950 亂碼
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde

# ─────────────────────────────────────────
# 設定路徑
# ─────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'backup_archive', 'main_unbundled_lasso_v3.csv')
OUT_DIR   = os.path.join(BASE_DIR, 'visuals', 'eda')
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────
# 字型設定（支援中文）
# ─────────────────────────────────────────
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ─────────────────────────────────────────
# 載入資料
# ─────────────────────────────────────────
print(f"載入資料：{DATA_PATH}")
df = pd.read_csv(DATA_PATH, encoding='utf-8')
print(f"  共 {len(df):,} 筆，{len(df.columns)} 欄")

# 確認關鍵欄位
KEY_COLS = {
    '建物型態': '建物型態',
    '單價元坪': '單價元坪',
    '車位筆棟數': '車位筆棟數',
    '建築完成年': '建築完成年',
    '屋齡': '屋齡',
    '公設比': '公設比',
    'is_zero_area': 'is_zero_area',
    '淨屋單價元坪': '淨屋單價元坪',
}
missing = [k for k in KEY_COLS if k not in df.columns]
if missing:
    print(f"[警告] 找不到欄位：{missing}")
    print(f"  實際欄位：{df.columns.tolist()}")
    sys.exit(1)

print(f"  欄位確認完成")


# ─────────────────────────────────────────
# 輔助：車位分類標籤
# ─────────────────────────────────────────
def get_parking_group(row):
    if row['車位筆棟數'] == 0:
        return '無車位'
    elif row['is_zero_area'] == 1:
        return 'A.零面積車位'
    else:
        return 'B.有面積車位'


# ─────────────────────────────────────────
# 圖A：parking_type_distribution.png
# 各建物型態中零面積車位的佔比（水平條形圖）
# ─────────────────────────────────────────
def plot_parking_type_distribution():
    # 只看有車位的案件
    has_parking = df[df['車位筆棟數'] > 0].copy()

    # 簡化建物型態標籤
    type_map = {
        '住宅大樓(11層含以上有電梯)': '住宅大樓',
        '華廈(10層含以下有電梯)': '華廈',
        '公寓(5樓含以下無電梯)': '公寓',
        '透天厝': '透天厝',
        '套房(1房1廳1衛)': '套房',
    }
    has_parking['型態'] = has_parking['建物型態'].map(type_map).fillna(has_parking['建物型態'])

    # 計算各型態中零面積車位的佔比
    grp = has_parking.groupby('型態')['is_zero_area'].agg(['sum', 'count'])
    grp['pct_zero'] = grp['sum'] / grp['count'] * 100
    grp['pct_normal'] = 100 - grp['pct_zero']
    grp = grp.sort_values('pct_zero', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    y = np.arange(len(grp))
    bar_h = 0.5

    bars_normal = ax.barh(y, grp['pct_normal'], height=bar_h,
                          color='#3B82C4', alpha=0.85, label='有面積車位')
    bars_zero   = ax.barh(y, grp['pct_zero'],   height=bar_h,
                          left=grp['pct_normal'],
                          color='#E05C3B', alpha=0.85, label='零面積車位')

    ax.set_yticks(y)
    ax.set_yticklabels(grp.index, fontsize=11)
    ax.set_xlabel('佔有車位案件之比例（%）', fontsize=11)
    ax.set_title('各建物型態中零面積車位比例', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlim(0, 105)
    ax.xaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)

    # 在各 bar 右側標注百分比
    for i, (_, row) in enumerate(grp.iterrows()):
        if row['pct_zero'] > 2:
            ax.text(row['pct_normal'] + row['pct_zero'] / 2,
                    i, f"{row['pct_zero']:.1f}%",
                    ha='center', va='center', fontsize=9, color='white', fontweight='bold')
        ax.text(101, i, f"n={int(row['count']):,}",
                ha='left', va='center', fontsize=8.5, color='#555')

    ax.legend(loc='lower right', fontsize=10)
    sns.despine(ax=ax, left=False, bottom=False)
    plt.tight_layout()

    out = os.path.join(OUT_DIR, 'parking_type_distribution.png')
    fig.savefig(out, bbox_inches='tight', dpi=130)
    plt.close(fig)
    print(f"  ✓ {out}")


# ─────────────────────────────────────────
# 圖B：parking_public_ratio_comparison.png
# 三組（零面積/有面積/無車位）公設比箱型圖
# ─────────────────────────────────────────
def plot_parking_public_ratio_comparison():
    # 篩選：大樓+華廈，公設比有效範圍
    valid_types = ['住宅大樓(11層含以上有電梯)', '華廈(10層含以下有電梯)']
    df_plot = df[
        df['建物型態'].isin(valid_types) &
        (df['公設比'] > 0.05) &
        (df['公設比'] < 0.60)
    ].copy()

    df_plot['車位分類'] = df_plot.apply(get_parking_group, axis=1)

    # 計算各組統計
    stats = df_plot.groupby('車位分類')['公設比'].agg(['median', 'mean', 'count'])

    order = ['B.有面積車位', 'A.零面積車位', '無車位']
    order = [o for o in order if o in df_plot['車位分類'].unique()]

    palette = {'A.零面積車位': '#E05C3B', 'B.有面積車位': '#3B82C4', '無車位': '#8CB4D2'}

    fig, ax = plt.subplots(figsize=(10, 6))
    # seaborn ≥0.13 需要 hue + legend=False 來使用 palette
    sns.boxplot(data=df_plot, x='車位分類', y='公設比', order=order,
                hue='車位分類', palette=palette, legend=False,
                showfliers=False,
                boxprops=dict(linewidth=1.5),
                medianprops=dict(color='#1E3A5F', linewidth=2.5),
                whiskerprops=dict(linewidth=1.2),
                capprops=dict(linewidth=1.2),
                ax=ax)

    # 標注均值
    for i, grp_name in enumerate(order):
        if grp_name in stats.index:
            mean_val = stats.loc[grp_name, 'mean']
            cnt = int(stats.loc[grp_name, 'count'])
            ax.scatter(i, mean_val, marker='D', color='white',
                       edgecolors='#1E3A5F', s=60, zorder=5)
            ax.text(i, mean_val + 0.015, f'均值\n{mean_val:.3f}',
                    ha='center', va='bottom', fontsize=9, color='#1E3A5F')
            ax.text(i, 0.06, f'n={cnt:,}',
                    ha='center', va='bottom', fontsize=8.5, color='#555')

    ax.set_xlabel('', fontsize=11)
    ax.set_ylabel('公設比', fontsize=12)
    ax.set_title('大樓/華廈：依車位類型分組的公設比對比\n（零面積車位組公設比顯著偏高）',
                 fontsize=13, fontweight='bold', pad=12)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)

    # 簡化 x 標籤（先設定 ticks 位置再設標籤文字）
    display_labels = {'A.零面積車位': '零面積車位\n（面積併入公設）',
                      'B.有面積車位': '有面積車位\n（正常登記）',
                      '無車位': '無車位'}
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([display_labels.get(o, o) for o in order], fontsize=10)

    sns.despine(ax=ax)
    plt.tight_layout()

    out = os.path.join(OUT_DIR, 'parking_public_ratio_comparison.png')
    fig.savefig(out, bbox_inches='tight', dpi=130)
    plt.close(fig)
    print(f"  ✓ {out}")


# ─────────────────────────────────────────
# 圖C：historical_public_ratio_trend.png（覆蓋舊版）
# 歷年建築公設比趨勢（依建築完成年，5年一組）
# ─────────────────────────────────────────
def plot_historical_public_ratio_trend():
    # 建築完成年已為西元年格式（如 1996, 2008），直接過濾有效範圍
    def parse_completion_year(val):
        try:
            yr = int(float(val))
            if 1960 <= yr <= 2026:
                return yr
        except Exception:
            pass
        return None

    df_work = df.copy()
    df_work['建築完成年_西元'] = df_work['建築完成年'].apply(parse_completion_year)

    # 篩選：大樓+華廈，公設比有效，完成年有效
    valid_types = ['住宅大樓(11層含以上有電梯)', '華廈(10層含以下有電梯)']
    df_trend = df_work[
        df_work['建物型態'].isin(valid_types) &
        df_work['建築完成年_西元'].notnull() &
        (df_work['公設比'] > 0.05) &
        (df_work['公設比'] < 0.60)
    ].copy()

    df_trend['車位分類'] = df_trend.apply(get_parking_group, axis=1)
    df_trend['建築時期'] = (df_trend['建築完成年_西元'] // 5) * 5

    # 計算各時期×分類的平均公設比
    grp = (df_trend.groupby(['建築時期', '車位分類'])['公設比']
           .mean().reset_index())

    fig, ax = plt.subplots(figsize=(12, 6))

    color_map = {
        'A.零面積車位': '#E05C3B',
        'B.有面積車位': '#3B82C4',
        '無車位':       '#8CB4D2',
    }
    label_map = {
        'A.零面積車位': '零面積車位（面積已併入公設）',
        'B.有面積車位': '有面積車位（正常登記）',
        '無車位':       '無車位',
    }

    for cat in ['B.有面積車位', 'A.零面積車位', '無車位']:
        sub = grp[grp['車位分類'] == cat].sort_values('建築時期')
        if sub.empty:
            continue
        ax.plot(sub['建築時期'], sub['公設比'],
                marker='o', linewidth=2.2, markersize=6,
                color=color_map[cat], label=label_map[cat])
        ax.fill_between(sub['建築時期'], sub['公設比'],
                        alpha=0.08, color=color_map[cat])

    ax.set_xlabel('建築完成年代（每5年一組）', fontsize=12)
    ax.set_ylabel('平均公設比', fontsize=12)
    ax.set_title('歷年建築公設比趨勢：依車位類型分組\n（零面積車位組持續維持偏高公設比，反映車位面積併入公設的歷史慣例）',
                 fontsize=13, fontweight='bold', pad=12)
    ax.legend(fontsize=10, loc='upper left')
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
    sns.despine(ax=ax)
    plt.tight_layout()

    out = os.path.join(OUT_DIR, 'historical_public_ratio_trend.png')
    fig.savefig(out, bbox_inches='tight', dpi=130)
    plt.close(fig)
    print(f"  ✓ {out}")


# ─────────────────────────────────────────
# 圖D：price_correction_kde.png（覆蓋舊版）
# Lasso 拆車位前後單價分布 KDE 對比
# ─────────────────────────────────────────
def plot_price_correction_kde():
    # 篩選有效值（去掉極端離群）
    col_before = '單價元坪'
    col_after  = '淨屋單價元坪'
    x_max = 100  # 顯示上限（萬/坪）

    valid = df[
        (df[col_before] > 0) &
        (df[col_after]  > 0) &
        (df[col_before] / 10000 <= x_max) &
        (df[col_after]  / 10000 <= x_max)
    ]

    prices_before = valid[col_before] / 10000  # 萬元/坪
    prices_after  = valid[col_after]  / 10000

    x = np.linspace(0, x_max, 800)
    kde_b = gaussian_kde(prices_before, bw_method=0.06)
    kde_a = gaussian_kde(prices_after,  bw_method=0.06)
    y_b = kde_b(x)
    y_a = kde_a(x)

    peak_b = x[np.argmax(y_b)]
    peak_a = x[np.argmax(y_a)]

    fig, ax = plt.subplots(figsize=(12, 5))

    # 修正前（橘）
    ax.fill_between(x, y_b, alpha=0.22, color='#E05C3B')
    ax.plot(x, y_b, color='#E05C3B', linewidth=2,
            label=f'修正前（含車位單價）  峰值 {peak_b:.0f} 萬/坪')

    # 修正後（藍）
    ax.fill_between(x, y_a, alpha=0.28, color='#3B82C4')
    ax.plot(x, y_a, color='#3B82C4', linewidth=2,
            label=f'修正後（淨屋單價）    峰值 {peak_a:.0f} 萬/坪')

    # 峰值移動箭頭
    y_arrow = max(y_b.max(), y_a.max()) * 0.75
    ax.annotate('',
                xy=(peak_a, y_arrow), xytext=(peak_b, y_arrow),
                arrowprops=dict(arrowstyle='->', color='#555', lw=1.5))
    ax.text((peak_b + peak_a) / 2, y_arrow * 1.05,
            f'峰值左移 {peak_b - peak_a:.1f} 萬',
            ha='center', fontsize=9.5, color='#555')

    ax.set_xlim(0, x_max)
    ax.set_ylim(0)
    ax.set_xlabel('單價（萬元／坪）', fontsize=12)
    ax.set_ylabel('密度', fontsize=12)
    ax.set_title('Lasso 車位拆算前後：單價分布對比\n（修正後分布峰值左移，右尾縮短，更集中反映純住宅單價）',
                 fontsize=13, fontweight='bold', pad=12)
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, linestyle='--', alpha=0.35)
    ax.set_axisbelow(True)
    sns.despine(ax=ax)
    plt.tight_layout()

    out = os.path.join(OUT_DIR, 'price_correction_kde.png')
    fig.savefig(out, bbox_inches='tight', dpi=130)
    plt.close(fig)
    print(f"  ✓ {out}")


# ─────────────────────────────────────────
# 主程式
# ─────────────────────────────────────────
if __name__ == '__main__':
    print("\n[1/4] 建物型態 × 零面積車位佔比圖...")
    plot_parking_type_distribution()

    print("[2/4] 公設比三組箱型圖...")
    plot_parking_public_ratio_comparison()

    print("[3/4] 歷年建築公設比趨勢圖...")
    plot_historical_public_ratio_trend()

    print("[4/4] Lasso 拆車位前後 KDE 對比...")
    plot_price_correction_kde()

    print("\n完成！共生成 4 張圖表至 visuals/eda/")
