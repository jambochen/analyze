"""台電過去電力供需資訊（d006005）的篩選用檢視。

僅供判斷此候選是否適合作為選題，非分析結果。
範圍限於 CLAUDE.md 允許的篩選檢視：筆數、欄位型別、缺失率、
描述統計、單變數分布、時間序列折線圖。相關係數矩陣另標註。

本檔由 `taipower-supply-demand.py` 機械轉換而來（見 _to_notebook.py），
內容完全相同，只改為 `# %%` 分格以便轉 notebook：

    jupytext --to notebook scripts/explore/nb_taipower_supply_demand.py
"""

# %%
"""台電過去電力供需資訊（d006005）的篩選用檢視。

僅供判斷此候選是否適合作為選題，非分析結果。
範圍限於 CLAUDE.md 允許的篩選檢視：筆數、欄位型別、缺失率、
描述統計、單變數分布、時間序列折線圖。相關係數矩陣另標註。
"""

from pathlib import Path

import matplotlib

import matplotlib.pyplot as plt

import polars as pl

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[2]

SRC = ROOT / "data" / "raw" / "taipower-supply-demand" / "d006005.csv"

FIGDIR = ROOT / "outputs" / "figures"

FIGDIR.mkdir(parents=True, exist_ok=True)

for family in ("Noto Sans CJK TC", "Noto Sans CJK JP", "WenQuanYi Zen Hei"):
    if any(family in f.name for f in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = family
        break

plt.rcParams["axes.unicode_minus"] = False

SYSTEM_COLS = [
    "淨尖峰供電能力(萬瓩)",
    "尖峰負載(萬瓩)",
    "備轉容量(萬瓩)",
    "備轉容量率(%)",
    "工業用電(百萬度)",
    "民生用電(百萬度)",
]

def load() -> pl.DataFrame:
    df = pl.read_csv(SRC, encoding="utf-8-lossy", infer_schema_length=0)
    # 日期為 YYYYMMDD 整數字串；其餘欄位一律轉數值，無法解析者成為 null
    return df.with_columns(
        pl.col("日期").str.strptime(pl.Date, "%Y%m%d"),
        *[
            pl.col(c).cast(pl.Float64, strict=False)
            for c in df.columns
            if c != "日期"
        ],
    )

def section(title: str) -> None:
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")

def _plot_reserve_series(df: pl.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    axes[0].plot(df["日期"], df["尖峰負載(萬瓩)"], linewidth=1.1, color="#C44E52", label="尖峰負載")
    axes[0].plot(
        df["日期"], df["淨尖峰供電能力(萬瓩)"], linewidth=1.1, color="#4C72B0", label="淨尖峰供電能力"
    )
    axes[0].set_ylabel("萬瓩")
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)
    axes[0].set_title("(a) 供給能力與尖峰負載", fontsize=11)

    axes[1].plot(df["日期"], df["備轉容量率(%)"], linewidth=1.1, color="#55A868")
    axes[1].axhline(10, color="#DD8452", linestyle="--", linewidth=1, label="黃燈 10%")
    axes[1].axhline(6, color="#C44E52", linestyle="--", linewidth=1, label="橘燈 6%")
    axes[1].set_ylabel("備轉容量率 (%)")
    axes[1].set_xlabel("日期")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)
    axes[1].set_title("(b) 備轉容量率", fontsize=11)

    fig.suptitle("圖 4　台電系統供需指標日序列　僅供篩選，非分析結果", fontsize=12)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIGDIR / "taipower_reserve_series.png", dpi=130)
    plt.close(fig)

def _plot_demand_split(df: pl.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(df["日期"], df["工業用電(百萬度)"], linewidth=1.1, label="工業用電")
    ax.plot(df["日期"], df["民生用電(百萬度)"], linewidth=1.1, label="民生用電")
    ax.set_ylabel("百萬度")
    ax.set_xlabel("日期")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_title("圖 5　工業與民生用電日序列　僅供篩選，非分析結果", fontsize=12)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIGDIR / "taipower_demand_split.png", dpi=130)
    plt.close(fig)

def _plot_reserve_hist(df: pl.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.hist(df["備轉容量率(%)"].drop_nulls().to_numpy(), bins=40, color="#55A868", edgecolor="white")
    ax.axvline(10, color="#DD8452", linestyle="--", linewidth=1.2, label="黃燈 10%")
    ax.axvline(6, color="#C44E52", linestyle="--", linewidth=1.2, label="橘燈 6%")
    ax.set_xlabel("備轉容量率 (%)")
    ax.set_ylabel("天數")
    ax.legend(fontsize=9)
    ax.set_title("圖 6　備轉容量率分布　僅供篩選，非分析結果", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGDIR / "taipower_reserve_hist.png", dpi=130)
    plt.close(fig)

def _plot_renewable(df: pl.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 4))
    for c in ("風力發電(萬瓩)", "太陽能發電(萬瓩)"):
        if c in df.columns:
            ax.plot(df["日期"], df[c], linewidth=1.1, label=c.replace("(萬瓩)", ""))
    ax.set_ylabel("萬瓩")
    ax.set_xlabel("日期")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_title("圖 7　風力與太陽能日出力序列　僅供篩選，非分析結果", fontsize=12)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIGDIR / "taipower_renewable.png", dpi=130)
    plt.close(fig)

# %%
df = load()
numeric = [c for c in df.columns if c != "日期"]
unit_cols = [c for c in numeric if c not in SYSTEM_COLS]

# %% [markdown]
# ## 1. 規模與欄位型別

# %%
print(f"  列數: {df.height:,}")
print(f"  欄數: {df.width}")
print(f"    系統層級指標: {len(SYSTEM_COLS)} 欄")
print(f"    機組層級出力: {len(unit_cols)} 欄")
print(f"  日期型別: {df['日期'].dtype}")

# %% [markdown]
# ## 2. 時間軸完整性

# %%
d = df["日期"]
span = (d.max() - d.min()).days + 1
print(f"  最早: {d.min()}")
print(f"  最晚: {d.max()}")
print(f"  涵蓋天數(起訖): {span:,}")
print(f"  實際列數: {df.height:,}")
print(f"  缺漏日數: {span - df.height:,}")
print(f"  日期重複: {df.height - d.n_unique():,}")
# 找出實際缺漏的日期區間
full = pl.date_range(d.min(), d.max(), "1d", eager=True)
missing = full.filter(~full.is_in(d.implode()))
if missing.len():
    print(f"  缺漏日期 (前 10 筆，共 {missing.len():,}):")
    for x in missing.head(10):
        print(f"    {x}")

# %% [markdown]
# ## 3. 缺失率（系統層級指標）

# %%
for c in SYSTEM_COLS:
    n = df[c].null_count()
    print(f"  {c:<22} 缺失 {n:>4,} ({n / df.height * 100:5.2f}%)")

# %% [markdown]
# ## 4. 缺失率（機組層級，僅列出 >0 者）

# %%
miss = [(c, df[c].null_count()) for c in unit_cols]
nonzero = [(c, n) for c, n in miss if n]
if nonzero:
    for c, n in sorted(nonzero, key=lambda x: -x[1])[:20]:
        print(f"  {c:<26} 缺失 {n:>4,} ({n / df.height * 100:5.2f}%)")
    print(f"  （共 {len(nonzero)} 欄有缺失，{len(unit_cols) - len(nonzero)} 欄無缺失）")
else:
    print("  機組層級欄位全無缺失")

# %% [markdown]
# ## 5. 系統層級指標描述統計

# %%
hdr = f"  {'指標':<22} {'平均':>9} {'中位':>9} {'標準差':>8} {'最小':>9} {'最大':>9}"
print(hdr)
for c in SYSTEM_COLS:
    s = df[c]
    print(
        f"  {c:<22} {s.mean():>9.2f} {s.median():>9.2f} {s.std():>8.2f} "
        f"{s.min():>9.2f} {s.max():>9.2f}"
    )

# %% [markdown]
# ## 6. 備轉容量率的風險分級

# %%
# 台電燈號：備轉容量率 <6% 為橘燈（供電吃緊），<10% 為黃燈（供電警戒）
r = df["備轉容量率(%)"]
for label, cond in (
    ("< 6%（橘燈以下，供電吃緊）", r < 6),
    ("6% ~ 10%（黃燈，供電警戒）", (r >= 6) & (r < 10)),
    ("10% ~ 15%", (r >= 10) & (r < 15)),
    (">= 15%（供電充裕）", r >= 15),
):
    n = df.filter(cond).height
    print(f"  {label:<28} {n:>4,} 天 ({n / df.height * 100:5.1f}%)")

# %% [markdown]
# ## 7. 機組層級：全期為零或近乎為零的欄位

# %%
dead = []
for c in unit_cols:
    s = df[c]
    mx = s.max()
    if mx is None or mx == 0:
        dead.append((c, 0.0))
    elif (s == 0).sum() / df.height > 0.95:
        dead.append((c, float(mx)))
for c, mx in dead:
    print(f"  {c:<26} 最大值 {mx:>8.1f}")
print(f"  （共 {len(dead)} 欄，佔機組欄位 {len(dead) / len(unit_cols) * 100:.0f}%）")

# %% [markdown]
# ## 8. 圖表輸出

# %%
_plot_reserve_series(df)
_plot_demand_split(df)
_plot_reserve_hist(df)
_plot_renewable(df)
print(f"  已寫入 {FIGDIR}")
