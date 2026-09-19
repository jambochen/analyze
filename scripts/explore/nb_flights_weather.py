"""候選「美國航班延誤 × 實測氣象」的篩選用檢視（notebook 版）。

與 `flights-weather.py` 內容相同，只改為 `# %%` 分格以便轉 notebook：

    jupytext --to notebook scripts/explore/nb_flights_weather.py

僅供判斷此候選是否適合作為選題，非分析結果。
範圍限於 CLAUDE.md 允許的篩選檢視：筆數、欄位型別、缺失率、唯一值數、
描述統計、單變數分布、時間序列折線圖。相關係數矩陣另行標註。

不配適任何模型。分組平均只用於確認資料結構是否存在可用的變異，
不構成處理效應估計。
"""

# %% [markdown]
# # 美國航班延誤 × 實測氣象：篩選檢視
#
# 資料源：https://www.kaggle.com/datasets/arimancy/observed-weather-us-flights
#
# 研究問題：以實際觀測氣象為基準，哪些航空公司、機場與航線系統性地
# 少報天氣延誤？
#
# 檢視使用 24 個月入門檔（1,406 萬列、1.1 GB），schema 與完整版
# （4,280 萬列、6.1 GB）相同。

# %%
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import polars as pl

for family in ("Noto Sans CJK TC", "Noto Sans CJK JP", "WenQuanYi Zen Hei"):
    if any(family in f.name for f in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = family
        break
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path.cwd()
while not (ROOT / "data" / "raw").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent

SRC = ROOT / "data" / "raw" / "flights-weather" / "flights_weather_recent_24mo.parquet"
AIRPORTS = ROOT / "data" / "raw" / "flights-weather" / "airports_reference.parquet"
FIGDIR = ROOT / "outputs" / "figures"


def prefix_of(col: str) -> str:
    for p in ("wx_", "oracle_", "reported_"):
        if col.startswith(p):
            return p
    return "(其他)"


lf = pl.scan_parquet(SRC)
schema = lf.collect_schema()
cols = schema.names()

# %% [markdown]
# ## 1. 規模與欄位

# %%
n_rows = lf.select(pl.len()).collect().item()
print(f"列數：{n_rows:,}")
print(f"欄數：{len(cols)}")
print(f"檔案大小：{SRC.stat().st_size / 1e9:.2f} GB")

# %% [markdown]
# ## 2. 欄位前綴分布（資料集的洩漏防護設計）
#
# 這是這份資料最值得注意的設計：欄位前綴直接標示了能不能安全使用。
#
# - `wx_`：航班資訊截止點**之前**的觀測氣象，預測時可安全使用
# - `oracle_`：航班自身事件時刻的氣象，**用於預測即構成洩漏**
# - `reported_`：航空公司自報的延誤成因，刻意保留但明確標示不可信

# %%
groups: dict[str, list[str]] = {}
for c in cols:
    groups.setdefault(prefix_of(c), []).append(c)

for p, members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
    print(f"\n{p}（{len(members)} 欄）")
    for c in members:
        print(f"  {c:46s} {schema[c]}")

# %% [markdown]
# ## 3. 缺失率與唯一值數
#
# 唯一值數對 1,400 萬列成本過高，只取類別型欄位。

# %%
nulls = lf.select([pl.col(c).null_count().alias(c) for c in cols]).collect().row(0)

cat_cols = [c for c in cols if schema[c] in (pl.Utf8, pl.Categorical, pl.Boolean)]
uniques = lf.select([pl.col(c).n_unique().alias(c) for c in cat_cols]).collect().row(0)
uniq_map = dict(zip(cat_cols, uniques))

print(f"{'欄位':46s} {'缺失率':>9s} {'唯一值':>12s}")
for c, nn in sorted(zip(cols, nulls), key=lambda kv: -kv[1]):
    u = uniq_map.get(c)
    print(f"{c:46s} {nn / n_rows:9.2%} {('' if u is None else f'{u:,}'):>12s}")

# %% [markdown]
# 缺失最高的幾欄是 `reported_` 系列與轉機相關欄位——前者只有航空公司
# 實際回報時才有值，後者只有存在前段航班時才有值，皆屬結構性缺失。
#
# 氣象欄位的缺失以 `obs_missing` 標記並留空，**作者不做任何插補**。

# %% [markdown]
# ## 4. 主要數值欄位的描述統計
#
# 211 欄橫排會爆版，改為每欄一列。

# %%
num_cols = [c for c in cols if schema[c].is_numeric() and not c.endswith("_id")]

stats = lf.select([
    pl.struct(
        pl.col(c).mean().alias("mean"),
        pl.col(c).std().alias("std"),
        pl.col(c).min().alias("min"),
        pl.col(c).quantile(0.25).alias("q25"),
        pl.col(c).median().alias("median"),
        pl.col(c).quantile(0.75).alias("q75"),
        pl.col(c).max().alias("max"),
    ).alias(c)
    for c in num_cols
]).collect().row(0, named=True)

print(f"{'欄位':44s} {'平均':>12s}{'標準差':>12s}{'最小':>12s}"
      f"{'q25':>12s}{'中位':>12s}{'q75':>12s}{'最大':>12s}")
for c, st in stats.items():
    vals = "".join(
        f"{v:>12.2f}" if isinstance(v, (int, float)) else f"{'':>12s}"
        for v in (st["mean"], st["std"], st["min"], st["q25"],
                  st["median"], st["q75"], st["max"])
    )
    print(f"{c:44s} {vals}")

# %% [markdown]
# ## 5. 時間跨度與每月航班量

# %%
date_col = next(
    (c for c in ("flight_date_local_origin", "sched_dep_utc") if c in cols), None
)

monthly = (
    lf.select(pl.col(date_col).cast(pl.Date).dt.truncate("1mo").alias("ym"))
    .group_by("ym").len().sort("ym").collect()
)
print(f"日期欄位：{date_col}")
print(f"起訖：{monthly['ym'].min()} 至 {monthly['ym'].max()}")
with pl.Config(tbl_rows=-1):
    print(monthly)

# %%
FIGDIR.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(monthly["ym"], monthly["len"])
ax.set_title("每月航班量（僅供篩選，非分析結果）")
ax.set_ylabel("航班數")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIGDIR / "explore_flights_monthly.png", dpi=120)
plt.show()

# %% [markdown]
# ## 6. 結構可衍生性檢查
#
# 這份資料的複雜度不只來自 211 個欄位，也來自單位之間的關係：
# 機尾號可串接當日航段（追蹤結構），起降機場構成有向網路。

# %%
for label, col in (
    ("機尾號（追蹤結構）", "tail_number"),
    ("航空公司", "operating_carrier"),
    ("起點機場", "origin_iata"),
    ("終點機場", "dest_iata"),
):
    if col in cols:
        u = uniq_map.get(col)
        print(f"{label:20s} {col:24s} 唯一值 {u:,}" if isinstance(u, int)
              else f"{label:20s} {col:24s}")

n_routes = lf.select(
    pl.concat_str([pl.col("origin_iata"), pl.col("dest_iata")], separator="-").n_unique()
).collect().item()
print(f"航線（origin-dest 組合）唯一值：{n_routes:,}")

legs = (
    lf.filter(pl.col("tail_number").is_not_null())
    .group_by("tail_number").len().collect()["len"]
)
print(f"每架機尾號的航段數：中位數 {legs.median():,.0f}、"
      f"平均 {legs.mean():,.1f}、最大 {legs.max():,}")

# %% [markdown]
# ## 7. 延誤分布（單變數）

# %%
delay_col = next((c for c in ("arr_delay_min", "dep_delay_min") if c in cols), None)
s = lf.select(pl.col(delay_col)).collect()[delay_col].drop_nulls()

print(f"欄位：{delay_col}")
for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99):
    print(f"  q{q:>5.2f}  {s.quantile(q):8.1f} 分")

# %% [markdown]
# **結果變數強右偏**：中位數接近 0 甚至為負（提早抵達），但 99 分位遠在
# 高處。正式分析時須考慮轉換或改用適合長尾的模型。

# %%
# 長尾極端值會壓扁圖形，截在 1% 與 99% 分位之間看主體形狀。
fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(s.filter(s.is_between(s.quantile(0.01), s.quantile(0.99))), bins=120)
ax.set_title(f"{delay_col} 分布（僅供篩選，非分析結果）")
ax.set_xlabel("分鐘")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIGDIR / "explore_flights_delay_hist.png", dpi=120)
plt.show()

# %% [markdown]
# ## 8. 飛航天氣等級的樣本量
#
# 此表僅確認各等級樣本量是否足夠，**不構成處理效應估計**。

# %%
cat_col = next((c for c in cols if "flight_category" in c or "flight_cat" in c), None)
tab = (
    lf.group_by(cat_col)
    .agg(pl.len().alias("n"), pl.col(delay_col).mean().alias("平均延誤分"))
    .sort("n", descending=True).collect()
)
print(f"分組欄位：{cat_col}")
with pl.Config(tbl_rows=-1):
    print(tab)

# %% [markdown]
# VFR → MVFR → IFR → LIFR 是目視飛行到儀器飛行的惡化順序。
# 平均延誤隨天氣惡化**單調上升**，且最差的 LIFR 仍有 34 萬班，
# 樣本量足夠。
#
# 這確認了「實際氣象 vs 自我回報」的落差算得出來。
# **僅供篩選，非分析結果。**

# %% [markdown]
# ## 9. 回報偏誤素材的可用性
#
# 這是本候選的核心——`reported_` 欄位是航空公司自報的延誤成因，
# 要拿它跟實測氣象對照。先確認這些欄位真的有值。

# %%
rep_cols = [c for c in cols if c.startswith("reported_")]
print(f"reported_ 欄位共 {len(rep_cols)} 個：\n  {rep_cols}\n")

nonzero = lf.select([
    (pl.col(c) > 0).sum().alias(c) for c in rep_cols if schema[c].is_numeric()
]).collect()
print("各 reported_ 欄位大於零的列數：")
with pl.Config(tbl_cols=-1, tbl_width_chars=200):
    print(nonzero)

# %% [markdown]
# ## 10. 機場對照表
#
# 作者說這張表「不存在於任何維護中的公開資料」，是這份資料集的關鍵貢獻，
# 也是本候選第三問（我們有什麼他們拿不到的）的答案。

# %%
if AIRPORTS.exists():
    ar = pl.read_parquet(AIRPORTS)
    print(f"列數 {len(ar):,}、欄數 {ar.width}\n")
    print("全部欄位：")
    for c in ar.columns:
        print(f"  {c}")

    # 完整寬度會爆版，只看對照的關鍵欄位
    key = [c for c in (
        "iata_code", "icao_code", "isd_station_id", "isd_station_name",
        "station_distance_km", "station_match_method", "elevation_delta_m",
    ) if c in ar.columns]
    print("\n對照關鍵欄位（前 8 列）：")
    with pl.Config(tbl_cols=-1, tbl_width_chars=160, fmt_str_lengths=30):
        print(ar.select(key).head(8))

    print("\n比對方法的分布：")
    print(ar.group_by("station_match_method").agg(pl.len().alias("機場數"))
          .sort("機場數", descending=True))

    d = ar["station_distance_km"].drop_nulls()
    print(f"\n機場到測站的距離：中位數 {d.median():.1f} km、"
          f"q90 {d.quantile(0.9):.1f} km、最大 {d.max():.1f} km")

    # 作者自己標出了需要留意的比對
    flags = [c for c in ar.columns if c.startswith("flag_")]
    print("\n作者標記的品質旗標：")
    for f in flags:
        print(f"  {f:34s} {ar[f].sum():>4} 個機場")

# %% [markdown]
# ## 小結
#
# - 欄位前綴系統把洩漏風險寫進命名，實測確認三組前綴各有 106／54／5 欄
# - 缺失以結構性為主，氣象缺值以 `obs_missing` 標記且不插補
# - 結構可衍生：機尾號追蹤、機場網路、航線組合皆可建構
# - 天氣等級的樣本量足夠，延誤隨天氣惡化單調上升
# - `reported_` 欄位有值，回報偏誤的對照做得起來
# - 機場對測站的對照表存在，這是第三問的答案
#
# **以上全部為篩選用檢視，非分析結果。**
