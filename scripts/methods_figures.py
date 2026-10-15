# 方法報告的結果形狀圖。描述原始申報，不將公司間落差解讀為調整後效果。
import itertools

import matplotlib as mpl
import matplotlib.pyplot as plt
import polars as pl
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

import simplex

mpl.rcParams["font.sans-serif"] = ["DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["font.size"] = 11
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.right"] = False

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREY = "#8a8a86"

cols = {
    "carrier": "reported_carrier_delay_min",
    "weather": "reported_weather_delay_min",
    "NAS": "reported_nas_delay_min",
    "late-aircraft": "reported_late_aircraft_delay_min",
}
rep5 = {**cols, "security": "reported_security_delay_min"}
# 全樣本逾七百萬列，整檔讀入會佔去大半記憶體，只取用得到的欄位。
b = pl.read_parquet("data/processed/audit_clean.parquet", columns=list(dict.fromkeys(
    [*rep5.values(), "operating_carrier", "origin_iata", "dest_iata", "sched_dep_utc", *simplex.WX_UNIT]
)))
pl.DataFrame({"item": ["航班數", "執飛航空公司"],
              "n": [b.height, b["operating_carrier"].n_unique()]}).write_csv("outputs/tables/methods_sample.csv")

# 二元結果只畫公司別平均；航班層的 0/1 直方圖不顯示公司間差異。
y1 = (
    b.group_by("operating_carrier")
    .agg(pl.len().alias("n"), (pl.col(cols["weather"]) > 0).mean().alias("rate"))
    .filter(pl.col("n") > 5000)
    .sort("rate")
)
y1.write_csv("outputs/tables/methods_y1_rate.csv")
fig, ax = plt.subplots(figsize=(7.1, 4.8))
ax.barh(y1["operating_carrier"].to_list(), y1["rate"].to_list(), color=BLUE, height=0.72)
ax.set_xlabel("Share of flights with positive weather minutes")
ax.set_ylabel("Operating carrier")
ax.set_xlim(0, max(y1["rate"]) * 1.10)
ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))
ax.grid(axis="x", color="#e4e3df", linewidth=0.7)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("outputs/figures/methods_y1_rate.png", dpi=200)
plt.close(fig)

# 四欄分鐘數的零為有效申報。單元層的零才是 Poisson 模型直接使用的資料形狀；
# 航班層另呈現為何聚合後仍須容許零值。與模型一致，先排除四欄總量為零者。
tot = b.select([pl.col(c).sum().alias(k) for k, c in rep5.items()]).row(0, named=True)
cause_mix = pl.DataFrame({"cause": list(tot), "min_share": [v / sum(tot.values()) for v in tot.values()]}).sort("min_share")
cause_mix.write_csv("outputs/tables/methods_cause_mix.csv")
fig, ax = plt.subplots(figsize=(6.1, 4.0))
y = list(range(cause_mix.height))
ax.barh(y, cause_mix["min_share"].to_list(),
        color=BLUE, height=0.66)
ax.set_yticks(y, cause_mix["cause"].to_list())
ax.set_xlabel("Share of reported delay minutes")
ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))
ax.grid(axis="x", color="#e4e3df", linewidth=0.7)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("outputs/figures/methods_cause_mix.png", dpi=200)
plt.close(fig)

d = b.with_columns(pl.sum_horizontal(list(cols.values())).alias("total4"))
d = d.filter(pl.col("total4") > 0).drop_nulls(simplex.WX_UNIT).with_columns(
    pl.sum_horizontal([(pl.col(c) > 0).cast(pl.Int8) for c in cols.values()]).alias("active")
)
flight = d.group_by("active").agg(pl.len().alias("n")).sort("active")
flight = flight.with_columns((pl.col("n") / d.height).alias("share"))
flight.write_csv("outputs/tables/methods_active_causes.csv")

# 與模型的單元一致：航線 × 年月 × 公司。單元比機場 × 年月 × 公司細，零更多；
# 用較粗的單元會低估模型實際面對的零。
unit = d.group_by("origin_iata", "dest_iata", pl.col("sched_dep_utc").dt.truncate("1mo").alias("ym"),
                  "operating_carrier").agg([pl.col(c).sum().alias(k) for k, c in cols.items()])
# any：至少一欄為零的單元，即要求四欄皆正的做法必須丟掉的部分。
zeros = pl.DataFrame({
    "cause": [*cols, "any"],
    "zero_share": [*[(unit[k] == 0).mean() for k in cols],
                   unit.select(pl.min_horizontal(list(cols)) == 0).to_series().mean()],
    "n_units": [unit.height] * (len(cols) + 1),
})
zeros.write_csv("outputs/tables/methods_unit_zeros.csv")

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), gridspec_kw={"width_ratios": [0.9, 1.1]})
axes[0].bar(flight["active"].to_list(), flight["share"].to_list(), color=BLUE, width=0.65)
axes[0].set_xticks([1, 2, 3, 4])
axes[0].set_xlabel("Number of causes with positive minutes")
axes[0].set_ylabel("Share of flights")
axes[0].yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))
zero_of = dict(zip(zeros["cause"], zeros["zero_share"]))
shown = ["any", *reversed(cols)]
axes[1].barh(["At least one\nof the four", *shown[1:]], [zero_of[c] for c in shown],
             color=[GREY, *[ORANGE] * len(cols)], height=0.66)
axes[1].set_xlabel("Share of units with zero minutes")
axes[1].xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))
for ax in axes:
    ax.grid(axis="y" if ax is axes[0] else "x", color="#e4e3df", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(color=GREY)
fig.tight_layout(w_pad=2.0)
fig.savefig("outputs/figures/methods_simplex_zeros.png", dpi=200)
plt.close(fig)

# 天氣欄的逐班份額在 0 與 1 各有 point mass。兩格分別對應兩部模型的兩個問題；
# 合成一張直方圖時 0 的那根會把其餘壓平。
ws = d.select((pl.col(cols["weather"]) / pl.col("total4")).alias("s"))["s"]
pos = ws.filter(ws > 0)
pl.DataFrame({"item": ["zero", "positive", "one_given_positive"],
              "share": [(ws == 0).mean(), (ws > 0).mean(), (pos == 1).mean()],
              "n": [ws.len(), ws.len(), pos.len()]}).write_csv("outputs/tables/methods_weather_share.csv")
# 只用於投影片：圖占頁寬近九成，字級須放大到投影後仍讀得到。
mpl.rcParams["font.size"] = 15
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.1), gridspec_kw={"width_ratios": [0.6, 1.4]})
axes[0].bar(["Zero", "Positive"], [(ws == 0).mean(), (ws > 0).mean()], color=[GREY, BLUE], width=0.6)
axes[0].set_ylabel("Share of flights")
axes[0].set_xlabel("Weather minutes")
axes[0].yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, decimals=0))
axes[0].set_title("Any weather minutes?", fontsize=15, loc="left")
axes[1].hist(pos.to_numpy(), bins=40, range=(0, 1), weights=[1 / pos.len()] * pos.len(), color=BLUE)
axes[1].set_xlabel("Weather share of reported minutes (positive only)")
axes[1].yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1, decimals=0))
axes[1].set_title("How large a share?", fontsize=15, loc="left")
for ax in axes:
    ax.grid(axis="y", color="#e4e3df", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(color=GREY)
fig.tight_layout(w_pad=1.5)
fig.savefig("outputs/figures/methods_weather_share.png", dpi=200)
plt.close(fig)
mpl.rcParams["font.size"] = 11

# 以完整稽核樣本描述起點機場 × 年月的共同營運，不混入四欄模型的額外篩選。
fit_base = b.with_columns(pl.col("sched_dep_utc").dt.truncate("1mo").alias("ym"))
strata = fit_base.group_by("origin_iata", "ym").agg(
    pl.col("operating_carrier").n_unique().alias("k"), pl.len().alias("flights")
)
overlap = (
    strata.with_columns(pl.when(pl.col("k") >= 6).then(pl.lit("6+"))
                        .otherwise(pl.col("k").cast(pl.Utf8)).alias("k_group"))
    .group_by("k_group").agg(pl.len().alias("strata"), pl.col("flights").sum())
    .with_columns((pl.col("strata") / strata.height).alias("strata_share"),
                  (pl.col("flights") / strata["flights"].sum()).alias("flight_share"))
)
overlap.write_csv("outputs/tables/methods_overlap.csv")
ordered = ["1", "2", "3", "4", "5", "6+"]
overlap_rows = {row["k_group"]: row for row in overlap.to_dicts()}
x = list(range(len(ordered)))
fig, ax = plt.subplots(figsize=(7.8, 4.3))
ax.bar([v - 0.19 for v in x], [overlap_rows[v]["strata_share"] for v in ordered],
       width=0.38, color=BLUE, label="Share of groups")
ax.bar([v + 0.19 for v in x], [overlap_rows[v]["flight_share"] for v in ordered],
       width=0.38, color=ORANGE, label="Share of flights")
ax.set_xticks(x, ordered)
ax.set_xlabel("Operating carriers per airport-month group")
ax.set_ylabel("Share")
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))
ax.legend(frameon=False)
ax.grid(axis="y", color="#e4e3df", linewidth=0.7)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("outputs/figures/methods_overlap.png", dpi=200)
plt.close(fig)

# 機場月份只證明同場，航線月份才證明同向航線上的共同營運；兩者皆未檢查氣象重疊。
route_groups = fit_base.group_by("origin_iata", "dest_iata", "ym").agg(
    pl.col("operating_carrier").n_unique().alias("k"), pl.len().alias("flights")
)
# 只有一家公司的組合不因間接比較而變得可用；間接比較增加的是可比的公司對。
# 以公司為節點、曾在同一組合出現為邊：direct 是有邊的公司對，linked 是落在
# 同一連通成分的公司對。
carriers = sorted(fit_base["operating_carrier"].unique())
pairs = len(carriers) * (len(carriers) - 1) // 2
levels = []
for scope, keys, g in [("airport-month", ["origin_iata", "ym"], strata),
                       ("route-month", ["origin_iata", "dest_iata", "ym"], route_groups)]:
    direct = set()
    for cs in fit_base.group_by(keys).agg(pl.col("operating_carrier").unique().alias("cs"))["cs"].to_list():
        direct.update(itertools.combinations(sorted(cs), 2))
    i, j = zip(*[(carriers.index(a), carriers.index(c)) for a, c in direct])
    _, lab = connected_components(coo_matrix(([1] * len(i), (i, j)), shape=(len(carriers),) * 2), directed=False)
    levels.append({
        "scope": scope,
        "groups": g.height,
        "multi_carrier_groups": g.filter(pl.col("k") >= 2).height,
        "flights": g["flights"].sum(),
        "multi_carrier_flights": g.filter(pl.col("k") >= 2)["flights"].sum(),
        "carrier_pairs": pairs,
        "direct_pairs": len(direct),
        "linked_pairs": sum(int(lab[x] == lab[y]) for x, y in itertools.combinations(range(len(carriers)), 2)),
    })
overlap_levels = pl.DataFrame(levels).with_columns(
    (pl.col("multi_carrier_groups") / pl.col("groups")).alias("group_share"),
    (pl.col("multi_carrier_flights") / pl.col("flights")).alias("flight_share"),
)
# 既有文件依欄位位置讀 group_share 與 flight_share，新欄放在其後。
overlap_levels = overlap_levels.select(
    "scope", "groups", "multi_carrier_groups", "flights", "multi_carrier_flights", "group_share", "flight_share",
    "carrier_pairs", "direct_pairs", "linked_pairs",
)
overlap_levels.write_csv("outputs/tables/methods_overlap_levels.csv")

fig, ax = plt.subplots(figsize=(8.4, 3.3))
ys = [1, 0]
series = [
    (0.27, overlap_levels["flight_share"].to_list(), ORANGE, "Flights in groups with ≥2 carriers", None),
    (0.0, (overlap_levels["direct_pairs"] / overlap_levels["carrier_pairs"]).to_list(), BLUE,
     "Carrier pairs sharing a group", overlap_levels["direct_pairs"].to_list()),
    (-0.27, (overlap_levels["linked_pairs"] / overlap_levels["carrier_pairs"]).to_list(), GREY,
     "Carrier pairs linked through other carriers", overlap_levels["linked_pairs"].to_list()),
]
for off, shares, color, label, counts in series:
    ax.barh([v + off for v in ys], shares, height=0.24, color=color)
    for k, (y, value) in enumerate(zip(ys, shares)):
        text = f"{value:.0%}" if counts is None else f"{counts[k]} / {pairs}"
        ax.text(value + 0.015, y + off, text, va="center")
        # 三條長條的分母不同，名稱直接寫在長條內，不靠圖例對照。
        ax.text(0.012, y + off, label, va="center", color="white", fontsize=9.5)
ax.set_yticks(ys, ["Origin airport × month", "Route × month"])
ax.set_xlim(0, 1.18)
ax.set_xlabel("Share of flights or of carrier pairs")
ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1))
ax.grid(axis="x", color="#e4e3df", linewidth=0.7)
ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig("outputs/figures/methods_overlap_levels.png", dpi=200)
plt.close(fig)

# 既有估計以 DL 為基準；此圖展示對比及區間，不代替重新中心化後的正式清單。
weather = (pl.read_csv("outputs/tables/attribution_simplex.csv")
           .filter(pl.col("term").str.contains(r"^x_.*_weather$"))
           .with_columns(pl.col("term").str.extract(r"^x_(.*)_weather$", 1).alias("carrier"))
           .select("carrier", pl.col("Estimate").alias("estimate"),
                   pl.col("2.5%").alias("lo"), pl.col("97.5%").alias("hi")))
nas_boundary = (pl.read_csv("outputs/tables/attribution_boundary.csv")
                .filter(pl.col("contrast") == "weather-nas")
                .select("carrier", "estimate", "lo", "hi"))
order = weather.sort("estimate")["carrier"].to_list()
fig, axes = plt.subplots(1, 2, figsize=(12.0, 7.2), sharey=True)
for ax, data, title, color in [
    (axes[0], weather, "Weather vs carrier", BLUE),
    (axes[1], nas_boundary, "Weather vs NAS", ORANGE),
]:
    rows = {row["carrier"]: row for row in data.to_dicts()}
    for y, carrier in enumerate(order):
        row = rows[carrier]
        ax.hlines(y, row["lo"], row["hi"], color=color, linewidth=1.6)
        ax.plot(row["estimate"], y, "o", color=color, markersize=4.5)
    ax.axvline(0, color=GREY, linewidth=1, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Log ratio contrast vs DL")
    ax.set_yticks(range(len(order)), order)
    ax.grid(axis="x", color="#e4e3df", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_ylim(-0.7, len(order) - 0.3)
fig.tight_layout(w_pad=2.0)
fig.savefig("outputs/figures/methods_cause_contrasts.png", dpi=200)
plt.close(fig)
