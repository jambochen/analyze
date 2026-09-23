import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import statsmodels.api as sm
from scipy import stats


wx = [
    "wx_origin_visibility_m", "wx_origin_ceiling_m", "wx_origin_wind_speed_ms", "wx_origin_wind_gust_ms",
    "wx_origin_precip_1h_mm", "wx_origin_snow_depth_cm", "wx_origin_temp_c", "wx_origin_sky_cover_oktas",
    "wx_origin_flight_category_ordinal", "wx_dest_flight_category_ordinal", "wx_dest_visibility_m",
    "wx_dest_ceiling_m", "wx_dest_wind_speed_ms", "wx_dest_precip_1h_mm",
    "wx_origin_thunderstorm", "wx_origin_snow", "wx_origin_freezing_precip", "wx_origin_fog",
    "wx_origin_m3_flight_category_ordinal", "wx_origin_m6_flight_category_ordinal",
    "arr_delay_min", "distance_miles", "origin_sched_departures_this_hour", "prior_leg_arr_delay_min",
]

b = (
    pl.read_parquet("data/processed/bad_weather.parquet")
    .with_columns(
        pl.col("sched_dep_utc").dt.truncate("1mo").alias("ym"),
        pl.col("sched_dep_utc").dt.year().alias("yr"),
    )
    .drop_nulls(wx)
)


def effects(d, outcome):
    cell = d.group_by("origin_iata", "ym").agg(
        pl.col("operating_carrier").n_unique().alias("nc"), pl.col(outcome).mean().alias("cr")
    )
    keep = cell.filter((pl.col("nc") >= 2) & (pl.col("cr") > 0) & (pl.col("cr") < 1)).select("origin_iata", "ym")
    d = d.join(keep, on=["origin_iata", "ym"]).filter(pl.col(outcome).is_not_null())
    cs = sorted(c for c in d["operating_carrier"].unique().to_list() if c != "DL")
    Xn = d.select(wx).to_numpy().astype(float)
    Xn = (Xn - Xn.mean(0)) / (Xn.std(0) + 1e-9)
    X = np.column_stack([Xn] + [(d["operating_carrier"] == c).to_numpy().astype(float) for c in cs])
    names = wx + [f"carrier_{c}" for c in cs]
    g = d.select(pl.struct("origin_iata", "ym").rank("dense").alias("i"))["i"].to_numpy() - 1
    # 同一機尾號跨上百個格子，格子聚類吸收不到機隊層的相關，故雙向聚類。
    tail = d.select(pl.col("tail_number").rank("dense").alias("i"))["i"].to_numpy() - 1
    dm = pl.DataFrame(X, schema=names).with_columns(
        pl.Series("g", g), pl.Series("y", d[outcome].to_numpy().astype(float))
    )
    dd = dm.select([(pl.col(c) - pl.col(c).mean().over("g")).alias(c) for c in names + ["y"]])
    M = dd.select(names).to_numpy()
    nz = M.std(0) > 1e-10
    M, nm = M[:, nz], [n for n, k in zip(names, nz) if k]
    r = sm.OLS(dd["y"].to_numpy(), M).fit(
        cov_type="cluster", cov_kwds={"groups": np.column_stack([g, tail])}
    )
    assert not np.isnan(r.bse).any()
    return (
        pl.DataFrame({"term": nm, "coef": r.params, "se": r.bse})
        .filter(pl.col("term").str.starts_with("carrier_"))
        .with_columns(pl.col("term").str.replace("carrier_", "").alias("carrier"))
        .drop("term")
    )


# hold-out 顯示係數水準會隨時期漂移，故主清單只用近期資料；前期僅用於算變化。
recent, prior = b.filter(pl.col("yr") >= 2022), b.filter(pl.col("yr") <= 2021)

parts = []
for outcome, tag in [("y1_attributed", "y1"), ("y2_share", "y2"), ("y3_wx_vs_nas", "y3")]:
    a = effects(recent, outcome).rename({"coef": f"{tag}", "se": f"{tag}_se"})
    p = effects(prior, outcome).select("carrier", pl.col("coef").alias(f"{tag}_prior"))
    parts.append(a.join(p, on="carrier", how="left").with_columns(
        (pl.col(f"{tag}") - pl.col(f"{tag}_prior")).alias(f"{tag}_chg"),
        (pl.col(f"{tag}") - 1.96 * pl.col(f"{tag}_se")).alias(f"{tag}_lo"),
        (pl.col(f"{tag}") + 1.96 * pl.col(f"{tag}_se")).alias(f"{tag}_hi"),
    ))

lst = parts[0]
for q in parts[1:]:
    lst = lst.join(q, on="carrier", how="outer_coalesce")

# 反向檢核：營運表現差與回報不誠實是兩個量。表現差但誠實者不該進清單。
perf = (
    recent.group_by("operating_carrier")
    .agg(
        pl.len().alias("n"),
        pl.col("arr_delay_min").median().alias("delay_median"),
        pl.col("y1_attributed").mean().alias("raw_y1"),
        pl.col("is_regional").mean().alias("regional"),
    )
    .rename({"operating_carrier": "carrier"})
)
lst = lst.join(perf, on="carrier", how="left")

# 第二個反向檢核：取消班的 arr_delay_min 為 null，被申報門檻排除，故稽核樣本
# 看不到它。但取消也是一次天氣歸因決策，一家公司可以在延誤欄少報天氣卻以天氣
# 為由大量取消。歸因偏低者須並列此欄，否則「以取消消化天氣」會被誤讀為誠實申報。
# 須回讀原始檔，惡劣天氣定義與 clean.py 一致。
cxl = (
    pl.scan_parquet([f"data/raw/flights-weather/flights_weather_{y}.parquet" for y in range(2022, 2025)])
    .select("operating_carrier", "cancelled", "cancellation_reason",
            "wx_origin_flight_category", "wx_dest_flight_category")
    .filter(
        pl.col("wx_origin_flight_category").is_in(["IFR", "LIFR"])
        | pl.col("wx_dest_flight_category").is_in(["IFR", "LIFR"])
    )
    .group_by("operating_carrier")
    .agg(
        pl.col("cancelled").mean().alias("cxl_rate"),
        # 分母為取消班：取消時有多少比例掛天氣。與 Y1「延誤時是否掛天氣」同形態，
        # 兩者才可並排讀。
        ((pl.col("cancellation_reason") == "Weather").sum() / pl.col("cancelled").sum()).alias("cxl_wx"),
    )
    .rename({"operating_carrier": "carrier"})
    .collect(engine="streaming")
)
lst = lst.join(cxl, on="carrier", how="left")

# 三個指標乘各家公司構成數十個同時檢定，未校正的區間會高估異常家數。
# 清單本身是排序不是檢定，但「幾項顯著」這個粗排序既然用了門檻，就須校正。
from statsmodels.stats.multitest import multipletests

for t in ["y1", "y2", "y3"]:
    z = (lst[t] / lst[f"{t}_se"]).to_numpy()
    pv = 2 * (1 - stats.norm.cdf(np.abs(z)))
    lst = lst.with_columns(pl.Series(f"{t}_p", pv))

allp = np.concatenate([lst[f"{t}_p"].to_numpy() for t in ["y1", "y2", "y3"]])
rej = multipletests(allp, alpha=0.05, method="fdr_bh")[0].reshape(3, -1)
for i, t in enumerate(["y1", "y2", "y3"]):
    lst = lst.with_columns(pl.Series(f"{t}_bh", rej[i]))

lst = lst.with_columns(
    sum(((pl.col(f"{t}_lo") > 0) & pl.col(f"{t}_bh")).cast(pl.Int8) for t in ["y1", "y2", "y3"]).alias("n_pos"),
    sum(((pl.col(f"{t}_hi") < 0) & pl.col(f"{t}_bh")).cast(pl.Int8) for t in ["y1", "y2", "y3"]).alias("n_neg"),
    sum((pl.col(f"{t}_lo") > 0).cast(pl.Int8) for t in ["y1", "y2", "y3"]).alias("n_pos_raw"),
).sort("n_pos", "y1", descending=[True, True])

cols = ["carrier", "n", "regional", "n_pos", "n_neg", "n_pos_raw",
        "y1", "y1_se", "y1_lo", "y1_hi", "y1_chg", "y2", "y2_se", "y2_lo", "y2_hi", "y2_chg",
        "y3", "y3_se", "y3_lo", "y3_hi", "y3_chg", "raw_y1", "delay_median", "cxl_rate", "cxl_wx"]
lst = lst.select(cols)
print(lst)
lst.write_csv("outputs/tables/priority_list.csv")

print("\n三項皆顯著為正")
print(lst.filter(pl.col("n_pos") == 3).select("carrier", "y1", "y2", "y3", "y1_chg", "y3_chg", "delay_median"))
print("\n表現差但未進清單者（延誤中位高於整體，卻無指標顯著為正）")
med = float(recent["arr_delay_min"].median())
print(lst.filter((pl.col("delay_median") > med) & (pl.col("n_pos") == 0)).select("carrier", "delay_median", "y1", "n_pos"))

# 歸因顯著偏低者：取消天氣歸因率高者，其偏低可能只是改用取消欄消化天氣。
print("\n歸因顯著偏低者的取消檢核")
print(lst.filter(pl.col("n_neg") > 0).select("carrier", "n_neg", "y1", "raw_y1", "cxl_rate", "cxl_wx").sort("cxl_wx", descending=True))

fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), sharey=True)
order = lst["carrier"].to_list()[::-1]
for ax, tag, title in zip(axes, ["y1", "y2", "y3"],
                          ["Y1 weather attributed", "Y2 weather share when attributed", "Y3 weather vs NAS margin"]):
    t = lst.with_columns(pl.col("carrier").replace_strict({k: i for i, k in enumerate(order)}).alias("pos"))
    ax.errorbar(t[tag], t["pos"], xerr=1.96 * t[f"{tag}_se"], fmt="o", ms=4, lw=1)
    ax.axvline(0, ls="--", lw=0.8, color="gray")
    ax.set_title(title, fontsize=10)
axes[0].set_yticks(range(len(order)))
axes[0].set_yticklabels(order, fontsize=8)
fig.supxlabel("Effect vs DL (2022-2024, within airport x month)")
fig.tight_layout()
fig.savefig("outputs/figures/priority_list.png", dpi=150)
