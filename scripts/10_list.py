import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import statsmodels.api as sm

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK TC", "Noto Sans CJK SC", "Noto Sans CJK JP"]
plt.rcParams["axes.unicode_minus"] = False

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
    dm = pl.DataFrame(X, schema=names).with_columns(
        pl.Series("g", g), pl.Series("y", d[outcome].to_numpy().astype(float))
    )
    dd = dm.select([(pl.col(c) - pl.col(c).mean().over("g")).alias(c) for c in names + ["y"]])
    M = dd.select(names).to_numpy()
    nz = M.std(0) > 1e-10
    M, nm = M[:, nz], [n for n, k in zip(names, nz) if k]
    r = sm.OLS(dd["y"].to_numpy(), M).fit(cov_type="cluster", cov_kwds={"groups": g})
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

# 三個指標問的是不同問題，不合成單一分數。以「幾項顯著為正」作為粗排序，
# 並保留各指標原值供判讀。
lst = lst.with_columns(
    sum((pl.col(f"{t}_lo") > 0).cast(pl.Int8) for t in ["y1", "y2", "y3"]).alias("n_pos"),
    sum((pl.col(f"{t}_hi") < 0).cast(pl.Int8) for t in ["y1", "y2", "y3"]).alias("n_neg"),
).sort("n_pos", "y1", descending=[True, True])

cols = ["carrier", "n", "regional", "n_pos", "n_neg",
        "y1", "y1_se", "y1_lo", "y1_hi", "y1_chg", "y2", "y2_se", "y2_lo", "y2_hi", "y2_chg",
        "y3", "y3_se", "y3_lo", "y3_hi", "y3_chg", "raw_y1", "delay_median"]
lst = lst.select(cols)
print(lst)
lst.write_csv("outputs/tables/13_priority_list.csv")

print("\n三項皆顯著為正")
print(lst.filter(pl.col("n_pos") == 3).select("carrier", "y1", "y2", "y3", "y1_chg", "y3_chg", "delay_median"))
print("\n表現差但未進清單者（延誤中位高於整體，卻無指標顯著為正）")
med = float(recent["arr_delay_min"].median())
print(lst.filter((pl.col("delay_median") > med) & (pl.col("n_pos") == 0)).select("carrier", "delay_median", "y1", "n_pos"))

fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), sharey=True)
order = lst["carrier"].to_list()[::-1]
for ax, tag, title in zip(axes, ["y1", "y2", "y3"],
                          ["Y1 是否歸因天氣", "Y2 歸因時掛多少比例", "Y3 天氣 vs NAS 邊界"]):
    t = lst.with_columns(pl.col("carrier").replace_strict({k: i for i, k in enumerate(order)}).alias("pos"))
    ax.errorbar(t[tag], t["pos"], xerr=1.96 * t[f"{tag}_se"], fmt="o", ms=4, lw=1)
    ax.axvline(0, ls="--", lw=0.8, color="gray")
    ax.set_title(title, fontsize=10)
axes[0].set_yticks(range(len(order)))
axes[0].set_yticklabels(order, fontsize=8)
fig.supxlabel("相對 DL 的效應（2022-2024，機場 × 年月內比較）")
fig.tight_layout()
fig.savefig("outputs/figures/13_priority_list.png", dpi=150)
