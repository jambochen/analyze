import numpy as np
import polars as pl
import statsmodels.api as sm

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
    .with_columns(pl.col("sched_dep_utc").dt.truncate("1mo").alias("ym"))
    .drop_nulls(wx)
    .filter(pl.col("y2_share").is_not_null())
)

# Y2 的 1.0 是「整段延誤全掛天氣」，是二元決策的結果而非連續分布的邊界堆積。
# 拆成兩個決策分別估：是否整段全掛、若非整段則掛多少。
b = b.with_columns(
    (pl.col("y2_share") >= 1.0).alias("is_one"),
    pl.when(pl.col("y2_share") < 1.0).then(pl.col("y2_share")).alias("partial"),
)
print("Y2 有效", b.height, "其中 1.0 佔", round(b["is_one"].mean(), 4))
print(
    b.group_by("operating_carrier")
    .agg(pl.len().alias("n"), pl.col("is_one").mean().alias("p_one"), pl.col("partial").mean().alias("mean_partial"))
    .filter(pl.col("n") >= 500)
    .sort("p_one", descending=True)
)

cell = b.group_by("origin_iata", "ym").agg(pl.col("operating_carrier").n_unique().alias("nc"))
keep = cell.filter(pl.col("nc") >= 2).select("origin_iata", "ym")
b = b.join(keep, on=["origin_iata", "ym"])
carriers = sorted(c for c in b["operating_carrier"].unique().to_list() if c != "DL")
cell_id = b.select(pl.struct("origin_iata", "ym").rank("dense").alias("i"))["i"].to_numpy() - 1

Xn = b.select(wx).to_numpy().astype(float)
Xn = (Xn - Xn.mean(0)) / (Xn.std(0) + 1e-9)
X = np.column_stack([Xn] + [(b["operating_carrier"] == c).to_numpy().astype(float) for c in carriers])
names = wx + [f"carrier_{c}" for c in carriers]


def demean_fit(y, mask, label):
    g = cell_id[mask]
    dm = pl.DataFrame(X[mask], schema=names).with_columns(pl.Series("g", g), pl.Series("y", y))
    dd = dm.select([(pl.col(c) - pl.col(c).mean().over("g")).alias(c) for c in names + ["y"]])
    M = dd.select(names).to_numpy()
    nz = M.std(0) > 1e-10
    M, nm = M[:, nz], [n for n, k in zip(names, nz) if k]
    r = sm.OLS(dd["y"].to_numpy(), M).fit(cov_type="cluster", cov_kwds={"groups": g})
    assert not np.isnan(r.bse).any(), f"{label} 標準誤含 NaN"
    out = pl.DataFrame({"term": nm, "coef": r.params, "se": r.bse}).with_columns(
        (pl.col("coef") - 1.96 * pl.col("se")).alias("lo"), (pl.col("coef") + 1.96 * pl.col("se")).alias("hi")
    )
    print(f"\n=== {label} n={mask.sum()} cells={len(set(g.tolist()))}")
    print(out.filter(pl.col("term").str.starts_with("carrier_")).sort("coef", descending=True))
    out.write_csv(f"outputs/tables/y2_{label}.csv")
    return out


all_mask = np.ones(b.height, dtype=bool)
p1 = demean_fit(b["is_one"].to_numpy().astype(float), all_mask, "part1_is_one")

pm = b["partial"].is_not_null().to_numpy()
p2 = demean_fit(b["partial"].fill_null(0).to_numpy().astype(float)[pm], pm, "part2_partial")

# 兩部的公司效應方向是否一致。不一致即表示合併估的單一係數在混合兩種行為。
cmp = (
    p1.filter(pl.col("term").str.starts_with("carrier_"))
    .select("term", pl.col("coef").alias("coef_is_one"), pl.col("se").alias("se_is_one"))
    .join(
        p2.filter(pl.col("term").str.starts_with("carrier_")).select(
            "term", pl.col("coef").alias("coef_partial"), pl.col("se").alias("se_partial")
        ),
        on="term",
    )
    .with_columns(
        (pl.col("coef_is_one").sign() != pl.col("coef_partial").sign()).alias("方向相反"),
    )
    .sort("coef_is_one", descending=True)
)
print()
print(cmp)
print("方向相反者", cmp["方向相反"].sum(), "/", cmp.height)
cmp.write_csv("outputs/tables/y2_twopart_compare.csv")
