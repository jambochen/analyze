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

b = pl.read_parquet("data/processed/bad_weather.parquet").with_columns(
    pl.col("sched_dep_utc").dt.truncate("1mo").alias("ym")
).drop_nulls(wx)

cell = b.group_by("origin_iata", "ym").agg(
    pl.col("operating_carrier").n_unique().alias("nc"), pl.col("y1_attributed").mean().alias("cr")
)
keep = cell.filter((pl.col("nc") >= 2) & (pl.col("cr") > 0) & (pl.col("cr") < 1)).select("origin_iata", "ym")
b = b.join(keep, on=["origin_iata", "ym"])

carriers = sorted(c for c in b["operating_carrier"].unique().to_list() if c != "DL")
cell_id = b.select(pl.struct("origin_iata", "ym").rank("dense").alias("i"))["i"].to_numpy() - 1
y = b["y1_attributed"].to_numpy().astype(float)


def fit(cols, label):
    X_num = b.select(cols).to_numpy().astype(float)
    X_num = (X_num - X_num.mean(0)) / (X_num.std(0) + 1e-9)
    X = np.column_stack([X_num] + [(b["operating_carrier"] == c).to_numpy().astype(float) for c in carriers])
    names = cols + [f"carrier_{c}" for c in carriers]
    dm = pl.DataFrame(X, schema=names).with_columns(pl.Series("g", cell_id), pl.Series("y", y))
    dd = dm.select([(pl.col(c) - pl.col(c).mean().over("g")).alias(c) for c in names + ["y"]])
    r = sm.OLS(dd["y"].to_numpy(), dd.select(names).to_numpy()).fit(
        cov_type="cluster", cov_kwds={"groups": cell_id}
    )
    out = pl.DataFrame({"term": names, f"coef_{label}": r.params, f"se_{label}": r.bse})
    print(label, "R2", round(r.rsquared, 5))
    print(out.filter(pl.col("term").is_in(["wx_origin_ceiling_m", "wx_origin_sky_cover_oktas"])))
    return out


full = fit(wx, "full")
no_ceil = fit([c for c in wx if c != "wx_origin_ceiling_m"], "no_ceiling")
no_sky = fit([c for c in wx if c != "wx_origin_sky_cover_oktas"], "no_sky")

# 公司效應是交付物。若三個設定下係數幾乎不動，共線就與清單無關。
cmp = (
    full.filter(pl.col("term").str.starts_with("carrier_"))
    .join(no_ceil.filter(pl.col("term").str.starts_with("carrier_")), on="term")
    .join(no_sky.filter(pl.col("term").str.starts_with("carrier_")), on="term")
    .with_columns(
        (pl.col("coef_no_ceiling") - pl.col("coef_full")).abs().alias("d_ceil"),
        (pl.col("coef_no_sky") - pl.col("coef_full")).abs().alias("d_sky"),
    )
    .sort("coef_full", descending=True)
)
print()
print(cmp.select("term", "coef_full", "se_full", "d_ceil", "d_sky"))
print()
print("公司效應最大變動 去雲冪", round(cmp["d_ceil"].max(), 5), "去雲量", round(cmp["d_sky"].max(), 5))
print("對照：公司效應標準誤中位數", round(cmp["se_full"].median(), 5))
cmp.write_csv("outputs/tables/collinear_check.csv")
