import numpy as np
import polars as pl
import statsmodels.api as sm

ords = [
    "wx_origin_flight_category_ordinal", "wx_dest_flight_category_ordinal",
    "wx_origin_m3_flight_category_ordinal", "wx_origin_m6_flight_category_ordinal",
]
wx = [
    "wx_origin_visibility_m", "wx_origin_ceiling_m", "wx_origin_wind_speed_ms", "wx_origin_wind_gust_ms",
    "wx_origin_precip_1h_mm", "wx_origin_snow_depth_cm", "wx_origin_temp_c", "wx_origin_sky_cover_oktas",
    *ords,
    "wx_dest_visibility_m", "wx_dest_ceiling_m", "wx_dest_wind_speed_ms", "wx_dest_precip_1h_mm",
    "wx_origin_thunderstorm", "wx_origin_snow", "wx_origin_freezing_precip", "wx_origin_fog",
    "arr_delay_min", "distance_miles", "origin_sched_departures_this_hour", "prior_leg_arr_delay_min",
]

b = (
    pl.read_parquet("data/processed/bad_weather.parquet")
    .with_columns(pl.col("sched_dep_utc").dt.truncate("1mo").alias("ym"))
    .drop_nulls(wx)
)
cell = b.group_by("origin_iata", "ym").agg(
    pl.col("operating_carrier").n_unique().alias("nc"), pl.col("y1_attributed").mean().alias("cr")
)
keep = cell.filter((pl.col("nc") >= 2) & (pl.col("cr") > 0) & (pl.col("cr") < 1)).select("origin_iata", "ym")
b = b.join(keep, on=["origin_iata", "ym"])
carriers = sorted(c for c in b["operating_carrier"].unique().to_list() if c != "DL")
cell_id = b.select(pl.struct("origin_iata", "ym").rank("dense").alias("i"))["i"].to_numpy() - 1
y = b["y1_attributed"].to_numpy().astype(float)
car = [(b["operating_carrier"] == c).to_numpy().astype(float) for c in carriers]


def fit(num, dummies, label):
    Xn = b.select(num).to_numpy().astype(float)
    Xn = (Xn - Xn.mean(0)) / (Xn.std(0) + 1e-9)
    cols, names = [Xn], list(num)
    for c, lv in dummies:
        # 以 0（VFR，最佳天氣）為基準，其餘三級各自估計，不假設等距。
        for v in [1.0, 2.0, 3.0]:
            cols.append((b[c] == v).to_numpy().astype(float).reshape(-1, 1))
            names.append(f"{c}_lv{int(v)}")
    for c, v in zip(carriers, car):
        cols.append(v.reshape(-1, 1))
        names.append(f"carrier_{c}")
    X = np.column_stack(cols)
    dm = pl.DataFrame(X, schema=names).with_columns(pl.Series("g", cell_id), pl.Series("y", y))
    dd = dm.select([(pl.col(c) - pl.col(c).mean().over("g")).alias(c) for c in names + ["y"]])
    M = dd.select(names).to_numpy()
    nz = M.std(0) > 1e-10
    M, nm = M[:, nz], [n for n, k in zip(names, nz) if k]
    r = sm.OLS(dd["y"].to_numpy(), M).fit(cov_type="cluster", cov_kwds={"groups": cell_id})
    assert not np.isnan(r.bse).any(), f"{label} 標準誤含 NaN"
    print(label, "R2", round(r.rsquared, 6), "參數", len(nm))
    out = pl.DataFrame({"term": nm, f"coef_{label}": r.params, f"se_{label}": r.bse})
    out.write_csv(f"outputs/tables/11_{label}.csv")
    return out, r


lin, r_lin = fit(wx, [], "linear")
dum, r_dum = fit([c for c in wx if c not in ords], [(c, None) for c in ords], "dummy")

print()
print(dum.filter(pl.col("term").str.contains("_lv")))

cmp = (
    lin.filter(pl.col("term").str.starts_with("carrier_"))
    .join(dum.filter(pl.col("term").str.starts_with("carrier_")), on="term")
    .with_columns((pl.col("coef_dummy") - pl.col("coef_linear")).abs().alias("d"))
    .sort("coef_linear", descending=True)
)
print()
print(cmp.select("term", "coef_linear", "coef_dummy", "d", "se_linear"))
print("公司效應最大變動", round(float(cmp["d"].max()), 5),
      "對照 se 中位", round(float(cmp["se_linear"].median()), 5))
print("排序 spearman", round(float(np.corrcoef(cmp["coef_linear"].rank(), cmp["coef_dummy"].rank())[0, 1]), 4))
cmp.write_csv("outputs/tables/11_ordinal_compare.csv")
