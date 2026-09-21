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

b = b.with_columns(
    pl.when(pl.col("sched_dep_utc") < pl.lit("2020-04-01").str.to_datetime().dt.replace_time_zone("UTC"))
    .then(pl.lit("pre"))
    .when(pl.col("sched_dep_utc") < pl.lit("2021-06-01").str.to_datetime().dt.replace_time_zone("UTC"))
    .then(pl.lit("break"))
    .otherwise(pl.lit("new"))
    .alias("era")
)

# 連續壓力指標：機場當日排定起飛架次相對該機場自身的常態水準。取自班表而非
# 實際營運，不受回報行為影響。以機場為基準避免大小機場的規模差異混進來。
b = b.with_columns(
    (
        pl.col("origin_sched_departures_this_day")
        / pl.col("origin_sched_departures_this_day").mean().over("origin_iata")
    ).alias("load")
)
print(b.group_by("era").agg(pl.len(), pl.col("load").mean()).sort("era"))

cell = b.group_by("origin_iata", "ym").agg(
    pl.col("operating_carrier").n_unique().alias("nc"), pl.col("y1_attributed").mean().alias("cr")
)
keep = cell.filter((pl.col("nc") >= 2) & (pl.col("cr") > 0) & (pl.col("cr") < 1)).select("origin_iata", "ym")
b = b.join(keep, on=["origin_iata", "ym"])

carriers = sorted(c for c in b["operating_carrier"].unique().to_list() if c != "DL")
cell_id = b.select(pl.struct("origin_iata", "ym").rank("dense").alias("i"))["i"].to_numpy() - 1

X_num = b.select(wx).to_numpy().astype(float)
X_num = (X_num - X_num.mean(0)) / (X_num.std(0) + 1e-9)
car = {c: (b["operating_carrier"] == c).to_numpy().astype(float) for c in carriers}


def run(mod_name, mod_vals, label):
    # 交互項：公司 × 調節變數。主效應由 era/load 自身承載，但 era 的主效應與
    # 機場 × 年月固定效應共線（同格子內 era 恆定），去均值後自動消失，不另加。
    cols, names = [X_num], list(wx)
    for c in carriers:
        cols.append(car[c].reshape(-1, 1))
        names.append(f"carrier_{c}")
    for c in carriers:
        for mv_name, mv in mod_vals:
            cols.append((car[c] * mv).reshape(-1, 1))
            names.append(f"carrier_{c}:{mod_name}_{mv_name}")
    for mv_name, mv in mod_vals:
        cols.append(mv.reshape(-1, 1))
        names.append(f"{mod_name}_{mv_name}")
    X = np.column_stack(cols)

    for col, suffix in [("y1_attributed", "y1"), ("y2_share", "y2"), ("y3_wx_vs_nas", "y3")]:
        mask = b[col].is_not_null().to_numpy()
        y = b[col].fill_null(0).to_numpy().astype(float)[mask]
        g = cell_id[mask]
        dm = pl.DataFrame(X[mask], schema=names).with_columns(pl.Series("g", g), pl.Series("y", y))
        dd = dm.select([(pl.col(c) - pl.col(c).mean().over("g")).alias(c) for c in names + ["y"]])
        M = dd.select(names).to_numpy()
        # era 主效應在格子內恆定，去均值後全為零，會使矩陣降秩。剔除零欄位。
        nz = M.std(0) > 1e-10
        M, nm = M[:, nz], [n for n, k in zip(names, nz) if k]
        r = sm.OLS(dd["y"].to_numpy(), M).fit(cov_type="cluster", cov_kwds={"groups": g})
        assert not np.isnan(r.bse).any(), f"{label} {suffix} 標準誤含 NaN"
        out = pl.DataFrame({"term": nm, "coef": r.params, "se": r.bse}).with_columns(
            (pl.col("coef") - 1.96 * pl.col("se")).alias("lo"),
            (pl.col("coef") + 1.96 * pl.col("se")).alias("hi"),
            (pl.col("coef").abs() / pl.col("se")).alias("t"),
        )
        ix = out.filter(pl.col("term").str.contains(":"))
        print(f"\n=== {label} {suffix} n={len(y)} 交互項顯著數 {ix.filter(pl.col('t') > 1.96).height}/{ix.height}")
        print(ix.sort("coef", descending=True).head(8).select("term", "coef", "se", "t"))
        print(ix.sort("coef").head(5).select("term", "coef", "se", "t"))
        out.write_csv(f"outputs/tables/{label}_{suffix}.csv")


brk = (b["era"] == "break").to_numpy().astype(float)
new = (b["era"] == "new").to_numpy().astype(float)
run("era", [("break", brk), ("new", new)], "era")

load = b["load"].to_numpy().astype(float)
load = (load - load.mean()) / load.std()
run("load", [("z", load)], "load")
