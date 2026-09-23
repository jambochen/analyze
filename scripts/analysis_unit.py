import numpy as np
import polars as pl
import statsmodels.api as sm

b = pl.read_parquet("data/processed/bad_weather.parquet").with_columns(
    pl.col("sched_dep_utc").dt.truncate("1mo").alias("ym")
)

wx = [
    "wx_origin_visibility_m", "wx_origin_ceiling_m", "wx_origin_wind_speed_ms", "wx_origin_wind_gust_ms",
    "wx_origin_precip_1h_mm", "wx_origin_snow_depth_cm", "wx_origin_temp_c", "wx_origin_sky_cover_oktas",
    "wx_origin_flight_category_ordinal", "wx_dest_flight_category_ordinal", "wx_dest_visibility_m",
    "wx_dest_ceiling_m", "wx_dest_wind_speed_ms", "wx_dest_precip_1h_mm",
    "wx_origin_thunderstorm", "wx_origin_snow", "wx_origin_freezing_precip", "wx_origin_fog",
    "wx_origin_m3_flight_category_ordinal", "wx_origin_m6_flight_category_ordinal",
    "arr_delay_min", "distance_miles", "origin_sched_departures_this_hour", "prior_leg_arr_delay_min",
    # 當日首航段的 prior_leg 填零是「不適用」而非「上段準點」，兩者形態相同，
    # 須留指示變數供模型分辨。與雪深、陣風的 _absent 同一原則。
    "prior_leg_available",
]
b = b.drop_nulls(wx)


# 兩個分析單位跑完全相同的流程，係數才可直接對比。差別只在 unit 欄位。
def estimate(d, unit, ref, label):
    cell = d.group_by("origin_iata", "ym").agg(
        pl.col(unit).n_unique().alias("nc"), pl.col("y1_attributed").mean().alias("cr")
    )
    keep = cell.filter((pl.col("nc") >= 2) & (pl.col("cr") > 0) & (pl.col("cr") < 1)).select("origin_iata", "ym")
    d = d.join(keep, on=["origin_iata", "ym"])

    X_num = d.select(wx).to_numpy().astype(float)
    X_num = (X_num - X_num.mean(0)) / (X_num.std(0) + 1e-9)
    units = sorted(u for u in d[unit].unique().to_list() if u != ref)
    X = np.column_stack([X_num] + [(d[unit] == u).to_numpy().astype(float) for u in units])
    names = wx + [f"{unit}_{u}" for u in units]
    cell_id = d.select(pl.struct("origin_iata", "ym").rank("dense").alias("i"))["i"].to_numpy() - 1

    for col, suffix in [("y1_attributed", "y1"), ("y2_share", "y2"), ("y3_wx_vs_nas", "y3")]:
        mask = d[col].is_not_null().to_numpy()
        y = d[col].fill_null(0).to_numpy().astype(float)[mask]
        g = cell_id[mask]
        dm = pl.DataFrame(X[mask], schema=names).with_columns(pl.Series("g", g), pl.Series("y", y))
        dd = dm.select([(pl.col(c) - pl.col(c).mean().over("g")).alias(c) for c in names + ["y"]])
        r = sm.OLS(dd["y"].to_numpy(), dd.select(names).to_numpy()).fit(
            cov_type="cluster", cov_kwds={"groups": g}
        )
        # 共線會讓係數照常估出而標準誤變 NaN，只發一個 warning。必須明確檢查。
        assert not np.isnan(r.bse).any(), f"{label} {suffix} 標準誤含 NaN"
        out = pl.DataFrame({"term": names, "coef": r.params, "se": r.bse}).with_columns(
            (pl.col("coef") - 1.96 * pl.col("se")).alias("lo"),
            (pl.col("coef") + 1.96 * pl.col("se")).alias("hi"),
        )
        print(label, suffix, "n", len(y), "cells", len(set(g.tolist())))
        print(out.filter(pl.col("term").str.starts_with(f"{unit}_")).sort("coef"))
        out.write_csv(f"outputs/tables/{label}_{suffix}.csv")


estimate(b, "operating_carrier", "DL", "op")
estimate(b, "marketing_carrier", "DL", "mkt")
