import numpy as np
import polars as pl
import statsmodels.api as sm

b = pl.read_parquet("data/processed/bad_weather.parquet").filter("prior_leg_available")
b = b.with_columns((pl.col("prior_leg_turnaround_min") - pl.col("prior_leg_arr_delay_min")).alias("slack"))
b = b.filter(pl.col("prior_leg_turnaround_min") <= 240)
b = b.with_columns(pl.col("sched_dep_utc").dt.truncate("1mo").alias("ym"))

cell = b.group_by(["origin_iata", "ym"]).agg(
    pl.col("operating_carrier").n_unique().alias("nc"), pl.col("y1_attributed").mean().alias("cr")
)
keep = cell.filter((pl.col("nc") >= 2) & (pl.col("cr") > 0) & (pl.col("cr") < 1)).select("origin_iata", "ym")
b = b.join(keep, on=["origin_iata", "ym"])

wx = [
    "wx_origin_visibility_m", "wx_origin_ceiling_m", "wx_origin_wind_speed_ms", "wx_origin_wind_gust_ms",
    "wx_origin_precip_1h_mm", "wx_origin_snow_depth_cm", "wx_origin_temp_c", "wx_origin_sky_cover_oktas",
    "wx_origin_flight_category_ordinal", "wx_dest_flight_category_ordinal", "wx_dest_visibility_m",
    "wx_dest_ceiling_m", "wx_dest_wind_speed_ms", "wx_dest_precip_1h_mm",
    "wx_origin_thunderstorm", "wx_origin_snow", "wx_origin_freezing_precip", "wx_origin_fog",
    "wx_origin_m3_flight_category_ordinal", "wx_origin_m6_flight_category_ordinal",
    "arr_delay_min", "distance_miles", "origin_sched_departures_this_hour",
]
b = b.drop_nulls(wx)

X_num = b.select(wx).to_numpy().astype(float)
X_num = (X_num - X_num.mean(0)) / (X_num.std(0) + 1e-9)
carriers = sorted(b["operating_carrier"].unique().to_list())
others = [c for c in carriers if c != "DL"]
X_car = np.column_stack([(b["operating_carrier"] == c).to_numpy().astype(float) for c in others])
cell_id = b.select(pl.struct("origin_iata", "ym").rank("dense").alias("i"))["i"].to_numpy() - 1
slack = b["slack"].to_numpy()
y = b["y1_attributed"].to_numpy().astype(float)

# 壓力組的切點是選定值，非資料給定的斷點。錨點若只在某一個切點成立即不可信。
rows = []
for q in [0.1, 0.15, 0.2, 0.25, 0.3, 0.4]:
    cut = np.quantile(slack, q)
    sq = (slack <= cut).astype(float)
    X = np.column_stack([X_num, sq, X_car, X_car * sq[:, None]])
    names = wx + ["squeezed"] + [f"carrier_{c}" for c in others] + [f"sq_x_{c}" for c in others]
    dm = pl.DataFrame(X, schema=names).with_columns(pl.Series("g", cell_id), pl.Series("y", y))
    dd = dm.select([(pl.col(c) - pl.col(c).mean().over("g")).alias(c) for c in names + ["y"]])
    r = sm.OLS(dd["y"].to_numpy(), dd.select(names).to_numpy()).fit(
        cov_type="cluster", cov_kwds={"groups": cell_id}
    )
    assert not np.isnan(r.bse).any() and (r.bse > 0).all()
    t = pl.DataFrame({"term": names, "coef": r.params, "se": r.bse}).filter(
        pl.col("term").str.starts_with("sq_x_")
    ).with_columns(
        pl.lit(q).alias("q"), pl.lit(float(cut)).alias("cut"),
        (pl.col("coef") - 1.96 * pl.col("se")).alias("lo"),
        (pl.col("coef") + 1.96 * pl.col("se")).alias("hi"),
        pl.col("term").str.replace("sq_x_", "").alias("carrier"),
    )
    rows.append(t)
    print(q, "cut", cut, "壓力組佔", round(float(sq.mean()), 3),
          "OO", round(float(t.filter(pl.col("carrier") == "OO")["coef"][0]), 5),
          "lo", round(float(t.filter(pl.col("carrier") == "OO")["lo"][0]), 5))

a = pl.concat(rows)
a.write_csv("outputs/tables/prior_leg_cutoff.csv")

# 每個切點下，係數為正且區間不含零的公司。錨點要可信，這份名單應穩定。
print()
for q in a["q"].unique(maintain_order=True):
    pos = a.filter((pl.col("q") == q) & (pl.col("lo") > 0)).sort("coef", descending=True)
    print(q, pos["carrier"].to_list())
