import numpy as np
import polars as pl
import statsmodels.api as sm

b = pl.read_parquet("data/processed/bad_weather.parquet")

# clean.py 對不適用者填 0，壓力指標須限定確有前一航段者才有意義。
b = b.filter("prior_leg_available")

# 排定周轉是班表給的緩衝，上一段實際晚到吃掉它。剩餘緩衝即本班承受的壓力，
# 兩個輸入皆外生：排定值來自班表，上一段實際到達由 gate 保證早於本班的
# info_cutoff_utc。
b = b.with_columns(
    (pl.col("prior_leg_turnaround_min") - pl.col("prior_leg_arr_delay_min")).alias("slack")
)

# 過夜停場的排定周轉長達數百分鐘，壓力無從談起，且會主導連續指標的尺度。
b = b.filter(pl.col("prior_leg_turnaround_min") <= 240)

# 剩餘緩衝恆為正：緩衝若真被吃穿，飛機尚未到場，本班不可能由同一架執飛，
# 且發布者的 gate 另要求上一段實際到達早於本班 cutoff。故壓力無法以「緩衝
# 耗盡」界定，改取當日轉接樣本中緩衝最緊的五分之一。
cut = b["slack"].quantile(0.2)
b = b.with_columns((pl.col("slack") <= cut).alias("squeezed"))
print("當日轉接樣本", b.height, "緩衝門檻", cut, "壓力組比例", round(float(b["squeezed"].mean()), 4))

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
print("入模", b.height)

# 機械預期的錨點：上一段吃掉緩衝時，本班延誤已有外部成因，late-aircraft 欄
# 應吸收之。以它為對照，確認衝擊本身確實有效。
print(b.group_by("squeezed").agg(
    pl.len().alias("n"),
    pl.col("y1_attributed").mean().alias("y1"),
    (pl.col("reported_late_aircraft_delay_min") > 0).mean().alias("late_ac"),
).sort("squeezed"))

X_num = b.select(wx).to_numpy().astype(float)
X_num = (X_num - X_num.mean(0)) / (X_num.std(0) + 1e-9)

carriers = sorted(b["operating_carrier"].unique().to_list())
ref = "DL"
others = [c for c in carriers if c != ref]
sq = b["squeezed"].to_numpy().astype(float)
X_car = np.column_stack([(b["operating_carrier"] == c).to_numpy().astype(float) for c in others])
# 主效應與交互同時入模：主效應是該公司的水準差異，交互才是對衝擊的反應差異。
X_int = X_car * sq[:, None]
X = np.column_stack([X_num, sq, X_car, X_int])
names = wx + ["squeezed"] + [f"carrier_{c}" for c in others] + [f"sq_x_{c}" for c in others]

cell_id = b.select(pl.struct("origin_iata", "ym").rank("dense").alias("i"))["i"].to_numpy() - 1

rows = []
for col, label in [("y1_attributed", "y1"), ("y3_wx_vs_nas", "y3")]:
    mask = b[col].is_not_null().to_numpy()
    y = b[col].fill_null(0).to_numpy().astype(float)[mask]
    g = cell_id[mask]
    dm = pl.DataFrame(X[mask], schema=names).with_columns(pl.Series("g", g), pl.Series("y", y))
    dd = dm.select([(pl.col(c) - pl.col(c).mean().over("g")).alias(c) for c in names + ["y"]])
    r = sm.OLS(dd["y"].to_numpy(), dd.select(names).to_numpy()).fit(cov_type="cluster", cov_kwds={"groups": g})
    assert not np.isnan(r.bse).any() and (r.bse > 0).all()
    out = pl.DataFrame({"term": names, "coef": r.params, "se": r.bse}).with_columns(
        pl.lit(label).alias("y"),
        (pl.col("coef") - 1.96 * pl.col("se")).alias("lo"),
        (pl.col("coef") + 1.96 * pl.col("se")).alias("hi"),
    )
    print(label, "n", len(y))
    print(out.filter(pl.col("term").str.starts_with("sq_x_")).sort("coef", descending=True))
    rows.append(out)

pl.concat(rows).write_csv("outputs/tables/prior_leg_anchor.csv")
