import numpy as np
import polars as pl

wx = [
    "wx_origin_visibility_m", "wx_origin_ceiling_m", "wx_origin_wind_speed_ms", "wx_origin_wind_gust_ms",
    "wx_origin_precip_1h_mm", "wx_origin_snow_depth_cm", "wx_origin_temp_c", "wx_origin_sky_cover_oktas",
    "wx_origin_flight_category_ordinal", "wx_dest_flight_category_ordinal", "wx_dest_visibility_m",
    "wx_dest_ceiling_m", "wx_dest_wind_speed_ms", "wx_dest_precip_1h_mm",
    "wx_origin_thunderstorm", "wx_origin_snow", "wx_origin_freezing_precip", "wx_origin_fog",
    "wx_origin_m3_flight_category_ordinal", "wx_origin_m6_flight_category_ordinal",
    "arr_delay_min", "distance_miles", "origin_sched_departures_this_hour", "prior_leg_arr_delay_min",
]

b = pl.read_parquet("data/processed/bad_weather.parquet", columns=wx).drop_nulls(wx)
X = b.to_numpy().astype(float)
X = (X - X.mean(0)) / X.std(0)

u, s, vt = np.linalg.svd(X, full_matrices=False)
ev = s**2 / (len(X) - 1)
ratio = ev / ev.sum()
cum = np.cumsum(ratio)

print("有效維度：累積解釋變異")
for i, (r, c) in enumerate(zip(ratio, cum), 1):
    print(f"PC{i:2d} {r:.4f} {c:.4f}")

# 條件數是共線的直接指標；>30 視為有問題，>100 為嚴重。
print()
print("條件數", round(float(s.max() / s.min()), 1))
print("90% 變異所需維度", int(np.searchsorted(cum, 0.90) + 1), "/", len(wx))

# VIF 逐欄看，找出真正共線的是哪幾欄。用相關矩陣的反對角線算。
R = np.corrcoef(X, rowvar=False)
vif = np.diag(np.linalg.inv(R))
v = pl.DataFrame({"col": wx, "vif": vif}).sort("vif", descending=True)
print()
print(v)
v.write_csv("outputs/tables/wx_vif.csv")

# 前幾個主成分各自由哪些欄位主導，判斷是否為可解釋的「惡劣度」維度。
print()
for k in range(4):
    load = pl.DataFrame({"col": wx, "load": vt[k]}).with_columns(pl.col("load").abs().alias("a")).sort(
        "a", descending=True
    )
    print(f"PC{k + 1} ({ratio[k]:.1%})", [(r["col"].replace("wx_", ""), round(r["load"], 2)) for r in load.head(5).iter_rows(named=True)])
