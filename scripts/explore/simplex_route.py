import numpy as np
import polars as pl

b = pl.read_parquet("data/processed/bad_weather.parquet")
rep = ["reported_carrier_delay_min", "reported_weather_delay_min",
       "reported_nas_delay_min", "reported_late_aircraft_delay_min"]
short = ["carrier", "weather", "nas", "late_ac"]

# security 佔總申報分鐘極低且幾乎不填，剔除後重新歸一化為四維單體。
b = b.with_columns(sum(pl.col(c) for c in rep).alias("tot")).filter(pl.col("tot") > 0)
print("四維單體樣本", b.height)

b = b.with_columns(pl.col("sched_dep_utc").dt.truncate("1mo").alias("ym"))

# 路線判準一：把分鐘數當多項計數，格子內各公司的分配是否有足夠樣本。
# 以格子 × 公司為單位彙總分鐘數，零值在彙總層會大幅減少。
agg = b.group_by("origin_iata", "ym", "operating_carrier").agg(
    [pl.col(c).sum().alias(s) for c, s in zip(rep, short)] + [pl.len().alias("n_flights")]
)
agg = agg.with_columns(sum(pl.col(s) for s in short).alias("tot"))
print("格子 × 公司 單元數", agg.height)
print("其中總分鐘 > 0", agg.filter(pl.col("tot") > 0).height)
for s in short:
    print(f"彙總後 {s} 為零的比例", round(float((agg[s] == 0).mean()), 4))

# 判準二：多公司格子的保留量。沿用 carrier_effects.py 的格子條件。
cell = agg.group_by("origin_iata", "ym").agg(
    pl.len().alias("nc"), pl.col("n_flights").sum().alias("cn")
)
keep = cell.filter(pl.col("nc") >= 2)
print("\n多公司格子", keep.height, "佔", round(keep.height / cell.height, 3))
kept = agg.join(keep.select("origin_iata", "ym"), on=["origin_iata", "ym"])
print("保留單元", kept.height, "涵蓋航班", int(kept["n_flights"].sum()), "佔", round(float(kept["n_flights"].sum() / b.height), 3))

# 判準三：彙總層的零值是否低到可用 ILR。四維 ILR 需三個座標，任一欄為零即失效。
allpos = kept.filter(pl.all_horizontal([pl.col(s) > 0 for s in short]))
print("\n四欄皆正的單元", allpos.height, "佔", round(allpos.height / kept.height, 3),
      "涵蓋航班佔", round(float(allpos["n_flights"].sum() / kept["n_flights"].sum()), 3))

# 樣本量夠大的單元，零值應更少。
for m in [20, 50, 100]:
    sub = kept.filter(pl.col("n_flights") >= m)
    ap = sub.filter(pl.all_horizontal([pl.col(s) > 0 for s in short]))
    print(f"n_flights>={m}: 單元 {sub.height}, 四欄皆正 {ap.height}, 佔 {round(ap.height/max(sub.height,1),3)}")
