import numpy as np
import pandas as pd
import polars as pl
import pyfixest as pf

b = pl.read_parquet("data/processed/bad_weather.parquet")
rep = ["reported_carrier_delay_min", "reported_weather_delay_min",
       "reported_nas_delay_min", "reported_late_aircraft_delay_min"]
short = ["carrier", "weather", "nas", "late_ac"]

b = b.with_columns(
    sum(pl.col(c) for c in rep).alias("tot"),
    pl.col("sched_dep_utc").dt.truncate("1mo").alias("ym"),
).filter(pl.col("tot") > 0)

wx = [
    "wx_origin_flight_category_ordinal", "wx_dest_flight_category_ordinal",
    "wx_origin_visibility_m", "wx_origin_ceiling_m", "wx_origin_wind_speed_ms",
    "wx_origin_precip_1h_mm", "wx_origin_snow_depth_cm",
    "wx_origin_thunderstorm", "wx_origin_snow", "wx_origin_freezing_precip", "wx_origin_fog",
    "arr_delay_min",
]
b = b.drop_nulls(wx)

agg = b.group_by("origin_iata", "ym", "operating_carrier").agg(
    [pl.col(c).sum().alias(s) for c, s in zip(rep, short)]
    + [pl.col(c).mean().alias(f"m_{c}") for c in wx] + [pl.len().alias("n_flights")]
)
cell = agg.group_by("origin_iata", "ym").agg(pl.col("operating_carrier").n_unique().alias("nc"))
agg = agg.join(cell.filter(pl.col("nc") >= 2).select("origin_iata", "ym"), on=["origin_iata", "ym"])

mcols = [f"m_{c}" for c in wx]
long = agg.unpivot(
    index=["origin_iata", "ym", "operating_carrier", "n_flights"] + mcols,
    on=short, variable_name="cause", value_name="minutes",
).with_columns(
    pl.concat_str(["origin_iata", "ym", "operating_carrier"], separator="|").alias("unit"),
    pl.concat_str(["origin_iata", "ym", "cause"], separator="|").alias("cellcause"),
)

d = long.to_pandas().reset_index(drop=True)
for c in mcols:
    d[c] = (d[c] - d[c].mean()) / (d[c].std() + 1e-9)

new = {}
for c in sorted(set(d["operating_carrier"]) - {"DL"}):
    for s in [x for x in short if x != "carrier"]:
        new[f"x_{c}_{s}"] = ((d["operating_carrier"] == c) & (d["cause"] == s)).astype(float)
for c in mcols:
    for s in [x for x in short if x != "carrier"]:
        new[f"w_{c}_{s}"] = d[c] * (d["cause"] == s).astype(float)
d = pd.concat([d, pd.DataFrame(new, index=d.index)], axis=1)
terms = list(new)

m = pf.fepois(f"minutes ~ {' + '.join(terms)} | unit + cellcause", data=d, vcov={"CRV1": "unit"})
dropped = np.asarray(m.na_index, dtype=int)
print("移除列數", len(dropped), "佔", round(len(dropped) / len(d), 4))

d["dropped"] = False
d.loc[d.index.isin(dropped), "dropped"] = True
k = pl.from_pandas(d[["operating_carrier", "cause", "minutes", "n_flights", "dropped"]])

# separation 若集中在特定公司，該公司的係數就是以殘餘樣本估的，須另行說明。
print(k.group_by("operating_carrier").agg(
    pl.len().alias("n"), pl.col("dropped").sum().alias("drop"), pl.col("dropped").mean().alias("rate")
).sort("rate", descending=True))
print(k.group_by("cause").agg(
    pl.len().alias("n"), pl.col("dropped").sum().alias("drop"), pl.col("dropped").mean().alias("rate")
).sort("rate", descending=True))
print("被移除者的分鐘數", k.filter("dropped").select(
    (pl.col("minutes") == 0).mean().alias("zero_share"), pl.col("minutes").median().alias("med")))
print("被移除者的單元航班數中位", k.filter("dropped")["n_flights"].median(),
      "保留者", k.filter(~pl.col("dropped"))["n_flights"].median())
