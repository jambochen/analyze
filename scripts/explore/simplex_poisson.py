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

agg = b.group_by("origin_iata", "ym", "operating_carrier").agg(
    [pl.col(c).sum().alias(s) for c, s in zip(rep, short)] + [pl.len().alias("n_flights")]
)
cell = agg.group_by("origin_iata", "ym").agg(pl.col("operating_carrier").n_unique().alias("nc"))
agg = agg.join(cell.filter(pl.col("nc") >= 2).select("origin_iata", "ym"), on=["origin_iata", "ym"])

# 多項 logit 與 Poisson 對數線性等價：每個 (單元, 欄位) 一列計數。單元固定效應
# 吸收該公司在該格子的申報總量，格子 × 欄位固定效應吸收當地的基準成因分配，
# 故公司 × 欄位交互即「相對同格子同儕，該公司把延誤多推向哪一欄」。
long = agg.unpivot(
    index=["origin_iata", "ym", "operating_carrier", "n_flights"],
    on=short, variable_name="cause", value_name="minutes",
).with_columns(
    pl.concat_str(["origin_iata", "ym", "operating_carrier"], separator="|").alias("unit"),
    pl.concat_str(["origin_iata", "ym", "cause"], separator="|").alias("cellcause"),
)
print("長表", long.shape, "單元", long["unit"].n_unique(), "格子×欄位", long["cellcause"].n_unique())

# DL 為對照公司（沿用 carrier_effects.py），carrier 欄為對照欄位。
d = long.to_pandas()
# 交互項須顯式建構。以公式的 A:B 展開會連基準欄位的交互一併估出，那組與單元
# 固定效應完全共線，求解時被任意丟棄，剩下的係數不再是對 carrier 欄的對比。
terms = []
for c in sorted(set(d["operating_carrier"]) - {"DL"}):
    for s in [x for x in short if x != "carrier"]:
        col = f"x_{c}_{s}"
        d[col] = ((d["operating_carrier"] == c) & (d["cause"] == s)).astype(float)
        terms.append(col)

m = pf.fepois(f"minutes ~ {' + '.join(terms)} | unit + cellcause", data=d, vcov={"CRV1": "unit"})
print(m.summary())
r = m.tidy()
r.to_csv("outputs/tables/simplex_poisson.csv")
