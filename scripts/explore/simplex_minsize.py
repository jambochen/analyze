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
mcols = [f"m_{c}" for c in wx]

agg_all = b.group_by("origin_iata", "ym", "operating_carrier").agg(
    [pl.col(c).sum().alias(s) for c, s in zip(rep, short)]
    + [pl.col(c).mean().alias(f"m_{c}") for c in wx] + [pl.len().alias("n_flights")]
)

# separation 移除的全是零，且集中在 weather 欄與極小的單元。只丟零是不對稱的，
# 會讓表現正常的格子被選擇性剔除。改為整批排除小單元，檢查係數是否隨之改變。
rows = []
for minsize in [1, 3, 5, 10]:
    agg = agg_all.filter(pl.col("n_flights") >= minsize)
    cell = agg.group_by("origin_iata", "ym").agg(pl.col("operating_carrier").n_unique().alias("nc"))
    agg = agg.join(cell.filter(pl.col("nc") >= 2).select("origin_iata", "ym"), on=["origin_iata", "ym"])

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

    m = pf.fepois(f"minutes ~ {' + '.join(new)} | unit + cellcause", data=d, vcov={"CRV1": "unit"})
    t = m.tidy().reset_index()
    t.columns = ["term"] + list(t.columns[1:])
    t = t[t["term"].str.startswith("x_")].copy()
    t["carrier"] = t["term"].str.split("_").str[1]
    t["cause"] = t["term"].str.split("_", n=2).str[2]
    t["minsize"] = minsize
    t["n_sep"] = getattr(m, "n_separation_na", 0)
    rows.append(t)
    oo = t[t["carrier"] == "OO"].set_index("cause")["Estimate"]
    print(minsize, "單元", d["unit"].nunique(), "separation", getattr(m, "n_separation_na", 0),
          "OO weather", round(oo["weather"], 4), "nas", round(oo["nas"], 4), "late_ac", round(oo["late_ac"], 4))

a = pd.concat(rows)
a.to_csv("outputs/tables/simplex_minsize.csv", index=False)

base = a[a["minsize"] == 1].set_index(["carrier", "cause"])["Estimate"]
for ms in [3, 5, 10]:
    cur = a[a["minsize"] == ms].set_index(["carrier", "cause"])["Estimate"]
    se = a[a["minsize"] == ms].set_index(["carrier", "cause"])["Std. Error"]
    shift = ((cur - base).abs() / se)
    print(f"\nminsize={ms} 相對無限制的最大位移（標準誤計）", round(shift.max(), 3))
    for s in ["weather", "nas", "late_ac"]:
        x = a[(a["minsize"] == ms) & (a["cause"] == s)].set_index("carrier")["Estimate"]
        y = a[(a["minsize"] == 1) & (a["cause"] == s)].set_index("carrier")["Estimate"]
        print(f"  {s} Spearman", round(x.corr(y, method="spearman"), 4))
