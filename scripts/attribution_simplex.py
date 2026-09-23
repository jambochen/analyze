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

# 格子 × 成因固定效應吸收的是當地的平均成因分配，格子內的天氣變異未進模型。
# 這些欄位在單元內取均值後帶入，使「該公司在該格子實際遭遇的天氣」得以控制。
wx = [
    "wx_origin_flight_category_ordinal", "wx_dest_flight_category_ordinal",
    "wx_origin_visibility_m", "wx_origin_ceiling_m", "wx_origin_wind_speed_ms",
    "wx_origin_precip_1h_mm", "wx_origin_snow_depth_cm",
    "wx_origin_thunderstorm", "wx_origin_snow", "wx_origin_freezing_precip", "wx_origin_fog",
    # 起點惡劣與目的地惡劣是不同情境，兩端須分開進模型；單端氣象會讓本腳本的
    # 樣本口徑與 priority_list.py 不一致。
    "wx_dest_visibility_m", "wx_dest_ceiling_m", "wx_dest_wind_speed_ms",
    "wx_dest_precip_1h_mm",
    "arr_delay_min",
]
b = b.drop_nulls(wx)

agg = b.group_by("origin_iata", "ym", "operating_carrier").agg(
    [pl.col(c).sum().alias(s) for c, s in zip(rep, short)]
    + [pl.col(c).mean().alias(f"m_{c}") for c in wx]
    + [pl.len().alias("n_flights")]
)
cell = agg.group_by("origin_iata", "ym").agg(pl.col("operating_carrier").n_unique().alias("nc"))
agg = agg.join(cell.filter(pl.col("nc") >= 2).select("origin_iata", "ym"), on=["origin_iata", "ym"])
print("單元", agg.height)

mcols = [f"m_{c}" for c in wx]
long = agg.unpivot(
    index=["origin_iata", "ym", "operating_carrier", "n_flights"] + mcols,
    on=short, variable_name="cause", value_name="minutes",
).with_columns(
    pl.concat_str(["origin_iata", "ym", "operating_carrier"], separator="|").alias("unit"),
    pl.concat_str(["origin_iata", "ym", "cause"], separator="|").alias("cellcause"),
)

d = long.to_pandas()
for c in mcols:
    d[c] = (d[c] - d[c].mean()) / (d[c].std() + 1e-9)

terms = []
for c in sorted(set(d["operating_carrier"]) - {"DL"}):
    for s in [x for x in short if x != "carrier"]:
        col = f"x_{c}_{s}"
        d[col] = ((d["operating_carrier"] == c) & (d["cause"] == s)).astype(float)
        terms.append(col)

# 氣象在單元內不隨成因變動，主效應會被單元固定效應完全吸收。須與成因交互
# 才進得了模型，係數即「天氣越惡劣時，延誤多記入該欄」。
for c in mcols:
    for s in [x for x in short if x != "carrier"]:
        col = f"w_{c}_{s}"
        d[col] = d[c] * (d["cause"] == s).astype(float)
        terms.append(col)

m = pf.fepois(f"minutes ~ {' + '.join(terms)} | unit + cellcause", data=d, vcov={"CRV1": "unit"})
r = m.tidy().reset_index()
r.columns = ["term"] + list(r.columns[1:])
r.to_csv("outputs/tables/attribution_simplex.csv", index=False)

# 歸因邊界是係數的線性對比，非另立的結果變數。對比的標準誤須取兩係數的
# 共變異數，各自的標準誤不夠。
coefs = m.coef()
V = pd.DataFrame(m._vcov, index=coefs.index, columns=coefs.index)
rows = []
for c in sorted(set(d["operating_carrier"]) - {"DL"}):
    for a, bcause in [("weather", "nas"), ("weather", "late_ac"), ("nas", "late_ac")]:
        ta, tb = f"x_{c}_{a}", f"x_{c}_{bcause}"
        est = coefs[ta] - coefs[tb]
        se = np.sqrt(V.loc[ta, ta] + V.loc[tb, tb] - 2 * V.loc[ta, tb])
        rows.append({"carrier": c, "contrast": f"{a}-{bcause}", "estimate": est, "se": se,
                     "lo": est - 1.96 * se, "hi": est + 1.96 * se})
cb = pd.DataFrame(rows)
cb.to_csv("outputs/tables/attribution_boundary.csv", index=False)
print(cb[cb["contrast"] == "weather-nas"].sort_values("estimate", ascending=False).to_string())

car = r[r["term"].str.startswith("x_")].copy()
car["carrier"] = car["term"].str.split("_").str[1]
car["cause"] = car["term"].str.split("_", n=2).str[2]
base = pd.read_csv("outputs/tables/simplex_poisson.csv")
base.columns = ["term"] + list(base.columns[1:])
base["carrier"] = base["term"].str.split("_").str[1]
base["cause"] = base["term"].str.split("_", n=2).str[2]
cmp = car.merge(base[["carrier", "cause", "Estimate"]], on=["carrier", "cause"], suffixes=("", "_base"))
cmp["shift_in_se"] = (cmp["Estimate"] - cmp["Estimate_base"]).abs() / cmp["Std. Error"]
print(cmp[["carrier", "cause", "Estimate", "Estimate_base", "shift_in_se"]].sort_values("shift_in_se", ascending=False).head(12).to_string())
print("\n加氣象後係數最大位移（以標準誤計）", round(cmp["shift_in_se"].max(), 3))
for s in ["weather", "nas", "late_ac"]:
    t = cmp[cmp["cause"] == s].sort_values("Estimate", ascending=False)
    print(f"\n{s} 前五", t["carrier"].head(5).to_list())
    print(f"{s} 與原設定的 Spearman", round(t["Estimate"].corr(t["Estimate_base"], method="spearman"), 4))
