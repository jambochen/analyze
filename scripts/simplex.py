import pandas as pd
import polars as pl
import pyfixest as pf

import stratum

REPORTED = ["reported_carrier_delay_min", "reported_weather_delay_min",
            "reported_nas_delay_min", "reported_late_aircraft_delay_min"]
CAUSES = ["carrier", "weather", "nas", "late_ac"]

# 層 × 成因固定效應只吸收當地的平均成因分配，這些欄位在單元內取均值後與成因
# 交互帶入，控制「該公司在該層實際遭遇的天氣」。起點與目的地是不同情境，須分開。
WX_UNIT = [
    "wx_origin_flight_category_ordinal", "wx_dest_flight_category_ordinal",
    "wx_origin_visibility_m", "wx_origin_ceiling_m", "wx_origin_wind_speed_ms",
    "wx_origin_precip_1h_mm", "wx_origin_snow_depth_cm",
    "wx_origin_thunderstorm", "wx_origin_snow", "wx_origin_freezing_precip", "wx_origin_fog",
    "wx_dest_visibility_m", "wx_dest_ceiling_m", "wx_dest_wind_speed_ms", "wx_dest_precip_1h_mm",
    "arr_delay_min",
]


def data(b, controls=WX_UNIT, minsize=1):
    """每個 (單元, 成因) 一列的長表與其解釋變數名。單元 = 層 × 執飛者。

    multinomial logit 以 Poisson 對數線性的等價形式估：單元固定效應吸收該公司在
    該層的申報總量，層 × 成因固定效應吸收當地的基準成因分配，公司 × 成因交互
    即 multinomial 的係數，基準成因為 carrier、基準公司為 stratum.REF。
    """
    # security 佔總申報分鐘極低且幾乎不填，剔除後重新歸一化為四維 simplex。
    b = b.with_columns(pl.sum_horizontal(REPORTED).alias("tot")).filter(pl.col("tot") > 0)
    b = b.drop_nulls(list(controls))
    agg = b.group_by("origin_iata", "ym", "operating_carrier").agg(
        [pl.col(c).sum().alias(s) for c, s in zip(REPORTED, CAUSES)]
        + [pl.col(c).mean().alias(f"m_{c}") for c in controls]
        + [pl.len().alias("n_flights")]
    ).filter(pl.col("n_flights") >= minsize)
    cell = agg.group_by("origin_iata", "ym").agg(pl.col("operating_carrier").n_unique().alias("nc"))
    agg = agg.join(cell.filter(pl.col("nc") >= 2).select("origin_iata", "ym"), on=["origin_iata", "ym"])

    mcols = [f"m_{c}" for c in controls]
    d = agg.unpivot(
        index=["origin_iata", "ym", "operating_carrier", "n_flights"] + mcols,
        on=CAUSES, variable_name="cause", value_name="minutes",
    ).with_columns(
        pl.concat_str(["origin_iata", "ym", "operating_carrier"], separator="|").alias("unit"),
        pl.concat_str(["origin_iata", "ym", "cause"], separator="|").alias("cellcause"),
    ).to_pandas()
    for c in mcols:
        d[c] = (d[c] - d[c].mean()) / (d[c].std() + 1e-9)

    # 交互項須顯式建構。以公式的 A:B 展開會連基準成因的交互一併估出，那組與單元
    # 固定效應完全共線，求解時被任意丟棄，剩下的係數不再是對 carrier 欄的對比，
    # 且不報錯。
    new = {}
    for c in sorted(set(d["operating_carrier"]) - {stratum.REF}):
        for s in CAUSES[1:]:
            new[f"x_{c}_{s}"] = ((d["operating_carrier"] == c) & (d["cause"] == s)).astype(float)
    # 氣象在單元內不隨成因變動，主效應會被單元固定效應完全吸收，須與成因交互。
    for c in mcols:
        for s in CAUSES[1:]:
            new[f"w_{c}_{s}"] = d[c] * (d["cause"] == s).astype(float)
    return pd.concat([d, pd.DataFrame(new, index=d.index)], axis=1), list(new)


def fit(d, terms, vcov={"CRV1": "unit"}):
    return pf.fepois(f"minutes ~ {' + '.join(terms)} | unit + cellcause", data=d, vcov=vcov)


def carrier_terms(m):
    """公司 × 成因係數，拆出 carrier 與 cause 兩欄。"""
    t = m.tidy().reset_index()
    t.columns = ["term"] + list(t.columns[1:])
    t = t[t["term"].str.startswith("x_")].copy()
    t["carrier"] = t["term"].str.split("_").str[1]
    t["cause"] = t["term"].str.split("_", n=2).str[2]
    return t
