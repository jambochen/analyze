import sys

import numpy as np
import polars as pl

sys.path.insert(0, "scripts")
import stratum

# oracle_ 是事件時刻的氣象。用於預測延誤即構成洩漏，但本題不預測延誤——估的是
# 給定氣象下的歸因傾向，屬發布者明說的 retrospective 用途。
# 用它的理由：航空公司是事後填申報表，其資訊集接近 oracle_ 而非 wx_（排定
# 起飛前）。以 wx_ 為唯一基準，等於拿起飛前可知的天氣評斷事後填的表。
# 目的地取 scharr（排定到達時刻）而非 arr（實際到達時刻）：後者的觀測時點由
# 航班何時落地決定，班機越晚到就取越晚的觀測，與延誤本身機械相關。
oracle = [
    "oracle_origin_dep_visibility_m", "oracle_origin_dep_ceiling_m", "oracle_origin_dep_wind_speed_ms",
    "oracle_origin_dep_wind_gust_ms", "oracle_origin_dep_precip_1h_mm", "oracle_origin_dep_temp_c",
    "oracle_origin_dep_flight_category_ordinal", "oracle_origin_dep_thunderstorm", "oracle_origin_dep_snow",
    "oracle_origin_dep_freezing_precip", "oracle_origin_dep_fog",
    "oracle_dest_scharr_visibility_m", "oracle_dest_scharr_ceiling_m", "oracle_dest_scharr_wind_speed_ms",
    "oracle_dest_scharr_precip_1h_mm", "oracle_dest_scharr_flight_category_ordinal",
    "oracle_dest_scharr_thunderstorm", "oracle_dest_scharr_snow", "oracle_dest_scharr_freezing_precip",
    "oracle_dest_scharr_fog",
]
keys = ["sched_dep_utc", "tail_number", "origin_iata"]

files = [f"data/raw/flights-weather/flights_weather_{y}.parquet" for y in range(2019, 2025)]
# 先套稽核樣本的兩道篩選再取 oracle_，否則全樣本的 unique 會吃爆記憶體。
orc = (
    pl.scan_parquet(files)
    .filter(pl.col("arr_delay_min") >= 15)
    .filter(
        pl.col("wx_origin_flight_category").is_in(["IFR", "LIFR"])
        | pl.col("wx_dest_flight_category").is_in(["IFR", "LIFR"])
    )
    .select(keys + oracle)
    .collect(engine="streaming")
)

b = stratum.load()
n0 = b.height
b = b.join(orc.unique(subset=keys), on=keys, how="inner")
print("join 保留", b.height, "/", n0)

wx = stratum.WX
ctrl = ["arr_delay_min", "distance_miles", "origin_sched_departures_this_hour", "prior_leg_arr_delay_min",
        "prior_leg_available"]
b = b.drop_nulls(wx + oracle)
print("drop_nulls 後", b.height)

# 起飛後的天氣變化：事件時刻減排定起飛前。正值為惡化（ordinal 越大越差）。
b = b.with_columns(
    (pl.col("oracle_origin_dep_flight_category_ordinal") - pl.col("wx_origin_flight_category_ordinal")).alias("worsen")
)
print(b.group_by("worsen").agg(pl.len(), pl.col("y1_attributed").mean()).sort("worsen"))


def fit(controls, label, extra=()):
    out = stratum.fit(b, "y1_attributed", controls=controls, extra=extra)
    out = out.select("term", pl.col("coef").alias(f"coef_{label}"), pl.col("se").alias(f"se_{label}"))
    out.write_csv(f"outputs/tables/{label}.csv")
    return out


base = fit(wx, "wx_base")
orc_only = fit(oracle + ctrl, "oracle_base")
both = fit(wx + oracle, "both")

cmp = (
    base.filter(pl.col("term").str.starts_with("carrier_"))
    .join(orc_only.filter(pl.col("term").str.starts_with("carrier_")), on="term")
    .join(both.filter(pl.col("term").str.starts_with("carrier_")), on="term")
    .with_columns((pl.col("coef_oracle_base") - pl.col("coef_wx_base")).alias("shift"))
    .sort("coef_wx_base", descending=True)
)
print()
print(cmp.select("term", "coef_wx_base", "coef_oracle_base", "coef_both", "shift", "se_wx_base"))
cmp.write_csv("outputs/tables/basis_compare.csv")
print()
print("排序相關 spearman", round(
    np.corrcoef(cmp["coef_wx_base"].rank(), cmp["coef_oracle_base"].rank())[0, 1], 4))
print("最大位移", round(cmp["shift"].abs().max(), 4), "對照 se 中位", round(cmp["se_wx_base"].median(), 4))

# 機會主義式歸因：公司是否只在天氣事後惡化時才歸因天氣。
wz = (pl.col("worsen") - pl.col("worsen").mean()) / pl.col("worsen").std()
carriers = sorted(c for c in b["operating_carrier"].unique().to_list() if c != stratum.REF)
w = fit(wx, "worsen_ix", extra=[wz.alias("worsen")] + [
    ((pl.col("operating_carrier") == c).cast(pl.Float64) * wz).alias(f"carrier_{c}:worsen") for c in carriers
])
ix = w.filter(pl.col("term").str.contains(":")).with_columns(
    (pl.col("coef_worsen_ix").abs() / pl.col("se_worsen_ix")).alias("t")
).sort("coef_worsen_ix", descending=True)
print()
print("天氣事後惡化 × 公司 交互項，顯著", ix.filter(pl.col("t") > 1.96).height, "/", ix.height)
print(ix.head(6))
print(ix.tail(3))
