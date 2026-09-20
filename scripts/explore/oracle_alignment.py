import numpy as np
import polars as pl
import statsmodels.api as sm

# oracle_ 是事件時刻的氣象。用於預測延誤即構成洩漏，但本題不預測延誤——估的是
# 給定氣象下的歸因傾向，屬發布者明說的 retrospective 用途。
# 用它的理由：航空公司是事後填申報表，其資訊集接近 oracle_ 而非 wx_（排定
# 起飛前）。以 wx_ 為唯一基準，等於拿起飛前可知的天氣評斷事後填的表。
oracle = [
    "oracle_origin_dep_visibility_m", "oracle_origin_dep_ceiling_m", "oracle_origin_dep_wind_speed_ms",
    "oracle_origin_dep_wind_gust_ms", "oracle_origin_dep_precip_1h_mm", "oracle_origin_dep_temp_c",
    "oracle_origin_dep_flight_category_ordinal", "oracle_origin_dep_thunderstorm", "oracle_origin_dep_snow",
    "oracle_origin_dep_freezing_precip", "oracle_origin_dep_fog",
    "oracle_dest_arr_visibility_m", "oracle_dest_arr_ceiling_m", "oracle_dest_arr_wind_speed_ms",
    "oracle_dest_arr_precip_1h_mm", "oracle_dest_arr_flight_category_ordinal",
    "oracle_dest_arr_thunderstorm", "oracle_dest_arr_snow", "oracle_dest_arr_freezing_precip",
    "oracle_dest_arr_fog",
]
keys = ["sched_dep_utc", "tail_number", "origin_iata"]

files = [f"data/raw/flights-weather/flights_weather_{y}.parquet" for y in range(2019, 2025)]
orc = pl.scan_parquet(files).select(keys + oracle).collect(engine="streaming")

b = pl.read_parquet("data/processed/bad_weather.parquet").with_columns(
    pl.col("sched_dep_utc").dt.truncate("1mo").alias("ym")
)
n0 = b.height
b = b.join(orc.unique(subset=keys), on=keys, how="inner")
print("join 保留", b.height, "/", n0)

wx = [
    "wx_origin_visibility_m", "wx_origin_ceiling_m", "wx_origin_wind_speed_ms", "wx_origin_wind_gust_ms",
    "wx_origin_precip_1h_mm", "wx_origin_snow_depth_cm", "wx_origin_temp_c", "wx_origin_sky_cover_oktas",
    "wx_origin_flight_category_ordinal", "wx_dest_flight_category_ordinal", "wx_dest_visibility_m",
    "wx_dest_ceiling_m", "wx_dest_wind_speed_ms", "wx_dest_precip_1h_mm",
    "wx_origin_thunderstorm", "wx_origin_snow", "wx_origin_freezing_precip", "wx_origin_fog",
    "wx_origin_m3_flight_category_ordinal", "wx_origin_m6_flight_category_ordinal",
    "arr_delay_min", "distance_miles", "origin_sched_departures_this_hour", "prior_leg_arr_delay_min",
]
ctrl = ["arr_delay_min", "distance_miles", "origin_sched_departures_this_hour", "prior_leg_arr_delay_min"]
b = b.drop_nulls(wx + oracle)
print("drop_nulls 後", b.height)

# 起飛後的天氣變化：事件時刻減排定起飛前。正值為惡化（ordinal 越大越差）。
b = b.with_columns(
    (pl.col("oracle_origin_dep_flight_category_ordinal") - pl.col("wx_origin_flight_category_ordinal")).alias("worsen")
)
print(b.group_by("worsen").agg(pl.len(), pl.col("y1_attributed").mean()).sort("worsen"))

cell = b.group_by("origin_iata", "ym").agg(
    pl.col("operating_carrier").n_unique().alias("nc"), pl.col("y1_attributed").mean().alias("cr")
)
keep = cell.filter((pl.col("nc") >= 2) & (pl.col("cr") > 0) & (pl.col("cr") < 1)).select("origin_iata", "ym")
b = b.join(keep, on=["origin_iata", "ym"])
carriers = sorted(c for c in b["operating_carrier"].unique().to_list() if c != "DL")
cell_id = b.select(pl.struct("origin_iata", "ym").rank("dense").alias("i"))["i"].to_numpy() - 1
car = {c: (b["operating_carrier"] == c).to_numpy().astype(float) for c in carriers}


def fit(num, label, extra=None):
    Xn = b.select(num).to_numpy().astype(float)
    Xn = (Xn - Xn.mean(0)) / (Xn.std(0) + 1e-9)
    cols, names = [Xn], list(num)
    for c in carriers:
        cols.append(car[c].reshape(-1, 1))
        names.append(f"carrier_{c}")
    if extra is not None:
        ev = (extra - extra.mean()) / extra.std()
        for c in carriers:
            cols.append((car[c] * ev).reshape(-1, 1))
            names.append(f"carrier_{c}:worsen")
        cols.append(ev.reshape(-1, 1))
        names.append("worsen")
    X = np.column_stack(cols)
    y = b["y1_attributed"].to_numpy().astype(float)
    dm = pl.DataFrame(X, schema=names).with_columns(pl.Series("g", cell_id), pl.Series("y", y))
    dd = dm.select([(pl.col(c) - pl.col(c).mean().over("g")).alias(c) for c in names + ["y"]])
    M = dd.select(names).to_numpy()
    nz = M.std(0) > 1e-10
    M, nm = M[:, nz], [n for n, k in zip(names, nz) if k]
    r = sm.OLS(dd["y"].to_numpy(), M).fit(cov_type="cluster", cov_kwds={"groups": cell_id})
    assert not np.isnan(r.bse).any(), f"{label} 標準誤含 NaN"
    out = pl.DataFrame({"term": nm, f"coef_{label}": r.params, f"se_{label}": r.bse})
    print(label, "R2", round(r.rsquared, 5))
    out.write_csv(f"outputs/tables/07_{label}.csv")
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
cmp.write_csv("outputs/tables/07_basis_compare.csv")
print()
print("排序相關 spearman", round(
    np.corrcoef(cmp["coef_wx_base"].rank(), cmp["coef_oracle_base"].rank())[0, 1], 4))
print("最大位移", round(cmp["shift"].abs().max(), 4), "對照 se 中位", round(cmp["se_wx_base"].median(), 4))

# 機會主義式歸因：公司是否只在天氣事後惡化時才歸因天氣。
w = fit(wx, "worsen_ix", extra=b["worsen"].to_numpy().astype(float))
ix = w.filter(pl.col("term").str.contains(":")).with_columns(
    (pl.col("coef_worsen_ix").abs() / pl.col("se_worsen_ix")).alias("t")
).sort("coef_worsen_ix", descending=True)
print()
print("天氣事後惡化 × 公司 交互項，顯著", ix.filter(pl.col("t") > 1.96).height, "/", ix.height)
print(ix.head(6))
print(ix.tail(3))
