import numpy as np
import polars as pl
from scipy import stats

b = pl.read_parquet("data/processed/bad_weather.parquet").with_columns(
    pl.col("sched_dep_utc").dt.truncate("1mo").alias("ym")
)

# 一、完美分離：格子內結果全同的公司，其係數會發散，區間不可用。
cell = b.group_by("origin_iata", "ym", "operating_carrier").agg(
    pl.len().alias("n"), pl.col("y1_attributed").mean().alias("p")
)
sep = cell.filter((pl.col("p") == 0) | (pl.col("p") == 1))
print("公司 × 機場 × 月 格子", cell.height, "結果全同者", sep.height, round(sep.height / cell.height, 4))
print("其中 n>=30 者", sep.filter(pl.col("n") >= 30).height)
print(
    cell.group_by("operating_carrier")
    .agg(pl.len().alias("cells"), ((pl.col("p") == 0) | (pl.col("p") == 1)).mean().alias("sep_rate"))
    .sort("sep_rate", descending=True)
    .head(6)
)

# 二、申報分鐘數的尾數：人工填報會湊整到 5、10 的倍數。湊整程度本身即
# 申報品質指標，且不依賴同儕比較。
w = b.filter(pl.col("reported_weather_delay_min") > 0).with_columns(
    (pl.col("reported_weather_delay_min") % 10).alias("last_digit"),
    (pl.col("reported_weather_delay_min") % 5 == 0).alias("mult5"),
)
print("\n尾數分布（天氣分鐘數 > 0）")
dig = w.group_by("last_digit").agg(pl.len().alias("n")).sort("last_digit")
obs = dig["n"].to_numpy()
chi2, pv = stats.chisquare(obs)
print(dig)
print("卡方", round(chi2, 1), "自由度 9", "均勻分布下的期望", int(obs.sum() / 10))
print("5 的倍數佔比", round(float(w["mult5"].mean()), 4), "均勻下應為 0.2")

r = (
    w.group_by("operating_carrier")
    .agg(pl.len().alias("n"), pl.col("mult5").mean().alias("mult5_rate"))
    .filter(pl.col("n") >= 500)
    .sort("mult5_rate", descending=True)
)
print(r)
r.write_csv("outputs/tables/10_digit_heaping.csv")

# 三、oracle_ 子樣本的代表性：該子樣本僅涵蓋部分資料，若氣象分布與全樣本
# 不同，以它做的穩健性結論外部效度有問題。看統計量不看 p 值。
oracle = ["oracle_origin_dep_visibility_m", "oracle_origin_dep_ceiling_m"]
files = [f"data/raw/flights-weather/flights_weather_{y}.parquet" for y in range(2019, 2025)]
keys = ["sched_dep_utc", "tail_number", "origin_iata"]
orc = pl.scan_parquet(files).select(keys + oracle).collect(engine="streaming").unique(subset=keys)
j = b.join(orc, on=keys, how="left").with_columns(
    pl.all_horizontal([pl.col(c).is_not_null() for c in oracle]).alias("has_oracle")
)
print("\noracle 子樣本代表性")
print("覆蓋率", round(float(j["has_oracle"].mean()), 4))
for c in ["wx_origin_visibility_m", "wx_origin_ceiling_m", "wx_origin_flight_category_ordinal", "arr_delay_min"]:
    a = j.filter("has_oracle")[c].drop_nulls().to_numpy()
    bb = j.filter(~pl.col("has_oracle"))[c].drop_nulls().to_numpy()
    sa = np.random.default_rng(0).choice(a, min(20000, len(a)), replace=False)
    sb = np.random.default_rng(0).choice(bb, min(20000, len(bb)), replace=False)
    d = stats.ks_2samp(sa, sb).statistic
    print(f"{c}: KS={d:.4f} 有oracle均值={a.mean():.2f} 無={bb.mean():.2f}")
print("Y1 有oracle", round(float(j.filter("has_oracle")["y1_attributed"].mean()), 4),
      "無", round(float(j.filter(~pl.col("has_oracle"))["y1_attributed"].mean()), 4))
