import sys

import polars as pl

sys.path.insert(0, "scripts")
import stratum

# 逐時氣象能否把「惡劣與否」換成連續的曝露量：目的地在排定到達前數小時內處於惡劣天氣
# 的時數。時間窗以排定到達時刻界定；以實際到達時刻界定則延誤越久窗越長。
H = 6
ref = pl.read_parquet("data/raw/flights-weather/airports_reference.parquet", columns=["iata_code", "isd_station_id"])
hw = pl.read_parquet(
    "data/raw/flights-weather/airport_hourly_weather.parquet",
    columns=["observation_hour_utc", "isd_station_id", "flight_category_ordinal", "obs_missing"],
)
b = stratum.load().select(
    "sched_dep_utc", "tail_number", "origin_iata", "dest_iata", "operating_carrier", "sched_arr_utc", "y1_attributed",
    "wx_dest_flight_category_ordinal", "oracle_dest_scharr_flight_category_ordinal",
).join(ref, left_on="dest_iata", right_on="iata_code", how="left")
print("目的地對得到測站", b["isd_station_id"].is_not_null().mean())

keys = ["sched_dep_utc", "tail_number", "origin_iata"]
w = (
    b.with_columns(pl.col("sched_arr_utc").dt.truncate("1h").alias("h0"))
    .with_columns(pl.int_ranges(0, H).alias("k")).explode("k")
    .with_columns((pl.col("h0") - pl.duration(hours=pl.col("k"))).alias("observation_hour_utc"))
    .join(hw, on=["isd_station_id", "observation_hour_utc"], how="left")
)
print("時間窗內對得到逐時觀測", w["flight_category_ordinal"].is_not_null().mean())
# 逐時檔應能重現 oracle_dest_scharr_：兩者同為目的地在排定到達時刻所在小時的觀測。
same = w.filter(pl.col("k") == 0).drop_nulls(["flight_category_ordinal", "oracle_dest_scharr_flight_category_ordinal"])
print("排定到達時刻與 oracle_ 一致", (same["flight_category_ordinal"] == same["oracle_dest_scharr_flight_category_ordinal"]).mean())

e = w.group_by(keys).agg(
    (pl.col("flight_category_ordinal") >= 2).sum().alias("bad_hours"),
    pl.col("flight_category_ordinal").is_not_null().sum().alias("obs_hours"),
    pl.col("y1_attributed").first(),
)
dist = e.filter(pl.col("obs_hours") == H).group_by("bad_hours").agg(pl.len().alias("n"), pl.col("y1_attributed").mean().alias("y1")).sort("bad_hours")
print(dist)
dist.write_csv("outputs/tables/hourly_window.csv")
pl.DataFrame({
    "item": ["目的地對得到測站", "時間窗內對得到逐時觀測", "時間窗完整的航班佔比", "排定到達時刻與 oracle_ 一致"],
    "v": [b["isd_station_id"].is_not_null().mean(), w["flight_category_ordinal"].is_not_null().mean(),
          (e["obs_hours"] == H).mean(), (same["flight_category_ordinal"] == same["oracle_dest_scharr_flight_category_ordinal"]).mean()],
}).write_csv("outputs/tables/hourly_window_coverage.csv")
