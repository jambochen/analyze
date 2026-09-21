import polars as pl

files = [f"data/raw/flights-weather/flights_weather_{y}.parquet" for y in range(2019, 2025)]

cols = pl.scan_parquet(files[0]).collect_schema().names()

# bts_pipeline 套件未隨 Kaggle 資料發布，select_safe_features() 無法呼叫。
# 依資料字典的兩條規則手動重建：oracle_ 為事件時刻氣象，以下 14 欄為事件後才確定。
# prior_leg_* 不在此列——發布者的 gate 已保證前一航段實際到站早於本班 info_cutoff_utc。
post_event = [
    "dep_delay_min",
    "arr_delay_min",
    "taxi_out_min",
    "taxi_in_min",
    "actual_elapsed_min",
    "air_time_min",
    "cancelled",
    "diverted",
    "arr_del15",
    "cancellation_code",
    "cancellation_reason",
    "actual_dep_utc",
    "actual_arr_utc",
    "time_identity_violation",
]
reported = [c for c in cols if c.startswith("reported_")]
features = [c for c in cols if not c.startswith(("oracle_", "reported_")) and c not in post_event]

print("features", len(features), "reported", len(reported))

# arr_delay_min 與 cancelled/diverted 僅供界定樣本與檢核，不進特徵矩陣。
df = (
    pl.scan_parquet(files)
    .select(features + reported + ["arr_delay_min", "cancelled", "diverted"])
    .filter(pl.col("sched_dep_utc").dt.year().is_between(2019, 2024))
    .filter(pl.col("arr_delay_min") >= 15)
    .collect(engine="streaming")
)

assert not [c for c in df.columns if c.startswith("oracle_")]

print(df.shape)
print(df.group_by(pl.col("sched_dep_utc").dt.year().alias("y")).agg(pl.len()).sort("y"))
print(df.select(pl.col("cancelled").sum(), pl.col("diverted").sum()))
print(df.select((pl.col("reported_weather_delay_min") > 0).mean()))

df.write_parquet("data/processed/audit_sample.parquet")
