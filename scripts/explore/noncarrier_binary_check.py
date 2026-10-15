import polars as pl

CAUSES = [
    "reported_carrier_delay_min",
    "reported_weather_delay_min",
    "reported_nas_delay_min",
    "reported_security_delay_min",
    "reported_late_aircraft_delay_min",
]

b = (
    pl.scan_parquet("data/processed/bad_weather.parquet")
    .select("operating_carrier", "origin_iata", "dest_iata", "sched_dep_utc", *CAUSES)
    .with_columns(pl.sum_horizontal(CAUSES).alias("total"))
    .filter(pl.col("total") > 0)
    .with_columns(
        (1 - pl.col(CAUSES[0]) / pl.col("total")).alias("noncarrier_share"),
        pl.col("sched_dep_utc").dt.truncate("1mo").alias("ym"),
    )
)

# 二分若幾乎全是單一端點，份額模型能辨認的行為就與預想不同。
endpoints = b.select(
    pl.len().alias("flights"),
    (pl.col("noncarrier_share") == 0).mean().alias("all_carrier"),
    (pl.col("noncarrier_share") == 1).mean().alias("all_noncarrier"),
    ((pl.col("noncarrier_share") > 0) & (pl.col("noncarrier_share") < 1)).mean().alias("mixed"),
).collect()
print(endpoints)

# 公司比較必須發生在同層競飛且二分份額有變異的層。
strata = (
    b.group_by("origin_iata", "ym", "operating_carrier")
    .agg(
        pl.col("total").sum().alias("minutes"),
        pl.col(CAUSES[0]).sum().alias("carrier_minutes"),
    )
    .with_columns((1 - pl.col("carrier_minutes") / pl.col("minutes")).alias("unit_share"))
    .group_by("origin_iata", "ym")
    .agg(
        pl.len().alias("carriers"),
        pl.col("unit_share").n_unique().alias("shares"),
    )
    .select(
        (pl.col("carriers") >= 2).sum().alias("competitive"),
        ((pl.col("carriers") >= 2) & (pl.col("shares") >= 2)).sum().alias("competitive_varying"),
    )
    .collect()
)
print(strata)

# 更嚴格的同航線 × 同月比較會減少可用航班，作為雙端固定效應的檢查口徑。
route = (
    b.group_by("origin_iata", "dest_iata", "ym")
    .agg(pl.col("operating_carrier").n_unique().alias("carriers"), pl.len().alias("flights"))
    .select(
        (pl.col("carriers") >= 2).sum().alias("competitive_routes"),
        pl.col("flights").filter(pl.col("carriers") >= 2).sum().alias("competitive_flights"),
    )
    .collect()
)
print(route)

# 原始份額僅用於看二分後的資料形態，未控制航網與氣象，不能當成稽核清單。
carrier = (
    b.group_by("operating_carrier")
    .agg(
        pl.len().alias("flights"),
        pl.col("noncarrier_share").mean().alias("flight_mean"),
        pl.col("total").sum().alias("minutes"),
        pl.col(CAUSES[0]).sum().alias("carrier_minutes"),
    )
    .with_columns((1 - pl.col("carrier_minutes") / pl.col("minutes")).alias("minute_share"))
    .sort("minute_share", descending=True)
    .collect()
)
print(carrier)
carrier.write_csv("outputs/tables/noncarrier_binary_check.csv")
