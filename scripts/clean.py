import polars as pl

d = pl.read_parquet("data/processed/audit_sample.parquet")

# obs_time_utc 是觀測時間戳記，用於驗證 gate，不是氣象條件。
# obs_missing 與 speci_count_1h 保留：前者標記測站當時無觀測，後者是特殊報次數，
# 天氣劇變時才發，兩者都帶訊號。
d = d.drop([c for c in d.columns if c.endswith("obs_time_utc")])

# 不符條件不填報：無雪不填雪深、無陣風不填陣風。填 0 才是這兩欄的真實語意。
# 另立指示變數，使模型能分辨「確實為 0」與「原本就沒有這個現象」。
for c in [c for c in d.columns if "snow_depth_cm" in c or "wind_gust_ms" in c]:
    d = d.with_columns(pl.col(c).is_null().alias(f"{c}_absent"), pl.col(c).fill_null(0))

# 風向在無風或風向不定時為 null，物理上無定義。轉為正弦餘弦分量避免 0 度與
# 360 度被當成兩端，無風時兩個分量皆為 0。
for c in [c for c in d.columns if c.endswith("wind_dir_deg")]:
    rad = pl.col(c).radians()
    d = d.with_columns(rad.sin().fill_null(0).alias(f"{c}_sin"), rad.cos().fill_null(0).alias(f"{c}_cos"))
d = d.drop([c for c in d.columns if c.endswith("wind_dir_deg")])

# 當日首航段沒有前一航段，null 不是缺失而是「不適用」。
d = d.with_columns(
    pl.col("prior_leg_arr_delay_min").fill_null(0),
    pl.col("prior_leg_turnaround_min").fill_null(0),
).drop("prior_leg_actual_arr_utc")

# 支線業者替主線掛牌飛，回報可能由主線系統決定，階段 4 要檢驗「該查誰」。
d = d.with_columns((pl.col("marketing_carrier") != pl.col("operating_carrier")).alias("is_regional"))

print(d.group_by("operating_carrier").agg(pl.col("is_regional").mean()).sort("is_regional", descending=True))

# 天氣晴朗時談低報天氣延誤沒有意義，主要分析限定於惡劣天氣。
bad = d.filter(
    pl.col("wx_origin_flight_category").is_in(["IFR", "LIFR"])
    | pl.col("wx_dest_flight_category").is_in(["IFR", "LIFR"])
)

# Y1 是否歸因天氣，Y2 歸因時掛多少比例，Y3 天氣佔天氣與 NAS 之和。
# 三者問的是不同問題，分別建模，不合成單一指標。
bad = bad.with_columns(
    (pl.col("reported_weather_delay_min") > 0).alias("y1_attributed"),
    # 未歸因天氣者沒有「掛多少比例」可言，留 null 使其不進 Y2 的統計量。
    pl.when(pl.col("reported_weather_delay_min") > 0)
    .then((pl.col("reported_weather_delay_min") / pl.col("arr_delay_min")).clip(0, 1))
    .alias("y2_share"),
    # 天氣與 NAS 皆為零時歸因邊界不存在，延誤歸在別的欄位。0/0 在 polars 得 NaN
    # 而非 null，不排除會混進統計量。
    pl.when((pl.col("reported_weather_delay_min") + pl.col("reported_nas_delay_min")) > 0)
    .then(
        pl.col("reported_weather_delay_min")
        / (pl.col("reported_weather_delay_min") + pl.col("reported_nas_delay_min"))
    )
    .alias("y3_wx_vs_nas"),
)

print(bad.shape)
print(bad.select(pl.col("y1_attributed").mean(), pl.col("y2_share").median(), pl.col("y3_wx_vs_nas").median()))
print(bad.group_by(pl.col("sched_dep_utc").dt.year().alias("y")).agg(pl.len()).sort("y"))

print(bad.select(pl.col("y2_share").null_count(), pl.col("y3_wx_vs_nas").null_count()))

# y2_share 與 y3_wx_vs_nas 的 null 是結構性的，不在此列。
nulls = {c: bad[c].null_count() / bad.height for c in bad.columns if not c.startswith("y")}
print({c: round(v, 4) for c, v in nulls.items() if v > 0.01})

bad.write_parquet("data/processed/bad_weather.parquet")
d.write_parquet("data/processed/audit_clean.parquet")
