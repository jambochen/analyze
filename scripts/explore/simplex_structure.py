import polars as pl

b = pl.read_parquet("data/processed/bad_weather.parquet")
rep = ["reported_carrier_delay_min", "reported_weather_delay_min", "reported_nas_delay_min",
       "reported_security_delay_min", "reported_late_aircraft_delay_min"]
short = ["carrier", "weather", "nas", "security", "late_ac"]

b = b.with_columns(sum(pl.col(c) for c in rep).alias("tot"))
print("申報總分鐘為 0 的比例", round(float((b["tot"] == 0).mean()), 4))
b = b.filter(pl.col("tot") > 0)
print("有申報樣本", b.height)

# 非零型態：單體的哪些面被實際使用。決定多項 logit 的類別要怎麼定。
pat = b.select(
    pl.concat_str([(pl.col(c) > 0).cast(pl.Int8).cast(pl.Utf8) for c in rep]).alias("pattern")
)
pc = pat.group_by("pattern").agg(pl.len().alias("n")).sort("n", descending=True)
pc = pc.with_columns((pl.col("n") / pat.height).alias("share"))
print(pc.head(15))
print("相異型態數", pc.height)

# security 的量體：能否剔除。
print("\nsecurity 非零比例", round(float((b["reported_security_delay_min"] > 0).mean()), 5))
print("security 佔總申報分鐘", round(float(b["reported_security_delay_min"].sum() / b["tot"].sum()), 5))

# 各欄佔比的分布：若以比例為結果變數，零堆積有多嚴重。
for c, s in zip(rep, short):
    r = (pl.col(c) / pl.col("tot"))
    q = b.select(r.alias("r")).select(
        (pl.col("r") == 0).mean().alias("zero"), (pl.col("r") == 1).mean().alias("one"),
        pl.col("r").mean().alias("mean"),
    )
    print(s, q.to_dicts()[0])

# 分鐘數加總後在公司間的分配差異：多項 logit 要估的就是這個。
tot_by = b.group_by("operating_carrier").agg(
    [ (pl.col(c).sum() / pl.col("tot").sum()).alias(s) for c, s in zip(rep, short) ] + [pl.len().alias("n")]
)
print(tot_by.sort("weather", descending=True))
