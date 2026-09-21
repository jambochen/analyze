import polars as pl

b = pl.read_parquet("data/processed/bad_weather.parquet")
print("樣本", b.shape)

# 一、prior_leg 錨點的可行性。
# clean.py 把 null 填 0，故壓力指標須限定 prior_leg_available 才有意義。
print("\n一、prior_leg")
print("覆蓋率", round(float(b["prior_leg_available"].mean()), 4))
p = b.filter("prior_leg_available")
print(p.select(
    pl.col("prior_leg_arr_delay_min").quantile(q).alias(f"delay_q{int(q*100)}") for q in [0.1, 0.25, 0.5, 0.75, 0.9]
))
print(p.select(
    pl.col("prior_leg_turnaround_min").quantile(q).alias(f"turn_q{int(q*100)}") for q in [0.1, 0.25, 0.5, 0.75, 0.9]
))

# 尾機壓力：上一段晚到超過 15 分鐘。與本班天氣歸因率交叉看有無反應。
p = p.with_columns((pl.col("prior_leg_arr_delay_min") >= 15).alias("prior_late"))
print("上一段晚到 >=15min 比例", round(float(p["prior_late"].mean()), 4))
print(p.group_by("prior_late").agg(
    pl.len().alias("n"),
    pl.col("y1_attributed").mean().alias("y1"),
    (pl.col("reported_late_aircraft_delay_min") > 0).mean().alias("late_ac"),
).sort("prior_late"))

# 公司間差異：誰的天氣歸因對尾機壓力不敏感。
d = p.group_by("operating_carrier", "prior_late").agg(pl.len().alias("n"), pl.col("y1_attributed").mean().alias("y1"))
d = d.pivot(on="prior_late", index="operating_carrier", values=["n", "y1"])
print(d.with_columns((pl.col("y1_true") - pl.col("y1_false")).alias("diff")).sort("diff", descending=True))

# 二、五個 reported_ 欄的零值比例，決定走多項 logit 或對數比轉換。
print("\n二、reported 五欄")
rep = [c for c in b.columns if c.startswith("reported_")]
print(b.select((pl.col(c) > 0).mean().alias(c) for c in rep))
print(b.select(pl.col(c).sum().alias(c) for c in rep))
# 同時有幾欄非零：單體的有效維度。
b = b.with_columns(sum((pl.col(c) > 0).cast(pl.Int8) for c in rep).alias("n_nonzero"))
print(b.group_by("n_nonzero").agg(pl.len().alias("n")).sort("n_nonzero"))
# 申報總分鐘數是否等於 arr_delay_min：決定單體的封閉性。
b = b.with_columns(sum(pl.col(c) for c in rep).alias("rep_total"))
print(b.filter(pl.col("rep_total") > 0).select(
    (pl.col("rep_total") == pl.col("arr_delay_min")).mean().alias("total_eq_arrdelay"),
    (pl.col("rep_total") - pl.col("arr_delay_min")).median().alias("median_gap"),
))

# 三、tail_number 跨執飛者的轉移量，決定能否做機內比較。
print("\n三、tail_number")
t = b.filter(pl.col("tail_number").is_not_null())
print("有尾號比例", round(t.height / b.height, 4), "相異尾號", t["tail_number"].n_unique())
per = t.group_by("tail_number").agg(pl.col("operating_carrier").n_unique().alias("k"), pl.len().alias("n"))
print(per.group_by("k").agg(pl.len().alias("tails"), pl.col("n").sum().alias("flights")).sort("k"))
