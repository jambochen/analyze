import matplotlib.pyplot as plt
import polars as pl

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK TC", "Noto Sans CJK SC", "Noto Sans CJK JP"]
plt.rcParams["axes.unicode_minus"] = False

b = pl.read_parquet("data/processed/bad_weather.parquet")

# COVID 切點取自航班量的實際崩跌與回復，不用行政上的疫情起訖日期。
# 須用原始檔的總航班量，稽核樣本的量受當年天氣影響，不是航班量的代理。
raw = [f"data/raw/flights-weather/flights_weather_{y}.parquet" for y in range(2019, 2025)]
m = (
    pl.scan_parquet(raw)
    .select("sched_dep_utc")
    .filter(pl.col("sched_dep_utc").dt.year().is_between(2019, 2024))
    .group_by(pl.col("sched_dep_utc").dt.truncate("1mo").alias("mo"))
    .agg(pl.len().alias("n"))
    .sort("mo")
    .collect(engine="streaming")
)

# 與同月的 2019 相比，季節性自動抵銷。用年均當基準會把 9 月的季節性低點
# 誤判為崩跌。
base19 = m.filter(pl.col("mo").dt.year() == 2019).with_columns(pl.col("mo").dt.month().alias("mth")).select("mth", pl.col("n").alias("n19"))
m = m.with_columns(pl.col("mo").dt.month().alias("mth")).join(base19, on="mth").sort("mo").with_columns((pl.col("n") / pl.col("n19")).alias("vs19"))
print(m.filter(pl.col("mo").dt.year() == 2020).select("mo", "vs19").head(6))
print(m.filter(pl.col("vs19") > 0.85).filter(pl.col("mo") > pl.lit("2020-04-01").str.to_datetime().dt.replace_time_zone("UTC")).select("mo", "vs19").head(3))

fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(m["mo"], m["vs19"])
for cut in ["2020-04-01", "2021-06-01"]:
    ax.axvline(pl.Series([cut]).str.to_datetime().dt.replace_time_zone("UTC")[0], ls="--", lw=0.8, color="gray")
ax.set_ylabel("航班量相對 2019 年同月")
fig.tight_layout()
fig.savefig("outputs/figures/03_covid_volume.png", dpi=150)

# 四個成因欄位的歸因分配。天氣與 NAS 的邊界是本分析的核心歧義。
causes = ["reported_weather_delay_min", "reported_nas_delay_min", "reported_carrier_delay_min", "reported_late_aircraft_delay_min"]
share = b.group_by("operating_carrier").agg(
    pl.len().alias("n"),
    *[(pl.col(c).sum() / pl.sum_horizontal([pl.col(x).sum() for x in causes])).alias(c.replace("reported_", "").replace("_delay_min", "")) for c in causes],
).filter(pl.col("n") > 5000).sort("weather")
print(share)

fig, ax = plt.subplots(figsize=(9, 4.5))
bottom = [0.0] * share.height
for c in ["weather", "nas", "carrier", "late_aircraft"]:
    ax.bar(share["operating_carrier"], share[c], bottom=bottom, label=c)
    bottom = [a + b_ for a, b_ in zip(bottom, share[c])]
ax.legend(ncols=4, fontsize=8)
ax.set_ylabel("延誤分鐘數佔比")
fig.tight_layout()
fig.savefig("outputs/figures/03_cause_share.png", dpi=150)

# Y2 在 OO 與 YV 的中位數為 1.0，須確認是離散尖峰而非連續分布——
# 若是尖峰，Y2 的模型須用分數型而非常態。
y2 = b.filter(pl.col("y2_share").is_not_null())
print(y2.select((pl.col("y2_share") == 1).mean().alias("frac_1"), (pl.col("y2_share") > 0.99).mean().alias("frac_gt99")))
print(y2.group_by("operating_carrier").agg(pl.len().alias("n"), (pl.col("y2_share") == 1).mean().alias("frac_1")).filter(pl.col("n") > 2000).sort("frac_1", descending=True))

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.hist(y2["y2_share"], bins=50)
ax.set_xlabel("y2_share（天氣分鐘數佔總延誤）")
fig.tight_layout()
fig.savefig("outputs/figures/03_y2_hist.png", dpi=150)

# 分層樣本量決定哪些格子須池化。
print(b.group_by(["operating_carrier", pl.col("sched_dep_utc").dt.year().alias("y")]).agg(pl.len().alias("n")).pivot("y", index="operating_carrier", values="n").sort("operating_carrier"))

cell = b.group_by(["operating_carrier", "origin_iata"]).agg(pl.len().alias("n"))
print(cell.select(pl.len().alias("cells"), (pl.col("n") < 30).mean().alias("frac_lt30"), pl.col("n").median()))

share.write_csv("outputs/tables/03_cause_share.csv")
