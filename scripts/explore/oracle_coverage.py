import polars as pl

# findings.md 只寫了「覆蓋率接近滿」，未交代缺的是哪些、為何缺。覆蓋率是結構量
# （由測站觀測制度決定，不隨分析口徑變動），依文件規範可寫進報告，故先算出來。

files = [f"data/raw/flights-weather/flights_weather_{y}.parquet" for y in range(2019, 2025)]
keys = ["sched_dep_utc", "tail_number", "origin_iata"]

cols = pl.scan_parquet(files[0]).collect_schema().names()
orc = [c for c in cols if c.startswith("oracle_")]

# 先套惡劣天氣與申報門檻再取 oracle_，否則 38M 列 × 54 欄的 unique 會吃爆記憶體。
d = (
    pl.scan_parquet(files)
    .filter(pl.col("sched_dep_utc").dt.year().is_between(2019, 2024))
    .filter(pl.col("arr_delay_min") >= 15)
    .filter(
        pl.col("wx_origin_flight_category").is_in(["IFR", "LIFR"])
        | pl.col("wx_dest_flight_category").is_in(["IFR", "LIFR"])
    )
    .select(keys + orc)
    .collect(engine="streaming")
    .unique(subset=keys)
)

# bad_weather 的 oracle_ 已經過 clean.py 的結構性填零，此處要的是原始檔的缺失狀態。
b = pl.read_parquet("data/processed/bad_weather.parquet").drop(pl.selectors.starts_with("oracle_"))
j = b.join(d, on=keys, how="left")
print("稽核樣本", b.height, "join 後", j.height)

cov = (
    pl.DataFrame({"col": orc, "null_frac": [j[c].null_count() / j.height for c in orc]})
    .with_columns(
        pl.col("col").str.extract(r"^(oracle_[a-z]+_[a-z]+)_").alias("group"),
        pl.col("col").str.replace(r"^oracle_[a-z]+_[a-z]+_", "").alias("var"),
    )
    .sort("null_frac", descending=True)
)
print(cov.head(20))
cov.write_csv("outputs/tables/oracle_coverage.csv")

print()
print("依觀測時點分組的缺失率")
print(cov.group_by("group").agg(
    pl.col("null_frac").median().alias("med"), pl.col("null_frac").max().alias("max")
).sort("group"))

# 缺失是否與天氣嚴重度相關：若測站在惡劣天氣時較易漏報，缺觀測的子樣本就不是
# 隨機抽樣，穩健性檢核的外部效度須打折。
j = j.with_columns(pl.col("oracle_origin_dep_flight_category_ordinal").is_null().alias("orc_missing"))
print()
print("核心欄缺失率", j["orc_missing"].mean())
print(
    j.group_by("orc_missing").agg(
        pl.len(),
        pl.col("y1_attributed").mean().alias("y1"),
        pl.col("arr_delay_min").median().alias("delay_med"),
        pl.col("wx_origin_flight_category_ordinal").mean().alias("wx_ord"),
        pl.col("wx_origin_visibility_m").median().alias("vis_med"),
    ).sort("orc_missing")
)

# 事後惡化的分布本身：這是第二判準的原料，須先知道它有多少變異。
# 取 scharr（排定到達時刻）而非 arr——後者觀測時點由航班何時落地決定，與延誤
# 機械相關。見 CLAUDE.md。
j = j.with_columns(
    (pl.col("oracle_origin_dep_flight_category_ordinal") - pl.col("wx_origin_flight_category_ordinal")).alias("worsen_origin"),
    (pl.col("oracle_dest_scharr_flight_category_ordinal") - pl.col("wx_dest_flight_category_ordinal")).alias("worsen_dest"),
)
for c in ["worsen_origin", "worsen_dest"]:
    print()
    print(c, "非零比例", j.select((pl.col(c) != 0).mean()).item())
    print(j.group_by(c).agg(pl.len(), pl.col("y1_attributed").mean().alias("y1")).sort(c))
