import polars as pl

b = pl.read_parquet("data/processed/bad_weather.parquet").with_columns(
    pl.col("sched_dep_utc").dt.truncate("1mo").alias("ym")
)

reg = b.filter("is_regional")

# 只有替多家主線飛的業者，公司內的主線效應才可識別；單一主線者的執飛效應
# 與主線效應完全共線。
ident = (
    reg.group_by("operating_carrier")
    .agg(pl.col("marketing_carrier").n_unique().alias("n_mkt"), pl.len().alias("n"))
    .sort("n_mkt", "n", descending=True)
)
print(ident)
ident.write_csv("outputs/tables/regional_identifiability.csv")

# 原始歸因率隨主線而變，但這可能只是航網差異——替不同主線飛的機場不同。
raw = (
    reg.group_by("operating_carrier", "marketing_carrier")
    .agg(pl.len().alias("n"), pl.col("y1_attributed").mean().alias("y1"))
    .filter(pl.col("n") >= 1000)
    .sort("operating_carrier", "y1", descending=[False, True])
)
print(raw)
raw.write_csv("outputs/tables/regional_raw.csv")

# 以階段 4 的識別策略排除航網混淆：限定同一執飛業者在同機場同月內同時替
# 多家主線飛的格子，格子內去均值後比較。殘留的差異不可能來自機場或季節。
multi = ident.filter(pl.col("n_mkt") >= 2)["operating_carrier"].to_list()
rows = []
for oc in multi:
    s = reg.filter(pl.col("operating_carrier") == oc)
    cell = s.group_by("origin_iata", "ym").agg(
        pl.col("marketing_carrier").n_unique().alias("nm"), pl.len().alias("cn")
    )
    keep = cell.filter(pl.col("nm") >= 2).select("origin_iata", "ym")
    s = s.join(keep, on=["origin_iata", "ym"])
    if s.height == 0:
        continue
    s = s.with_columns(
        (pl.col("y1_attributed") - pl.col("y1_attributed").mean().over("origin_iata", "ym")).alias("dev")
    )
    rows.append(
        s.group_by("marketing_carrier")
        .agg(pl.len().alias("n"), pl.col("dev").mean().alias("within_cell_dev"))
        .with_columns(pl.lit(oc).alias("operating_carrier"), pl.lit(keep.height).alias("cells"))
    )

within = pl.concat(rows).select("operating_carrier", "marketing_carrier", "cells", "n", "within_cell_dev").sort(
    "operating_carrier", "within_cell_dev"
)
print(within)
within.write_csv("outputs/tables/regional_within_cell.csv")

# 主線自營 vs 其支線代飛：若代飛系統性偏高，對照組 DL 的水準被抬高，
# 階段 4 所有相對 DL 的係數都被平移。
own = (
    b.group_by("marketing_carrier", "is_regional")
    .agg(pl.len().alias("n"), pl.col("y1_attributed").mean().alias("y1"))
    .pivot(on="is_regional", index="marketing_carrier", values=["n", "y1"])
    .filter(pl.col("n_false") + pl.col("n_true") >= 10000)
    .sort("marketing_carrier")
)
print(own)
own.write_csv("outputs/tables/mainline_vs_regional.csv")
