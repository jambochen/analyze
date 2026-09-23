import numpy as np
import polars as pl

b = pl.read_parquet(
    "data/processed/bad_weather.parquet",
    columns=["marketing_carrier", "operating_carrier", "origin_iata", "sched_dep_utc", "y1_attributed"],
).with_columns(pl.col("sched_dep_utc").dt.truncate("1mo").alias("ym"))

# 機場 × 年月格子內去均值，與 04/05 同一套識別策略。剩下的變異才是公司層的。
b = b.with_columns(
    (pl.col("y1_attributed") - pl.col("y1_attributed").mean().over("origin_iata", "ym")).alias("dev")
)

# 主線層 vs 執飛者層：巢狀分解。主線效應取主線內均值，執飛者效應取執飛者
# 在該主線內的偏離。兩者之和加殘差即總變異。
g = b.group_by("marketing_carrier", "operating_carrier").agg(
    pl.len().alias("n"), pl.col("dev").mean().alias("cell_mean")
)
mkt = g.group_by("marketing_carrier").agg(
    ((pl.col("cell_mean") * pl.col("n")).sum() / pl.col("n").sum()).alias("mkt_mean"),
    pl.col("n").sum().alias("mkt_n"),
)
g = g.join(mkt, on="marketing_carrier").with_columns(
    (pl.col("cell_mean") - pl.col("mkt_mean")).alias("op_within_mkt")
)

tot = b["dev"].var()
w = g["n"].to_numpy()
var_mkt = np.average(g["mkt_mean"].to_numpy() ** 2, weights=w)
var_op = np.average(g["op_within_mkt"].to_numpy() ** 2, weights=w)

print("總變異(去格子均值後)", round(tot, 6))
print("主線層", round(var_mkt, 6), "佔比", round(var_mkt / tot, 4))
print("執飛者層(主線內)", round(var_op, 6), "佔比", round(var_op / tot, 4))
print("殘差(航班層)", round(tot - var_mkt - var_op, 6), "佔比", round((tot - var_mkt - var_op) / tot, 4))
print()
print(g.sort("op_within_mkt").select("marketing_carrier", "operating_carrier", "n", "mkt_mean", "op_within_mkt"))

# 只看替多家主線飛的執飛者：這些公司內，主線間的差異與執飛者自身的水準
# 何者大？直接對應「回報規則由誰決定」。
multi = (
    b.group_by("operating_carrier")
    .agg(pl.col("marketing_carrier").n_unique().alias("nm"))
    .filter(pl.col("nm") >= 2)["operating_carrier"]
    .to_list()
)
sub = g.filter(pl.col("operating_carrier").is_in(multi))
print()
print("替多家主線飛者，主線間的全距 vs 執飛者間的全距")
for oc in multi:
    s = sub.filter(pl.col("operating_carrier") == oc)
    if s.height >= 2:
        print(oc, "主線間全距", round(s["cell_mean"].max() - s["cell_mean"].min(), 4), "n", s["n"].sum())
