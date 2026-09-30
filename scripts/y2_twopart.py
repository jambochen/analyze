import polars as pl

import stratum

b = stratum.load().drop_nulls(stratum.WX).filter(pl.col("y2_share").is_not_null())

# Y2 的 1.0 是「整段延誤全掛天氣」，是二元決策的結果而非連續分布的邊界堆積。
# 拆成兩個決策分別估：是否整段全掛、若非整段則掛多少。
b = b.with_columns(
    (pl.col("y2_share") >= 1.0).alias("is_one"),
    pl.when(pl.col("y2_share") < 1.0).then(pl.col("y2_share")).alias("partial"),
)
print("Y2 有效", b.height, "其中 1.0 佔", round(b["is_one"].mean(), 4))
print(
    b.group_by("operating_carrier")
    .agg(pl.len().alias("n"), pl.col("is_one").mean().alias("p_one"), pl.col("partial").mean().alias("mean_partial"))
    .filter(pl.col("n") >= 500)
    .sort("p_one", descending=True)
)

p1 = stratum.fit(b, "is_one")
p2 = stratum.fit(b, "partial")
for out, label in [(p1, "part1_is_one"), (p2, "part2_partial")]:
    print(f"\n=== {label}")
    print(out.filter(pl.col("term").str.starts_with("carrier_")).sort("coef", descending=True))
    out.write_csv(f"outputs/tables/y2_{label}.csv")

# 兩部的公司效應方向是否一致。不一致即表示合併估的單一係數在混合兩種行為。
cmp = (
    p1.filter(pl.col("term").str.starts_with("carrier_"))
    .select("term", pl.col("coef").alias("coef_is_one"), pl.col("se").alias("se_is_one"))
    .join(
        p2.filter(pl.col("term").str.starts_with("carrier_")).select(
            "term", pl.col("coef").alias("coef_partial"), pl.col("se").alias("se_partial")
        ),
        on="term",
    )
    .with_columns(
        (pl.col("coef_is_one").sign() != pl.col("coef_partial").sign()).alias("方向相反"),
    )
    .sort("coef_is_one", descending=True)
)
print()
print(cmp)
print("方向相反者", cmp["方向相反"].sum(), "/", cmp.height)
cmp.write_csv("outputs/tables/y2_twopart_compare.csv")
