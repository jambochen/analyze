import polars as pl

import stratum

b = stratum.load()

# 連續壓力指標：機場當日排定起飛架次相對該機場自身的常態水準。取自班表而非
# 實際營運，不受回報行為影響。以機場為基準避免大小機場的規模差異混進來。
b = b.with_columns(
    (
        pl.col("origin_sched_departures_this_day")
        / pl.col("origin_sched_departures_this_day").mean().over("origin_iata")
    ).alias("load")
).with_columns(((pl.col("load") - pl.col("load").mean()) / pl.col("load").std()).alias("load_z"))
print(b.group_by("era").agg(pl.len(), pl.col("load").mean()).sort("era"))

carriers = sorted(c for c in b["operating_carrier"].unique().to_list() if c != stratum.REF)
is_c = {c: (pl.col("operating_carrier") == c).cast(pl.Float64) for c in carriers}

# 交互項：公司 × 調節變數。era 的主效應在層內恆定（同層同月），去均值後消失，
# 由 stratum.fit 剔除；load 的主效應則保留。
mods = {
    "era": {"break": (pl.col("era") == "break").cast(pl.Float64), "new": (pl.col("era") == "new").cast(pl.Float64)},
    "load": {"z": pl.col("load_z")},
}
for label, mv in mods.items():
    extra = [e.alias(f"{label}_{k}") for k, e in mv.items()]
    extra += [(is_c[c] * e).alias(f"carrier_{c}:{label}_{k}") for c in carriers for k, e in mv.items()]
    for col, suffix in [("y1_attributed", "y1"), ("y2_share", "y2"), ("y3_wx_vs_nas", "y3")]:
        out = stratum.fit(b, col, extra=extra).with_columns((pl.col("coef").abs() / pl.col("se")).alias("t"))
        ix = out.filter(pl.col("term").str.contains(":"))
        print(f"\n=== {label} {suffix} 交互項顯著數 {ix.filter(pl.col('t') > 1.96).height}/{ix.height}")
        print(ix.sort("coef", descending=True).head(8).select("term", "coef", "se", "t"))
        print(ix.sort("coef").head(5).select("term", "coef", "se", "t"))
        out.write_csv(f"outputs/tables/{label}_{suffix}.csv")
