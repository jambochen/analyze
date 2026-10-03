import sys

import polars as pl

sys.path.insert(0, "scripts")
import stratum

# 兩個以 oracle_ 導出的限制：天氣惡化時應更常記入天氣；記入天氣的機率應隨實際天氣的
# 嚴重度遞增。先看各公司在各水準上的班次是否足以檢定。
o, w = "oracle_origin_dep_flight_category_ordinal", "oracle_dest_scharr_flight_category_ordinal"
b = stratum.load().drop_nulls(stratum.WX + [o, w]).with_columns(
    pl.max_horizontal(o, w).alias("severity"),
    pl.max_horizontal(
        pl.col(o) - pl.col("wx_origin_flight_category_ordinal"), pl.col(w) - pl.col("wx_dest_flight_category_ordinal")
    ).sign().alias("change"),
)

for col in ["change", "severity"]:
    c = b.group_by("operating_carrier", col).agg(pl.len().alias("n"), pl.col("y1_attributed").mean().alias("y1"))
    print(c.pivot(col, index="operating_carrier", values="n", sort_columns=True).sort("operating_carrier"))
    print(col, "最小格", c["n"].min(), "少於 100 班的格數", (c["n"] < 100).sum(), "/", c.height)
    c.write_csv(f"outputs/tables/restrictions_oracle_{col}.csv")

carriers = sorted(c for c in b["operating_carrier"].unique().to_list() if c != stratum.REF)
# 主效應與交互同時入模：交互才是各公司對嚴重度的反應差異。
extra = [pl.col("severity")] + [
    ((pl.col("operating_carrier") == c).cast(pl.Float64) * pl.col("severity")).alias(f"sev_x_{c}") for c in carriers
]
out = stratum.fit(b, "y1_attributed", extra=extra)
print(out.filter(pl.col("term").str.starts_with("sev")).sort("coef"))
out.write_csv("outputs/tables/restrictions_oracle_fit.csv")
