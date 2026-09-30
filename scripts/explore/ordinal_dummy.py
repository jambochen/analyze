import sys

import numpy as np
import polars as pl

sys.path.insert(0, "scripts")
import stratum

ords = [
    "wx_origin_flight_category_ordinal", "wx_dest_flight_category_ordinal",
    "wx_origin_m3_flight_category_ordinal", "wx_origin_m6_flight_category_ordinal",
]
b = stratum.load().drop_nulls(stratum.WX)


def fit(controls, extra, label):
    out = stratum.fit(b, "y1_attributed", controls=controls, extra=extra)
    print(label, "參數", out.height)
    out = out.select("term", pl.col("coef").alias(f"coef_{label}"), pl.col("se").alias(f"se_{label}"))
    out.write_csv(f"outputs/tables/ordinal_{label}.csv")
    return out


lin = fit(stratum.WX, [], "linear")
# 以 0（VFR，最佳天氣）為基準，其餘三級各自估計，不假設等距。
lv = [(pl.col(c) == v).cast(pl.Float64).alias(f"{c}_lv{v}") for c in ords for v in [1, 2, 3]]
dum = fit([c for c in stratum.WX if c not in ords], lv, "dummy")

print()
print(dum.filter(pl.col("term").str.contains("_lv")))

cmp = (
    lin.filter(pl.col("term").str.starts_with("carrier_"))
    .join(dum.filter(pl.col("term").str.starts_with("carrier_")), on="term")
    .with_columns((pl.col("coef_dummy") - pl.col("coef_linear")).abs().alias("d"))
    .sort("coef_linear", descending=True)
)
print()
print(cmp.select("term", "coef_linear", "coef_dummy", "d", "se_linear"))
print("公司效應最大變動", round(float(cmp["d"].max()), 5),
      "對照 se 中位", round(float(cmp["se_linear"].median()), 5))
print("排序 spearman", round(float(np.corrcoef(cmp["coef_linear"].rank(), cmp["coef_dummy"].rank())[0, 1]), 4))
cmp.write_csv("outputs/tables/ordinal_compare.csv")
