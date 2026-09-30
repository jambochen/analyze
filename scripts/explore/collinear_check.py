import sys

import polars as pl

sys.path.insert(0, "scripts")
import stratum

# 共線診斷在舊的控制變數集上做會與正式設定脫節，故一律以 stratum.WX 為準。
b = stratum.load().drop_nulls(stratum.WX)
wx = stratum.WX


def fit(cols, label):
    out = stratum.fit(b, "y1_attributed", controls=cols).select(
        "term", pl.col("coef").alias(f"coef_{label}"), pl.col("se").alias(f"se_{label}")
    )
    print(out.filter(pl.col("term").is_in(["wx_origin_ceiling_m", "wx_origin_sky_cover_oktas"])))
    return out


full = fit(wx, "full")
no_ceil = fit([c for c in wx if c != "wx_origin_ceiling_m"], "no_ceiling")
no_sky = fit([c for c in wx if c != "wx_origin_sky_cover_oktas"], "no_sky")

# 公司效應是交付物。若三個設定下係數幾乎不動，共線就與清單無關。
cmp = (
    full.filter(pl.col("term").str.starts_with("carrier_"))
    .join(no_ceil.filter(pl.col("term").str.starts_with("carrier_")), on="term")
    .join(no_sky.filter(pl.col("term").str.starts_with("carrier_")), on="term")
    .with_columns(
        (pl.col("coef_no_ceiling") - pl.col("coef_full")).abs().alias("d_ceil"),
        (pl.col("coef_no_sky") - pl.col("coef_full")).abs().alias("d_sky"),
    )
    .sort("coef_full", descending=True)
)
print()
print(cmp.select("term", "coef_full", "se_full", "d_ceil", "d_sky"))
print()
print("公司效應最大變動 去雲冪", round(cmp["d_ceil"].max(), 5), "去雲量", round(cmp["d_sky"].max(), 5))
print("對照：公司效應標準誤中位數", round(cmp["se_full"].median(), 5))
cmp.write_csv("outputs/tables/collinear_check.csv")
