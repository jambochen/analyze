import sys

import polars as pl

sys.path.insert(0, "scripts")
import stratum

# robustness value：未觀測變數與公司、與結果的 partial R² 須同時達到多少，才能使該公司的
# 係數歸零。以氣象共變量與結果的 partial R² 為比較基準。
# t 值取自 cluster-robust 標準誤，公司在層內才有變異，此處的數值只作可行性判斷。
b = stratum.load().drop_nulls(stratum.WX).filter(pl.col("yr") >= 2022)
out, _, df = stratum.fit(b, "y1_attributed", full=True)
f2 = (pl.col("coef") / pl.col("se")) ** 2 / df
out = out.with_columns(
    (0.5 * ((f2**2 + 4 * f2).sqrt() - f2)).alias("rv"),
    (f2 / (1 + f2)).alias("partial_r2"),
)
car = out.filter(pl.col("term").str.starts_with("carrier_")).sort("rv", descending=True)
wx = out.filter(~pl.col("term").str.starts_with("carrier_")).sort("partial_r2", descending=True)
print(car.select("term", "coef", "se", "rv"))
print(wx.select("term", "partial_r2").head(8))
print("殘差自由度", df)
out.write_csv("outputs/tables/sensitivity_rv.csv")
