import sys

import polars as pl

sys.path.insert(0, "scripts")
import stratum

# 申報在事後填寫，申報人的資訊集含事件時刻的氣象。三種共變量設定落在同一批航班上，
# 公司主效應才可比較。
b = stratum.load()
orc = [
    o for c in stratum.WX
    for o in [c.replace("wx_origin_", "oracle_origin_dep_").replace("wx_dest_", "oracle_dest_scharr_")]
    if o != c and o in b.columns
]
other = [c for c in stratum.WX if not c.startswith("wx_")]
n0 = b.height
b = b.drop_nulls(stratum.WX + orc)
print("oracle_ 對應欄", len(orc), "完整個案", b.height, "/", n0)

out = None
for tag, controls in [("wx", stratum.WX), ("oracle", other + orc), ("both", stratum.WX + orc)]:
    r = (
        stratum.fit(b, "y1_attributed", controls=controls)
        .filter(pl.col("term").str.starts_with("carrier_"))
        .select("term", pl.col("coef").alias(f"coef_{tag}"), pl.col("se").alias(f"se_{tag}"))
    )
    out = r if out is None else out.join(r, on="term")

print(out.sort("coef_wx", descending=True))
print(out.select(
    pl.corr("coef_wx", "coef_oracle", method="spearman").alias("wx-oracle"),
    pl.corr("coef_wx", "coef_both", method="spearman").alias("wx-both"),
))
out.write_csv("outputs/tables/weather_basis.csv")
pl.DataFrame({"item": ["oracle_ 對應欄數", "稽核樣本", "三組皆完整"], "n": [len(orc), n0, b.height]}).write_csv(
    "outputs/tables/weather_basis_coverage.csv"
)
