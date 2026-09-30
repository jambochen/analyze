import polars as pl

import stratum

b = stratum.load()

# 兩個分析單位跑完全相同的流程，係數才可直接對比。差別只在 unit 欄位。
for unit, label in [("operating_carrier", "op"), ("marketing_carrier", "mkt")]:
    for col, suffix in [("y1_attributed", "y1"), ("y2_share", "y2"), ("y3_wx_vs_nas", "y3")]:
        out = stratum.fit(b, col, unit=unit)
        print(label, suffix)
        print(out.filter(pl.col("term").str.starts_with("carrier_")).sort("coef"))
        out.write_csv(f"outputs/tables/{label}_{suffix}.csv")
