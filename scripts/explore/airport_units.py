import sys

import polars as pl

sys.path.insert(0, "scripts")
import stratum

# 機場層級清單的單位是公司 × 機場。先看單位的大小與 separation 有多普遍，
# 以及一座機場內有幾家公司可供比較。
b = stratum.load().filter(pl.col("yr") >= 2022)
u = b.group_by("operating_carrier", "origin_iata").agg(
    pl.len().alias("n"), pl.col("y1_attributed").mean().alias("p")
)
ap = u.group_by("origin_iata").agg(pl.len().alias("nc"), pl.col("n").sum().alias("n"))
out = pl.DataFrame({
    "item": ["公司 × 機場單位", "單位班次中位數", "班次少於 30 的單位佔比", "結果全同的單位佔比",
             "結果全同單位的班次佔比", "機場數", "兩家以上公司的機場佔比", "兩家以上公司機場的班次佔比"],
    "v": pl.Series([
        u.height, u["n"].median(), (u["n"] < 30).mean(), u["p"].is_in([0.0, 1.0]).mean(),
        u.filter(pl.col("p").is_in([0.0, 1.0]))["n"].sum() / b.height,
        ap.height, (ap["nc"] >= 2).mean(), ap.filter(pl.col("nc") >= 2)["n"].sum() / b.height,
    ], dtype=pl.Float64, strict=False),
})
print(out)
out.write_csv("outputs/tables/airport_units.csv")
