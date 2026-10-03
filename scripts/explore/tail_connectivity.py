import sys

import polars as pl

sys.path.insert(0, "scripts")
import stratum

# 加入機尾號固定效應後，公司主效應只由曾為多家公司執飛的飛機識別。先看這類飛機有多少、
# 把哪些公司連在一起。
b = stratum.load()
t = b.group_by("tail_number").agg(
    pl.col("operating_carrier").n_unique().alias("nc"),
    pl.col("operating_carrier").unique().alias("carriers"),
    pl.len().alias("n"),
)
multi = t.filter(pl.col("nc") >= 2)
print("機尾號", t.height, "跨公司者", multi.height, "其班次佔比", round(multi["n"].sum() / b.height, 4))

comp = {c: c for c in b["operating_carrier"].unique().to_list()}


def root(c):
    while comp[c] != c:
        c = comp[c]
    return c


for cs in multi["carriers"].to_list():
    for c in cs[1:]:
        comp[root(c)] = root(cs[0])
groups = {}
for c in comp:
    groups.setdefault(root(c), []).append(c)
print("連通成分", sorted(groups.values(), key=len, reverse=True))

pairs = (
    multi.explode("carriers").join(multi.explode("carriers"), on="tail_number")
    .filter(pl.col("carriers") < pl.col("carriers_right"))
    .group_by("carriers", "carriers_right").agg(pl.len().alias("shared_tails"))
    .sort("shared_tails", descending=True)
)
print(pairs)
pairs.write_csv("outputs/tables/tail_connectivity_pairs.csv")
pl.DataFrame({
    "item": ["機尾號", "跨公司的機尾號", "跨公司機尾號的班次佔比", "連通成分數", "最大成分的公司數"],
    "v": pl.Series([t.height, multi.height, multi["n"].sum() / b.height, len(groups), max(len(g) for g in groups.values())], dtype=pl.Float64, strict=False),
}).write_csv("outputs/tables/tail_connectivity.csv")
