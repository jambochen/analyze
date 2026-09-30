import sys

import numpy as np
import polars as pl

sys.path.insert(0, "scripts")
import simplex
import stratum

d, terms = simplex.data(stratum.load())
d = d.reset_index(drop=True)
m = simplex.fit(d, terms)
dropped = np.asarray(m.na_index, dtype=int)
print("移除列數", len(dropped), "佔", round(len(dropped) / len(d), 4))

d["dropped"] = d.index.isin(dropped)
k = pl.from_pandas(d[["operating_carrier", "cause", "minutes", "n_flights", "dropped"]])

# separation 若集中在特定公司，該公司的係數就是以殘餘樣本估的，須另行說明。
print(k.group_by("operating_carrier").agg(
    pl.len().alias("n"), pl.col("dropped").sum().alias("drop"), pl.col("dropped").mean().alias("rate")
).sort("rate", descending=True))
print(k.group_by("cause").agg(
    pl.len().alias("n"), pl.col("dropped").sum().alias("drop"), pl.col("dropped").mean().alias("rate")
).sort("rate", descending=True))
print("被移除者的分鐘數", k.filter("dropped").select(
    (pl.col("minutes") == 0).mean().alias("zero_share"), pl.col("minutes").median().alias("med")))
print("被移除者的單元航班數中位", k.filter("dropped")["n_flights"].median(),
      "保留者", k.filter(~pl.col("dropped"))["n_flights"].median())
