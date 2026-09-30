import sys

import numpy as np
import polars as pl

sys.path.insert(0, "scripts")
import stratum

b = stratum.load().filter("prior_leg_available")
b = b.with_columns((pl.col("prior_leg_turnaround_min") - pl.col("prior_leg_arr_delay_min")).alias("slack"))
b = b.filter(pl.col("prior_leg_turnaround_min") <= 240)
wx = [c for c in stratum.WX if c not in ("prior_leg_arr_delay_min", "prior_leg_available")]
b = b.drop_nulls(wx)
carriers = sorted(c for c in b["operating_carrier"].unique().to_list() if c != stratum.REF)

# 分位切點是選定值，非資料給定的斷點。交互作用若只在某一個切點成立即不可信。
rows = []
for q in [0.1, 0.15, 0.2, 0.25, 0.3, 0.4]:
    cut = float(np.quantile(b["slack"].to_numpy(), q))
    sq = (pl.col("slack") <= cut).cast(pl.Float64)
    extra = [sq.alias("squeezed")] + [
        ((pl.col("operating_carrier") == c).cast(pl.Float64) * sq).alias(f"sq_x_{c}") for c in carriers
    ]
    t = stratum.fit(b, "y1_attributed", controls=wx, extra=extra).filter(
        pl.col("term").str.starts_with("sq_x_")
    ).with_columns(
        pl.lit(q).alias("q"), pl.lit(cut).alias("cut"),
        pl.col("term").str.replace("sq_x_", "").alias("carrier"),
    )
    rows.append(t)
    oo = t.filter(pl.col("carrier") == "OO")
    print(q, "cut", cut, "OO", round(oo["coef"][0], 5), "lo", round(oo["lo"][0], 5))

a = pl.concat(rows)
a.write_csv("outputs/tables/prior_leg_cutoff.csv")

# 每個切點下，係數為正且區間不含零的公司。結論要可信，這份名單應穩定。
print()
for q in a["q"].unique(maintain_order=True):
    pos = a.filter((pl.col("q") == q) & (pl.col("lo") > 0)).sort("coef", descending=True)
    print(q, pos["carrier"].to_list())
