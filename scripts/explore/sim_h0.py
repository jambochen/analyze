import sys

import numpy as np
import polars as pl
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, "scripts")
import stratum

# 模擬研究的骨架：層、氣象、執飛公司、機尾號全部保留真實資料，只把「是否記入天氣」
# 換成 H0 下的模擬值，量測無偏移時的誤列。此處的 H0 取最簡形式，記入天氣的機率只依層；
# 報告所列較有彈性的生成模型待此骨架確認可行後再換上。
b = stratum.load().drop_nulls(stratum.WX)
p = b.select(pl.col("y1_attributed").mean().over("origin_iata", "ym")).to_series().to_numpy()
# 同一機場同一天的航班共用天氣與航管狀況。以 Gaussian copula 的共同衝擊帶入層內相依，
# 否則各班獨立，cluster-robust 標準誤等於沒被檢驗。
day = b.select(pl.struct("origin_iata", "flight_date_local_origin").rank("dense"))[:, 0].to_numpy() - 1
carriers = sorted(c for c in b["operating_carrier"].unique().to_list() if c != stratum.REF)
R = 100
rng = np.random.default_rng(0)

rows = []
for rho in [0.0, 0.3]:
    for r in range(R):
        u = norm.cdf(np.sqrt(rho) * rng.standard_normal(day.max() + 1)[day] + np.sqrt(1 - rho) * rng.standard_normal(len(p)))
        out = stratum.fit(b.with_columns(pl.Series("y1_attributed", u < p)), "y1_attributed").filter(
            pl.col("term").str.starts_with("carrier_")
        )
        z = (out["coef"] / out["se"]).to_numpy()
        bh = multipletests(2 * norm.sf(np.abs(z)), alpha=0.05, method="fdr_bh")[0]
        rows += [(rho, r, t.removeprefix("carrier_"), float(a), bool(k)) for t, a, k in zip(out["term"], z, bh)]

sim = pl.DataFrame(rows, schema=["rho", "rep", "carrier", "z", "bh"], orient="row")
sim.write_csv("outputs/tables/sim_h0_draws.csv")
by = sim.group_by("rho", "carrier").agg(
    (pl.col("z").abs() > 1.96).mean().alias("reject"), pl.col("z").std().alias("z_sd")
).sort("rho", "reject", descending=[False, True])
print(by)
summ = sim.group_by("rho", "rep").agg(
    (pl.col("z").abs() > 1.96).sum().alias("n_reject"), pl.col("bh").sum().alias("n_bh")
).group_by("rho").agg(
    (pl.col("n_reject").sum() / (R * len(carriers))).alias("type_I_error"),
    (pl.col("n_reject") > 0).mean().alias("任一家被拒絕的比例"),
    (pl.col("n_bh") > 0).mean().alias("BH 後任一家被列入的比例"),
).sort("rho")
print(summ)
by.write_csv("outputs/tables/sim_h0_by_carrier.csv")
summ.write_csv("outputs/tables/sim_h0.csv")
