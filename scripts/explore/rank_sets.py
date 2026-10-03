import sys

import numpy as np
import polars as pl

sys.path.insert(0, "scripts")
import stratum

# 交付物是排序，區間卻是係數的。此處由係數與其共變異矩陣建立 rank 的信賴集合，看集合
# 有多寬：太寬則完整排序無意義，交付物應改為分級。
b = stratum.load().drop_nulls(stratum.WX).filter(pl.col("yr") >= 2022)
out, V, _ = stratum.fit(b, "y1_attributed", full=True)
idx = [i for i, t in enumerate(out["term"].to_list()) if t.startswith("carrier_")]
names = [stratum.REF] + [out["term"][i].removeprefix("carrier_") for i in idx]
K = len(names)
# 基準公司的係數為零且無抽樣變異，與其餘公司的差即該公司的係數本身。
g = np.r_[0.0, out["coef"].to_numpy()[idx]]
S = np.zeros((K, K))
S[1:, 1:] = V[np.ix_(idx, idx)]

se = np.sqrt(np.diag(S)[:, None] + np.diag(S)[None, :] - 2 * S)
np.fill_diagonal(se, np.inf)
t = (g[:, None] - g[None, :]) / se

Z = np.random.default_rng(0).multivariate_normal(np.zeros(K), S, 20000, method="svd")
T = np.abs(Z[:, :, None] - Z[:, None, :]) / se
c_all = np.quantile(T.max(axis=(1, 2)), 0.95)
c_one = np.quantile(T.max(axis=2), 0.95, axis=0)

# rank 1 為係數最大者。確定排在 j 之前的家數給下界，確定排在 j 之後的家數給上界。
res = pl.DataFrame({
    "carrier": names,
    "coef": g,
    "rank": (-g).argsort().argsort() + 1,
    "marginal_lo": 1 + (t.T > c_one[:, None]).sum(1),
    "marginal_hi": K - (t > c_one[:, None]).sum(1),
    "simultaneous_lo": 1 + (t.T > c_all).sum(1),
    "simultaneous_hi": K - (t > c_all).sum(1),
}).sort("rank")
print(res)
print("同時臨界值", round(float(c_all), 3))
print(res.select(
    (pl.col("marginal_hi") - pl.col("marginal_lo") + 1).mean().alias("邊際集合平均大小"),
    (pl.col("simultaneous_hi") - pl.col("simultaneous_lo") + 1).mean().alias("同時集合平均大小"),
))
res.write_csv("outputs/tables/rank_sets.csv")
