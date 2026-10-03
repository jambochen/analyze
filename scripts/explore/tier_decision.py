import sys

import numpy as np
import polars as pl
from scipy.stats import norm

sys.path.insert(0, "scripts")
import stratum

# rank 的信賴集合在清單中段太寬，交付物改為分級。分幾級、切在哪取決於損失函數，
# 此處看分級結果對損失參數有多敏感：若級別隨參數大幅變動，分級本身就沒有依據。
b = stratum.load().drop_nulls(stratum.WX).filter(pl.col("yr") >= 2022)
out, V, _ = stratum.fit(b, "y1_attributed", full=True)
idx = [i for i, t in enumerate(out["term"].to_list()) if t.startswith("carrier_")]
names = [stratum.REF] + [out["term"][i].removeprefix("carrier_") for i in idx]
K = len(names)
g = np.r_[0.0, out["coef"].to_numpy()[idx]]
S = np.zeros((K, K))
S[1:, 1:] = V[np.ix_(idx, idx)]

# 基準公司的係數沒有抽樣變異，直接收縮會把它當成量得最準的一家。改為對全體平均的
# 離差，每家都有自己的變異數。
C = np.eye(K) - 1 / K
theta = C @ g
s2 = np.diag(C @ S @ C.T)
tau2 = max(float(np.mean(theta**2 - s2)), 0.0)
m = tau2 / (tau2 + s2) * theta
v = tau2 * s2 / (tau2 + s2)
print("tau", np.sqrt(tau2), "收縮係數", (s2 / (tau2 + s2)).min(), (s2 / (tau2 + s2)).max())

rows = []

# tail probability 規則：目標是先驗分布兩端各 alpha 的公司，依後驗落在該尾的機率
# 由高到低選入，直到入選者的平均誤列機率超過 gamma。
for alpha in (0.1, 0.2, 0.33):
    cut = np.sqrt(tau2) * norm.ppf(1 - alpha)
    hi = 1 - norm.cdf((cut - m) / np.sqrt(v))
    lo = norm.cdf((-cut - m) / np.sqrt(v))
    for gamma in (0.05, 0.1, 0.2):
        tier = np.full(K, "mid", dtype=object)
        for p, tag in ((hi, "high"), (lo, "low")):
            o = np.argsort(-p)
            keep = np.cumsum(1 - p[o]) / np.arange(1, K + 1) <= gamma
            # 累積平均不單調，取最後一個仍滿足限制的位置才是最大的入選集合。
            n = keep.nonzero()[0].max() + 1 if keep.any() else 0
            tier[o[:n]] = tag
        rows += [("tail", f"alpha={alpha},gamma={gamma}", names[i], tier[i], hi[i], lo[i]) for i in range(K)]

# pairwise 損失：把 i 排在 j 之上，排對得 1 - lam、排錯失 lam，同級不計。期望報酬為
# pi_ij - lam。只在依後驗平均排序後的相鄰切分中搜尋，級內成對報酬之和最小者即最佳。
o = np.argsort(-m)
pi = norm.cdf((m[o][:, None] - m[o][None, :]) / np.sqrt(v[o][:, None] + v[o][None, :]))
for lam in (0.5, 0.7, 0.8, 0.9, 0.95):
    w = np.triu(pi - lam, 1)
    seg = np.array([[w[a:e, a:e].sum() if e > a else 0.0 for e in range(K + 1)] for a in range(K + 1)])
    best = np.full(K + 1, np.inf)
    best[0] = 0.0
    prev = np.zeros(K + 1, dtype=int)
    for e in range(1, K + 1):
        cand = best[:e] + seg[:e, e]
        prev[e] = cand.argmin()
        best[e] = cand.min()
    cuts, e = [], K
    while e > 0:
        cuts.append(prev[e])
        e = prev[e]
    grade = np.searchsorted(sorted(cuts), np.arange(K), side="right")
    rows += [("pairwise", f"lam={lam}", names[o[r]], str(grade[r]), np.nan, np.nan) for r in range(K)]

res = pl.DataFrame(rows, schema=["rule", "param", "carrier", "tier", "p_high", "p_low"], orient="row")
res.write_csv("outputs/tables/tier_decision.csv")

post = pl.DataFrame({"carrier": names, "theta": theta, "se": np.sqrt(s2), "post_mean": m, "post_sd": np.sqrt(v)})
post.sort("post_mean", descending=True).write_csv("outputs/tables/tier_decision_posterior.csv")

wide = res.pivot(on="param", index="carrier", values="tier").join(post.select("carrier", "post_mean"), on="carrier")
with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=300):
    print(wide.sort("post_mean", descending=True))
    print(res.group_by("rule", "param", maintain_order=True).agg(pl.col("tier").n_unique().alias("levels"),
                                                              (pl.col("tier") == "high").sum().alias("n_high"),
                                                              (pl.col("tier") == "low").sum().alias("n_low")))
