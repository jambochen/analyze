import os

# 四個行程同時配適，各套件再各開滿執行緒只會互搶核心。配適的時間花在 rayon 的層內轉換，
# 故只留給它每行程五條，其餘設為一。須在 import 這些套件之前設定。
for v in ["NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "POLARS_MAX_THREADS"]:
    os.environ[v] = "1"
os.environ["RAYON_NUM_THREADS"] = "5"

import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.optimize import brentq

sys.path.insert(0, "scripts")
import simplex
import stratum

# 單元固定效應與層 × 成因固定效應重疊，而每層的公司數與成因數都不隨層數增加，
# 屬 Weidner 與 Zylkin 所指 FE-PPML 不一致的結構。理論只說偏誤不消失，沒說多大；
# 此處保留真實的單元、層、公司與氣象，只換掉分鐘數，量偏誤相對於標準誤的大小。
R = 20
THETA = 0.3
PLANT = ["OO", "B6", "HA"]

d, terms = simplex.data(stratum.load())
d = d.reset_index(drop=True)
wterms = [t for t in terms if t.startswith("w_")]

# 去掉公司 × 成因項的配適值作為生成均值，此時所有 theta 的真值為零。
# 被 separation 移除的列沒有配適值，其均值即零，模擬時維持為零，重估時會再被移除。
m0 = simplex.fit(d, wterms)
kept = ~d.index.isin(np.asarray(m0.na_index, dtype=int))
mu = np.zeros(len(d))
mu[kept] = m0.predict(type="response")
del m0
y = d["minutes"].to_numpy()

# 每單元僅四列，固定效應個數與觀測數同量級，Pearson 統計量不扣自由度會低估離散。
p = d.loc[kept, "unit"].nunique() + d.loc[kept, "cellcause"].nunique() + len(wterms)
phi = ((y[kept] - mu[kept]) ** 2 / mu[kept]).sum() / (kept.sum() - p)
# gamma-Poisson 下 Var = mu + alpha mu^2，Pearson 離散的期望值為 1 + alpha mean(mu)。
alpha_pearson = (phi - 1) / mu[kept].mean()
# Pearson 口徑配出的零值比例遠低於真實資料，另以零值比例配一個較重的離散作為上界。
zero = (y[kept] == 0).mean()
alpha_zero = brentq(lambda a: ((1 + a * mu[kept]) ** (-1 / a)).mean() - zero, 0.01, 100)
print("Pearson phi", phi, "alpha_pearson", alpha_pearson, "真實零值比例", zero, "alpha_zero", alpha_zero)

planted = np.exp(THETA * d[[f"x_{c}_weather" for c in PLANT]].sum(axis=1).to_numpy())
SCEN = {
    "null_poisson": (0, False),
    "null_nb": (alpha_pearson, False),
    "null_nb_zero": (alpha_zero, False),
    "planted_nb": (alpha_pearson, True),
}

# 理論預期每層公司數愈少偏誤愈大，以公司數少與多的兩個子樣本各自重做虛無情境。
nc = d.groupby(["origin_iata", "ym"])["operating_carrier"].transform("nunique").to_numpy()
ROWS = {"all": nc >= 2, "few": nc <= 3, "many": nc >= 10}
TERMS = {k: [t for t in terms if d.loc[r, t].abs().sum() > 0] for k, r in ROWS.items()}
units = pd.concat(
    [d[r].groupby("operating_carrier")["unit"].nunique().rename("n_units").reset_index().assign(sample=k)
     for k, r in ROWS.items()]
).rename(columns={"operating_carrier": "carrier"})


def rep(task):
    scen, samp, seed = task
    alpha, plant = SCEN[scen]
    rng = np.random.default_rng(seed)
    lam = mu * planted if plant else mu
    if alpha > 0:
        lam = lam * rng.gamma(1 / alpha, alpha, len(lam))
    sim = rng.poisson(lam).astype(float)
    s = d.loc[ROWS[samp]].assign(minutes=sim[ROWS[samp]])
    m = simplex.fit(s, TERMS[samp])
    t = simplex.carrier_terms(m)[["carrier", "cause", "Estimate", "Std. Error"]]
    return t.assign(scenario=scen, sample=samp, sep=len(m.na_index), converged=m.convergence,
                    zero_share=(sim[kept] == 0).mean())


tasks = [(sc, "all") for sc in SCEN] + [("null_nb", sm) for sm in ["few", "many"]]
tasks = [t for t in tasks for _ in range(R)]
seeds = np.random.SeedSequence(0).spawn(len(tasks))
tasks = [(a, b, s) for (a, b), s in zip(tasks, seeds)]

# 配適本身是單執行緒，只能以多行程平行。polars 的執行緒池在 fork 之後會讓子行程死鎖，
# 故由本腳本以參數重新啟動自己，各跑一份任務。一次配適的尖峰約 3 至 4 GB，
# 行程數由記憶體決定而非核心數。
W = 4
if len(sys.argv) > 1:
    k = int(sys.argv[1])
    pd.concat([rep(t) for t in tasks[k::W]]).to_pickle(f"/tmp/poisson_fe_bias_{k}.pkl")
    sys.exit()
for q in [subprocess.Popen([sys.executable, sys.argv[0], str(k)]) for k in range(W)]:
    q.wait()
draws = pd.concat([pd.read_pickle(f"/tmp/poisson_fe_bias_{k}.pkl") for k in range(W)])

draws["true"] = np.where(
    draws["scenario"].str.startswith("planted") & draws["carrier"].isin(PLANT) & (draws["cause"] == "weather"),
    THETA, 0.0)
draws["cover"] = (draws["Estimate"] - draws["true"]).abs() <= 1.96 * draws["Std. Error"]
g = draws.groupby(["scenario", "sample", "carrier", "cause"])
out = g.agg(true=("true", "first"), reps=("Estimate", "size"), mean_est=("Estimate", "mean"),
            sd_est=("Estimate", "std"), mean_se=("Std. Error", "mean"), coverage=("cover", "mean"),
            sep_mean=("sep", "mean"), sep_min=("sep", "min"), sep_max=("sep", "max"),
            converged=("converged", "mean"), zero_share=("zero_share", "mean")).reset_index()
out["bias"] = out["mean_est"] - out["true"]
out["mcse"] = out["sd_est"] / np.sqrt(out["reps"])
out["bias_over_se"] = out["bias"] / out["mean_se"]
out["se_ratio"] = out["mean_se"] / out["sd_est"]
out["alpha"] = out["scenario"].map({k: v[0] for k, v in SCEN.items()})
out["phi_real"] = phi
out["zero_share_real"] = zero
out = out.merge(units, on=["sample", "carrier"])
out.to_csv("outputs/tables/poisson_fe_bias.csv", index=False)

out["z_mc"] = out["bias"] / out["mcse"]
print(out.groupby(["sample", "scenario"]).agg(
    n_coef=("bias", "size"), reps=("reps", "min"), bias_mean=("bias", "mean"),
    bias_absmax=("bias", lambda s: s.abs().max()),
    bias_over_se_mean=("bias_over_se", "mean"), bias_over_se_absmed=("bias_over_se", lambda s: s.abs().median()),
    bias_over_se_absmax=("bias_over_se", lambda s: s.abs().max()),
    n_mc_signif=("z_mc", lambda s: (s.abs() > 1.96).sum()),
    z_mc_rms=("z_mc", lambda s: np.sqrt((s ** 2).mean())),
    se_ratio_med=("se_ratio", "median"), se_ratio_min=("se_ratio", "min"),
    coverage_mean=("coverage", "mean"), coverage_min=("coverage", "min"),
    sep_mean=("sep_mean", "mean"), sep_min=("sep_min", "min"), sep_max=("sep_max", "max"),
    converged=("converged", "min"), zero_share=("zero_share", "mean")).T.to_string())
print(out[out["true"] != 0].to_string())
print(out.sort_values("bias_over_se", key=abs, ascending=False).head(25).to_string())
