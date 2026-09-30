import sys

import numpy as np
import pandas as pd
import pyfixest as pf

sys.path.insert(0, "scripts")
import simplex
import stratum

d, terms = simplex.data(stratum.load(), controls=[])
f = f"minutes ~ {' + '.join(terms)} | unit + cellcause"

m = pf.fepois(f, data=d, vcov={"CRV1": "unit"})
used = d.loc[m._data.index].copy()
used["mu"] = m.predict()

# 全樣本 Pearson phi 在此設定下不可解讀：每單元僅四列，參數數與觀測數同量級，
# 殘差自由度是虛的；且分鐘數上看五萬，Pearson 殘差隨計數量級放大。
# 改以配適值分層看 var/mean，該量與計數量級無關。
q = pd.qcut(used["mu"], 10, duplicates="drop")
g = used.groupby(q, observed=True).apply(
    lambda t: pd.Series({"n": len(t), "mu_med": t["mu"].median(),
                         "mean": t["minutes"].mean(), "var": t["minutes"].var()}),
    include_groups=False,
)
g["var_over_mean"] = g["var"] / g["mean"]
print(g.to_string())
g.to_csv("outputs/tables/poisson_dispersion.csv")

# 離散是否已被 cluster SE 吸收：比較三種 vcov。
ses = {}
for tag, v in [("iid", "iid"), ("hetero", "hetero"), ("cluster", {"CRV1": "unit"})]:
    ses[tag] = pf.fepois(f, data=d, vcov=v).tidy()["Std. Error"]
cmp = pd.DataFrame(ses)
cmp["coef"] = m.tidy()["Estimate"]
print("\nSE 比值 cluster/iid 分位", np.percentile(cmp["cluster"] / cmp["iid"], [5, 50, 95]).round(2))
for tag in ["iid", "hetero", "cluster"]:
    print(tag, "顯著數", int(((cmp["coef"] / cmp[tag]).abs() > 1.96).sum()), "/", len(cmp))
print(cmp[cmp.index.str.contains("_OO_")].to_string())

# var/mean 隨均值上升，真實變異接近二次式而非 Poisson 的線性式。
# 極端單元是否撐起結果：winsorize 至 99 百分位重估。
d2 = d.copy()
d2["minutes"] = d2["minutes"].clip(upper=d["minutes"].quantile(0.99))
w = pf.fepois(f, data=d2, vcov={"CRV1": "unit"}).tidy()
c = pd.DataFrame({"base": m.tidy()["Estimate"], "base_se": m.tidy()["Std. Error"],
                  "wins": w["Estimate"], "wins_se": w["Std. Error"]})
c["shift_in_se"] = (c["wins"] - c["base"]).abs() / c["base_se"]
print("\nwinsorize 位移/SE 分位", np.percentile(c["shift_in_se"].dropna(), [50, 90, 100]).round(2))
print(c[c.index.str.contains("_OO_|_MQ_")].to_string())
for s in ["weather", "nas", "late_ac"]:
    k = c[c.index.str.endswith("_" + s)].dropna()
    print(s, "Spearman", round(k["base"].corr(k["wins"], method="spearman"), 3))
c.to_csv("outputs/tables/poisson_winsorize.csv")
