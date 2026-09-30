import numpy as np
import pandas as pd

import simplex
import stratum

b = stratum.load()

d, terms = simplex.data(b)
print("單元", d["unit"].nunique(), "層 × 成因", d["cellcause"].nunique())
m = simplex.fit(d, terms)
r = m.tidy().reset_index()
r.columns = ["term"] + list(r.columns[1:])
r.to_csv("outputs/tables/attribution_simplex.csv", index=False)

# 對比的標準誤須取兩係數的共變異數，各自的標準誤不夠。
coefs = m.coef()
V = pd.DataFrame(m._vcov, index=coefs.index, columns=coefs.index)
rows = []
for c in sorted(set(d["operating_carrier"]) - {stratum.REF}):
    for a, bcause in [("weather", "nas"), ("weather", "late_ac"), ("nas", "late_ac")]:
        ta, tb = f"x_{c}_{a}", f"x_{c}_{bcause}"
        est = coefs[ta] - coefs[tb]
        se = np.sqrt(V.loc[ta, ta] + V.loc[tb, tb] - 2 * V.loc[ta, tb])
        rows.append({"carrier": c, "contrast": f"{a}-{bcause}", "estimate": est, "se": se,
                     "lo": est - 1.96 * se, "hi": est + 1.96 * se})
cb = pd.DataFrame(rows)
cb.to_csv("outputs/tables/attribution_boundary.csv", index=False)
print(cb[cb["contrast"] == "weather-nas"].sort_values("estimate", ascending=False).to_string())

# 不控制氣象的版本，用以檢查層 × 成因固定效應之外的氣象控制改變多少。
d0, terms0 = simplex.data(b, controls=[])
base = simplex.carrier_terms(simplex.fit(d0, terms0))
base.to_csv("outputs/tables/simplex_poisson.csv", index=False)

car = simplex.carrier_terms(m)
cmp = car.merge(base[["carrier", "cause", "Estimate"]], on=["carrier", "cause"], suffixes=("", "_base"))
cmp["shift_in_se"] = (cmp["Estimate"] - cmp["Estimate_base"]).abs() / cmp["Std. Error"]
print(cmp[["carrier", "cause", "Estimate", "Estimate_base", "shift_in_se"]].sort_values("shift_in_se", ascending=False).head(12).to_string())
print("\n加氣象後係數最大位移（以標準誤計）", round(cmp["shift_in_se"].max(), 3))
for s in ["weather", "nas", "late_ac"]:
    t = cmp[cmp["cause"] == s].sort_values("Estimate", ascending=False)
    print(f"\n{s} 前五", t["carrier"].head(5).to_list())
    print(f"{s} 與不控制氣象的 Spearman", round(t["Estimate"].corr(t["Estimate_base"], method="spearman"), 4))
