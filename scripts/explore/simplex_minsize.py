import sys

import pandas as pd

sys.path.insert(0, "scripts")
import simplex
import stratum

b = stratum.load()

# separation 移除的全是零，且集中在 weather 欄與極小的單元。只丟零是不對稱的，
# 會讓表現正常的層被選擇性剔除。改為整批排除小單元，檢查係數是否隨之改變。
rows = []
for minsize in [1, 3, 5, 10]:
    d, terms = simplex.data(b, minsize=minsize)
    m = simplex.fit(d, terms)
    t = simplex.carrier_terms(m)
    t["minsize"] = minsize
    t["n_sep"] = getattr(m, "n_separation_na", 0)
    rows.append(t)
    oo = t[t["carrier"] == "OO"].set_index("cause")["Estimate"]
    print(minsize, "單元", d["unit"].nunique(), "separation", getattr(m, "n_separation_na", 0),
          "OO weather", round(oo["weather"], 4), "nas", round(oo["nas"], 4), "late_ac", round(oo["late_ac"], 4))

a = pd.concat(rows)
a.to_csv("outputs/tables/simplex_minsize.csv", index=False)

base = a[a["minsize"] == 1].set_index(["carrier", "cause"])["Estimate"]
for ms in [3, 5, 10]:
    cur = a[a["minsize"] == ms].set_index(["carrier", "cause"])["Estimate"]
    se = a[a["minsize"] == ms].set_index(["carrier", "cause"])["Std. Error"]
    print(f"\nminsize={ms} 相對無限制的最大位移（標準誤計）", round(((cur - base).abs() / se).max(), 3))
    for s in ["weather", "nas", "late_ac"]:
        x = a[(a["minsize"] == ms) & (a["cause"] == s)].set_index("carrier")["Estimate"]
        y = a[(a["minsize"] == 1) & (a["cause"] == s)].set_index("carrier")["Estimate"]
        print(f"  {s} Spearman", round(x.corr(y, method="spearman"), 4))
