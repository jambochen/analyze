# GAM 的發現放回專案自己的有序 logistic（和局帶寬隨等級分調整）檢驗：近況線性模型 P1 之上，
# 加 RD 的樣條（P2）、加 dr 與其差的樣條（P3）、兩者都加（P4）。樣條為 patsy 的自然三次樣條（df = 5），節點取自上半年。
# 切分與樣本同 state_adjusted_probs.py（上半年估計、下半年評估）；輸入來自 gam_glmm_export.py 的匯出檔。
import os
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, "scripts")

import jax.numpy as jnp
import numpy as np
import pandas as pd
import patsy
import polars as pl

import likelihood as lk
import ordinal

LIN = ["prev_draw", "prev_win", "dr", "d_prev_draw", "d_prev_win", "d_dr", "opp_missing"]
SPL = {"rd": ["rd", "opp_rd"], "state": ["dr", "d_dr"]}


def ll(t, eta, w, X, y):
    return ordinal.ll(eta + t[0] + X @ t[2:], w * jnp.exp(t[1]), y)


def losses(p, y):
    O = np.eye(3)[y]
    return -np.log(p[np.arange(len(y)), y]), ((np.cumsum(p, 1)[:, :2] - np.cumsum(O, 1)[:, :2]) ** 2).sum(1) / 2


rows = []
for sp in ["blitz", "bullet"]:
    d = pd.read_csv(f"data/processed/explore_gam_glmm_{sp}.csv")
    d["y"] = d["y"] - 1
    tr, te = d[d.train == 1].reset_index(drop=True), d[d.train == 0].reset_index(drop=True)
    designs = {}
    for key, cols in SPL.items():
        dm = {c: patsy.dmatrix(f"cr({c}, df=5)", tr, return_type="dataframe") for c in cols}
        designs[key] = (
            np.hstack([dm[c].iloc[:, 1:].to_numpy() for c in cols]),
            np.hstack([patsy.build_design_matrices([dm[c].design_info], te, return_type="dataframe")[0].iloc[:, 1:].to_numpy() for c in cols]),
        )
    lin_tr, lin_te = tr[LIN].to_numpy(), te[LIN].to_numpy()
    loss = {}
    for nm, keys in [("P1", []), ("P2", ["rd"]), ("P3", ["state"]), ("P4", ["rd", "state"])]:
        Xtr = np.hstack([lin_tr] + [designs[k][0] for k in keys])
        Xte = np.hstack([lin_te] + [designs[k][1] for k in keys])
        T0 = np.zeros(2 + Xtr.shape[1])
        th = lk.fit(ll, T0, np.arange(len(T0)), (tr.eta.to_numpy(), tr.w.to_numpy(), Xtr, tr.y.to_numpy()), tr.player.to_numpy(), tr.opponent.to_numpy())[0]
        p = ordinal.probs(te.eta.to_numpy() + th[0] + Xte @ th[2:], te.w.to_numpy() * np.exp(th[1]))
        loss[nm] = losses(p, te.y.to_numpy())
        print(sp, nm, flush=True)
    seen = te.player.isin(tr.player.unique()).to_numpy()
    for grp, msk in [("all", np.ones(len(te), bool)), ("seen", seen), ("unseen", ~seen)]:
        for nm in ["P2", "P3", "P4"]:
            for k, metric in enumerate(["log_loss", "rps"]):
                dlt = (loss[nm][k] - loss["P1"][k])[msk]
                se = lk.dyad_se((dlt - dlt.mean()) / msk.sum(), te.player.to_numpy()[msk], te.opponent.to_numpy()[msk])
                rows.append({"speed": sp, "group": grp, "model": nm, "vs": "P1", "metric": metric, "n": int(msk.sum()), "diff": dlt.mean(), "se": se, "z": dlt.mean() / se})
res = pl.DataFrame(rows)
with pl.Config(tbl_rows=-1, tbl_width_chars=160):
    print(res.filter(pl.col("metric") == "log_loss"))
res.write_csv("outputs/tables/explore_gam_in_project_family.csv")
