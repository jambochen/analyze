# 計數式的 Markov 轉移能否取代依狀態調整的勝率的有序 logistic（docs/methods.md「依狀態調整的勝率」）。
# 樣本、上半年估計、下半年評估與 dyadic 標準誤皆同 scripts/state_adjusted_probs.py，只換模型。
# 效果相近就用簡單的：轉移矩陣只是分格計數，不需最大概似。

import os
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, "scripts")

import numpy as np
import polars as pl

import likelihood as lk
import ordinal
from arena_batches import player_sample_rd

UTC = "UTC"
SPEEDS = ["Bullet", "Blitz", "Classical"]

grp = ["player", "tournament_id"]
t = (
    player_sample_rd()
    .sort("player", "tournament_id", "start", "game_id")
    .with_columns(prev_result=pl.col("result").shift(1).over(grp))
    .filter(pl.col("prev_result").is_not_null())
    .with_columns(
        y=(pl.col("score") * 2).cast(pl.Int64),
        prev=pl.col("prev_result").replace_strict({"loss": 0, "draw": 1, "win": 2}, return_dtype=pl.Int64),
        prev_draw=(pl.col("prev_result") == "draw").cast(pl.Float64),
        prev_win=(pl.col("prev_result") == "win").cast(pl.Float64),
        train=pl.col("start") < pl.datetime(2014, 7, 1, time_zone=UTC),
    )
)


def ll(t, eta, w, X, y):
    return ordinal.ll(eta + X @ t, w, y)


# 分格計數的條件機率；加 0.5 避免下半年遇到上半年沒出現過的格子時機率為 0
def cell_probs(tr_cell, tr_y, te_cell, ncell):
    C = np.zeros((ncell, 3)) + 0.5
    np.add.at(C, (tr_cell, tr_y), 1)
    return (C / C.sum(1, keepdims=True))[te_cell]


def losses(p, y):
    O = np.eye(3)[y]
    C, Co = np.cumsum(p, 1)[:, :2], np.cumsum(O, 1)[:, :2]
    return {"log_loss": -np.log(p[np.arange(len(y)), y]), "rps": ((C - Co) ** 2).sum(1) / 2}


rows, mats = [], []
for spd in SPEEDS:
    ts = t.filter(pl.col("speed") == spd)
    tr, te = ts.filter(pl.col("train")), ts.filter(~pl.col("train"))
    ytr, yte = tr["y"].to_numpy(), te["y"].to_numpy()
    ptr, pte = tr["prev"].to_numpy(), te["prev"].to_numpy()
    pb_tr = ordinal.probs(tr["eta"].to_numpy(), tr["w"].to_numpy())
    pb_te = ordinal.probs(te["eta"].to_numpy(), te["w"].to_numpy())
    # 十等分的切點取自上半年
    cuts = np.quantile(tr["e_base"].to_numpy(), np.linspace(0, 1, 11)[1:-1])
    btr, bte = np.searchsorted(cuts, tr["e_base"].to_numpy()), np.searchsorted(cuts, te["e_base"].to_numpy())

    P = {"baseline": pb_te}
    P["markov"] = cell_probs(ptr, ytr, pte, 3)
    P["ebin"] = cell_probs(btr, ytr, bte, 10)
    P["markov_ebin"] = cell_probs(btr * 3 + ptr, ytr, bte * 3 + pte, 30)
    # 觀察／期望的 3×3 矩陣：列為上一局，欄為本局，期望為主基準機率的加總
    obs, exp = np.zeros((3, 3)), np.zeros((3, 3))
    np.add.at(obs, (ptr, ytr), 1)
    for k in range(3):
        exp[k] = pb_tr[ptr == k].sum(0)
    R = obs / exp
    q = pb_te * R[pte]
    P["markov_ratio"] = q / q.sum(1, keepdims=True)
    for k, r in enumerate(["loss", "draw", "win"]):
        mats.append({"speed": spd, "prev": r, **{f"obs_{c}": obs[k, j] for j, c in enumerate(["loss", "draw", "win"])}, **{f"exp_{c}": exp[k, j] for j, c in enumerate(["loss", "draw", "win"])}})

    X = lambda d: d.select("prev_draw", "prev_win").to_numpy()
    th = lk.fit(ll, np.zeros(2), np.array([0, 1]), (tr["eta"].to_numpy(), tr["w"].to_numpy(), X(tr), ytr), tr["player"].to_numpy(), tr["opponent"].to_numpy())[0]
    P["ordinal_prev"] = ordinal.probs(te["eta"].to_numpy() + X(te) @ th, te["w"].to_numpy())

    L = {k: losses(v, yte) for k, v in P.items()}
    wte, bte_ = te["player"].to_numpy(), te["opponent"].to_numpy()
    for vs in ["baseline", "ordinal_prev"]:
        for nm in P:
            for metric in ["log_loss", "rps"]:
                dlt = L[nm][metric] - L[vs][metric]
                se = np.sqrt(lk.dyad_meat((dlt - dlt.mean())[:, None], wte, bte_)[0, 0]) / len(dlt)
                rows.append({"speed": spd, "model": nm, "vs": vs, "metric": metric, "n": len(dlt), "mean": L[nm][metric].mean(), "diff": dlt.mean(), "diff_se_dyadic": se})
    print(spd, "done", flush=True)

res = pl.DataFrame(rows)
mat = pl.DataFrame(mats)
with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250):
    print(res.filter(pl.col("metric") == "log_loss"))
    print(mat)
res.write_csv("outputs/tables/markov_transition_prediction.csv")
mat.write_csv("outputs/tables/markov_transition_matrix.csv")
