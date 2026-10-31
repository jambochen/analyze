# likelihood.py 的 frac_score、frac_score_at、frac_score_ci（先壓成相異 L 值的充分統計量）與逐局計算的參考實作是否一致。
# 輸入為 2014 年 Bullet 的對局：Elo 的 L（相異值少）與連續的 L（Elo 的 L 加上微小擾動，相異值約等於局數）。
# 加總的順序不同，只差浮點誤差；區間端點經 brentq，以相對誤差比較。

import sys
import time

sys.path.insert(0, "scripts")

import numpy as np
import polars as pl
from scipy import optimize, stats

import likelihood as lk


def frac_score_ref(L, s, t):
    X = np.column_stack([np.ones_like(L), L])
    p = 1 / (1 + np.exp(-(t[0] + t[1] * L)))
    U = X.T @ (s - p)
    return U @ np.linalg.solve(X.T @ (X * (p * (1 - p))[:, None]), U)


def frac_score_at_ref(L, s, j, v, t0):
    X = np.column_stack([np.ones_like(L), L])
    t = np.array(t0, dtype=float)
    t[j] = v
    k = 1 - j
    for _ in range(50):
        p = 1 / (1 + np.exp(-(t[0] + t[1] * L)))
        step = (X[:, k] @ (s - p)) / (X[:, k] ** 2 @ (p * (1 - p)))
        t[k] += step
        if abs(step) < 1e-12:
            break
    return frac_score_ref(L, s, t)


def frac_score_ci_ref(L, s, j, th, se):
    f = lambda v: frac_score_at_ref(L, s, j, v, th) - stats.chi2.ppf(0.95, 1)
    return optimize.brentq(f, th[j] - 4 * se, th[j]), optimize.brentq(f, th[j], th[j] + 4 * se)


g = (
    pl.scan_parquet("data/processed/lichess_games.parquet")
    .filter(pl.col("speed") == "Bullet", pl.col("start").dt.year() == 2014, pl.col("white_elo").is_not_null(), pl.col("black_elo").is_not_null())
    .select(
        d=(pl.col("white_elo") - pl.col("black_elo")).cast(pl.Float64),
        s=pl.col("result").replace_strict({"0-1": 0.0, "1/2-1/2": 0.5, "1-0": 1.0}, default=None),
    )
    .drop_nulls()
    .collect()
)
Le = g["d"].to_numpy() * np.log(10) / 400
s = g["s"].to_numpy()
rng = np.random.default_rng(1)
Lc = Le + rng.normal(0, 1e-3, len(Le))

rows = []
for name, L in [("elo", Le), ("continuous", Lc)]:
    th = lk.frac_newton(np.column_stack([np.ones_like(L), L]), s, [0.0, 1.0])[0]
    for j in [0, 1]:
        se = 0.01
        t0 = time.perf_counter()
        ref = (frac_score_ref(L, s, [0.0, 1.0]), frac_score_at_ref(L, s, j, [0.0, 1.0][j], th), *frac_score_ci_ref(L, s, j, th, se))
        t1 = time.perf_counter()
        new = (lk.frac_score(L, s, [0.0, 1.0]), lk.frac_score_at(L, s, j, [0.0, 1.0][j], th), *lk.frac_score_ci(L, s, j, th, se))
        t2 = time.perf_counter()
        for q, a, b in zip(["score", "score_at_h0", "ci_lo", "ci_hi"], ref, new):
            rows.append({"L": name, "j": j, "quantity": q, "ref": a, "new": b, "rel_err": abs(a - b) / abs(a), "ref_s": t1 - t0, "new_s": t2 - t1})

out = pl.DataFrame(rows)
pl.Config.set_tbl_rows(40)
print(len(Le), np.unique(Le).size, np.unique(Lc).size)
print(out)
print("max rel err", out["rel_err"].max())
