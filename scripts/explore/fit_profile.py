# calibration.py 與 threshold_rating_form.py 的耗時分段：前處理、jit 編譯、值/梯度/Hessian、
# 逐局分數、dyadic meat、其餘（trust-exact 與 Python）。每段同時記 CPU 時間，CPU / 牆鐘即平均用到的核心數。
# lk.fit 的內部以包裝 _compiled 與 dyad_meat 量測，不改 likelihood.py。

import os
import sys
import time

os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, "scripts")

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl

import game_source
import likelihood as lk
import novice
import ordinal

UTC = "UTC"
T = {}


def tick(name, t0, c0):
    w, c = time.perf_counter() - t0, sum(os.times()[:2]) - c0
    a = T.setdefault(name, [0.0, 0.0, 0])
    a[0] += w
    a[1] += c
    a[2] += 1


def clock():
    return time.perf_counter(), sum(os.times()[:2])


orig_compiled = lk._compiled
seen = set()


def timed_compiled(ll):
    vgh_, score_ = orig_compiled(ll)

    def vgh(*a):
        # 每種形狀第一次呼叫含編譯
        key = (ll, tuple(np.shape(x) for x in a[4:]), len(a[0]))
        t0, c0 = clock()
        out = jax.block_until_ready(vgh_(*a))
        tick("compile+first_vgh" if key not in seen else "vgh", t0, c0)
        seen.add(key)
        return out

    def score(*a):
        key = ("s", ll, tuple(np.shape(x) for x in a[3:]), len(a[0]))
        t0, c0 = clock()
        out = jax.block_until_ready(score_(*a))
        tick("compile+first_score" if key not in seen else "score", t0, c0)
        seen.add(key)
        return out

    return vgh, score


lk._compiled = timed_compiled
orig_meat = lk.dyad_meat


def timed_meat(S, w, b):
    t0, c0 = clock()
    out = orig_meat(S, w, b)
    tick("dyad_meat", t0, c0)
    return out


lk.dyad_meat = timed_meat


def fit(ll, t0, free, args, w, b):
    t0_, c0 = clock()
    out = lk.fit(ll, t0, free, args, w, b)
    tick("fit_total", t0_, c0)
    return out


t0, c0 = clock()
g = (
    pl.scan_parquet("data/processed/lichess_games.parquet")
    .filter(pl.col("speed") == "Bullet", pl.col("base_seconds").is_not_null(), pl.col("start") < pl.datetime(2015, 1, 1, time_zone=UTC))
    .select(
        "game_id", "start", "white_elo", "black_elo",
        arena=pl.col("tournament_id").is_not_null(),
        y=pl.col("result").replace_strict({"0-1": 0, "1/2-1/2": 1, "1-0": 2}, default=None, return_dtype=pl.Int8),
        w="white_id", b="black_id",
    )
    .with_row_index("i")
    .collect(engine="streaming")
)
nprev = (
    pl.concat([g.select("i", "start", p="w"), g.select("i", "start", p="b")])
    .sort("i")
    .with_columns(n_prev=pl.int_range(pl.len(), dtype=pl.UInt32).over("p"))
    .select("i", "p", "n_prev")
)
cal = (
    game_source.with_source(g.lazy())
    .filter(pl.col("start") >= pl.datetime(2014, 1, 1, time_zone=UTC), pl.col("y").is_not_null(), pl.col("white_elo").is_not_null(), pl.col("black_elo").is_not_null(), game_source.KEEP)
    .join(nprev.lazy().rename({"p": "w", "n_prev": "nw"}), on=["i", "w"])
    .join(nprev.lazy().rename({"p": "b", "n_prev": "nb"}), on=["i", "b"])
    .filter(pl.min_horizontal("nw", "nb") >= novice.GAMES)
    .sort("i")
    .with_columns(xr=((pl.col("white_elo") + pl.col("black_elo")) / 2 - 1500) / 100, xd=(pl.col("white_elo") - pl.col("black_elo")).cast(pl.Float64) / 100)
    .select("w", "b", "y", "xr", "xd")
    .collect(engine="streaming")
)
del g, nprev
tick("prep_polars", t0, c0)
print("Bullet 老手對局", cal.height, flush=True)


def ll_ord(t, xr, xd, y):
    eta = t[0] + t[1] * xr + t[2] * xr**2 + (t[3] + t[4] * xr) * xd
    return ordinal.ll(eta, jnp.exp(t[5] + t[6] * xr + t[7] * xr**2), y)


T0 = [0.1, 0, 0, 0.6, 0, -2.5, 0, 0]
IDX = np.arange(8)
rows = [{"sample": "prep", "n": cal.height, "phase": "prep_polars", "wall_s": T["prep_polars"][0], "cpu_s": T["prep_polars"][1], "calls": 1, "cores": T["prep_polars"][1] / T["prep_polars"][0]}]
for label, n in [("1M", 1_000_000), ("full", cal.height)]:
    T.clear()
    t0, c0 = clock()
    m = cal.sample(n, seed=1) if n < cal.height else cal
    w, b = m["w"].to_numpy(), m["b"].to_numpy()
    xr, xd, y = (m[c].to_numpy().astype(np.float64) for c in ["xr", "xd", "y"])
    tick("sample", t0, c0)
    # calibration.py 的一個速度類別：完整模型與五個受限模型
    fit(ll_ord, T0, IDX, (xr, xd, y), w, b)
    for idx, val in [([1, 2], [0, 0]), ([6, 7], [0, 0]), ([4], [0]), ([1, 2, 4, 6, 7], [0] * 5), ([3], [100 * ordinal.Q])]:
        t_ = np.array(T0, dtype=float)
        t_[idx] = val
        fit(ll_ord, t_, np.setdiff1d(IDX, idx), (xr, xd, y), w, b)
    # threshold_rating_form.py 的分組模型：43 個參數
    K = 20
    edges = np.quantile(xr, np.linspace(0, 1, K + 1)[1:-1])
    Z = np.column_stack([np.ones_like(xr), xr, xr**2, np.eye(K)[np.searchsorted(edges, xr, side="right")][:, 1:]])
    gxd = xd * 0.9

    def ll_thr(t, Z, xr, xd, gxd, y):
        p = Z.shape[1]
        eta = Z @ t[3 : 3 + p] + (t[0] + t[1] * xr) * xd + t[2] * gxd
        return ordinal.ll(eta, jnp.exp(Z @ t[3 + p :]), y)

    p = Z.shape[1]
    t_ = np.r_[0.6, 0, 0, 0.1, np.zeros(p - 1), -2.5, np.zeros(p - 1)]
    fit(ll_thr, t_, np.arange(len(t_)), (Z, xr, xd, gxd, y), w, b)
    for k, (wall, cpu, cnt) in T.items():
        rows.append({"sample": label, "n": n, "phase": k, "wall_s": wall, "cpu_s": cpu, "calls": cnt, "cores": cpu / wall if wall else None})
    print(label, flush=True)

out = pl.DataFrame(rows)
pl.Config.set_tbl_rows(40)
print(out)
out.write_csv("outputs/tables/fit_profile.csv")
