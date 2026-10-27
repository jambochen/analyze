# likelihood.fit 的時間分在哪：jit 編譯、minimize 對 f、g、h 的呼叫次數與每次耗時、收斂後的分數與 meat。
# 以 continuation_selection 最大的一格（Blitz 一局一批、lag 1）的續下 logistic 迴歸為例。

import os
import sys
import time

os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, "scripts")

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
from scipy import optimize

import likelihood as lk
from arena_batches import continuation_sample

XCOLS = ["const", "e_bar", "elo_c", "elo_c2", "log_games_in_tour", "log_remaining", "novice_rd", "white", "h", "win", "draw", "h_win", "h_draw"]
s = continuation_sample([1]).filter(pl.col("speed") == "Blitz", pl.col("layer") == "single", pl.col("cont1").is_not_null()).with_columns(const=pl.lit(1.0))
args = (s.select(XCOLS).to_numpy().astype(np.float64), s["cont1"].cast(pl.Float64).to_numpy())
w, b = s["player"].to_numpy(), s["opponent"].to_numpy()
print("rows", s.height, "chunks", -(-s.height // lk.CHUNK))


def ll(th, X, y):
    eta = X @ th
    return y * jax.nn.log_sigmoid(eta) + (1 - y) * jax.nn.log_sigmoid(-eta)


t0 = jnp.zeros(len(XCOLS))
free = np.arange(len(XCOLS))
full = lambda x: t0.at[free].set(x)
nll = lambda x, *a: -jnp.sum(ll(full(x), *a))
cs = [tuple(a[k : k + lk.CHUNK] for a in args) for k in range(0, len(w), lk.CHUNK)]
f_, g_, h_ = jax.jit(nll), jax.jit(jax.grad(nll)), jax.jit(jax.hessian(nll))
sc = jax.jit(jax.jacfwd(lambda x, *a: ll(full(x), *a)))
x0 = np.zeros(len(free))

# 第一次呼叫含編譯（最後一塊較短，形狀不同，另編一次）；第二次只有執行
for nm, fn in [("f", f_), ("g", g_), ("h", h_), ("sc", sc)]:
    t = time.perf_counter()
    [jax.block_until_ready(fn(x0, *c)) for c in cs]
    t1 = time.perf_counter() - t
    t = time.perf_counter()
    [jax.block_until_ready(fn(x0, *c)) for c in cs]
    print(nm, "first", round(t1, 3), "second", round(time.perf_counter() - t, 3))

calls = {"f": [0, 0.0], "g": [0, 0.0], "h": [0, 0.0]}


def timed(nm, fn, conv):
    def inner(x):
        t = time.perf_counter()
        out = conv(sum(fn(x, *c) for c in cs))
        calls[nm][0] += 1
        calls[nm][1] += time.perf_counter() - t
        return out
    return inner


t = time.perf_counter()
r = optimize.minimize(timed("f", f_, float), x0, jac=timed("g", g_, np.asarray), hess=timed("h", h_, np.asarray), method="trust-exact")
print("minimize", round(time.perf_counter() - t, 3), "nit", r.nit, {k: (v[0], round(v[1], 3)) for k, v in calls.items()})
t = time.perf_counter()
H = np.asarray(sum(h_(r.x, *c) for c in cs))
S = np.concatenate([np.asarray(sc(r.x, *c)) for c in cs])
print("H + S", round(time.perf_counter() - t, 3))
t = time.perf_counter()
lk.dyad_meat(S, w, b)
print("dyad_meat", round(time.perf_counter() - t, 3))
t = time.perf_counter()
lk.fit(ll, t0, free, args, w, b)
print("lk.fit total", round(time.perf_counter() - t, 3))
