# rust/likelihood（Python 模組名 likelihood_rs）與參考實作是否一致：
# - logit_ll 的 fit：以包住 logit_ll 的 lambda 呼叫 fit 即走 jax 路徑，與 Rust 路徑比參數、對數概似與共變異
# - dyad_meat、依玩家分群的 meat：numpy 的 np.unique 加 bincount
# 加總順序不同，只差浮點誤差；fit 另受 trust-exact 停止容差影響，以相對誤差比較。
# 資料為模擬：玩家局數為重尾，同一對手組合重複對局，玩家代碼稀疏（全庫代碼），分數含一維的情形（dyad_meat 的呼叫端常傳一維）。

import os
import sys
import time

os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, "scripts")

import likelihood_rs
import numpy as np

import likelihood as lk


def dyad_meat_np(S, w, b):
    S = S.reshape(len(w), -1)
    P = int(max(w.max(), b.max())) + 1
    Sp = np.stack([np.bincount(w, S[:, j], P) + np.bincount(b, S[:, j], P) for j in range(S.shape[1])], 1)
    pair = np.unique(np.minimum(w, b).astype(np.int64) * P + np.maximum(w, b), return_inverse=True)[1]
    Sq = np.stack([np.bincount(pair, S[:, j]) for j in range(S.shape[1])], 1)
    return Sp.T @ Sp - Sq.T @ Sq


def player_meat_np(S, w):
    pc = np.unique(w, return_inverse=True)[1]
    Sp = np.stack([np.bincount(pc, S[:, j]) for j in range(S.shape[1])], 1)
    return Sp.T @ Sp


def rel(a, b):
    return np.max(np.abs(a - b) / np.maximum(np.abs(a).max(), 1e-300))


rng = np.random.default_rng(0)
for P, n, k in [(300, 2_000, 1), (20_000, 200_000, 4), (60_000, 800_000, 13)]:
    act = rng.pareto(1.2, P) + 1
    w = rng.choice(P, n, p=act / act.sum()).astype(np.uint32)
    b = rng.choice(P, n, p=act / act.sum()).astype(np.uint32)
    # 一部分對局重複同一對手組合
    rep = rng.random(n) < 0.3
    b[rep] = b[np.flatnonzero(rep) - 1]
    b[b == w] = (w[b == w] + 1) % P
    # 實際的玩家代碼是全庫代碼，遠大於樣本內的玩家數且不連續
    code = rng.choice(2_000_000, P, replace=False).astype(np.uint32)
    w, b = code[w], code[b]
    S = rng.normal(size=(n, k)) if k > 1 else rng.normal(size=n)

    t0 = time.perf_counter()
    ref = dyad_meat_np(S, w.astype(np.int64), b.astype(np.int64))
    t_np = time.perf_counter() - t0
    t0 = time.perf_counter()
    rs = lk.dyad_meat(S, w, b)
    t_rs = time.perf_counter() - t0
    print(P, n, k, "dyad_meat rel diff", rel(ref, rs), "numpy", round(t_np, 3), "rust", round(t_rs, 3))
    assert rel(ref, rs) < 1e-12

    S2 = S.reshape(n, -1)
    ref = player_meat_np(S2, w)
    rs = likelihood_rs.group_outer(np.ascontiguousarray(S2), w.astype(np.int64))
    print(P, n, k, "player meat rel diff", rel(ref, rs))
    assert rel(ref, rs) < 1e-12

# logit_ll 的 fit：Rust 路徑與 jax 路徑。y 為得分（0、0.5、1），含全參數、受限與全固定
n, p = 600_000, 13
X = np.column_stack([np.ones(n), rng.normal(size=(n, p - 1))])
eta = X @ rng.normal(0, 0.3, p)
y = np.where(rng.random(n) < 0.1, 0.5, (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(float))
w = rng.integers(0, 20_000, n)
b = (w + rng.integers(1, 20_000, n)) % 20_000
via_jax = lambda th, X, y: lk.logit_ll(th, X, y)
for name, free in [("full", np.arange(p)), ("restricted", np.arange(8)), ("none", np.array([], int))]:
    t0 = time.perf_counter()
    ref = lk.fit(via_jax, np.zeros(p), free, (X, y), w, b)
    t_jax = time.perf_counter() - t0
    t0 = time.perf_counter()
    rs = lk.fit(lk.logit_ll, np.zeros(p), free, (X, y), w, b)
    t_rs = time.perf_counter() - t0
    d = [rel(np.asarray(a), np.asarray(c)) if np.size(a) else 0.0 for a, c in zip(ref, rs)]
    print("logit fit", name, "rel diff [theta, ll, V, Hi, S, H]", ["%.1e" % v for v in d], "jax", round(t_jax, 3), "rust", round(t_rs, 3))
    assert max(d) < 1e-8
