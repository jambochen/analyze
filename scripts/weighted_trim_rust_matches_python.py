# rust/weighted_trim（Python 模組名 weighted_trim_rs）與 numpy 參考實作是否一致。
# 參考實作即 continuation_selection.py 改用 Rust 之前的寫法：修剪平均展開「重抽 × 列」的權重矩陣再 cumsum；
# 傾斜平均逐一 gamma 以 scipy 的 brentq 求根。
# numpy 的 sum 為 pairwise、Rust 為逐項累加，兩者只差浮點誤差，以相對誤差比較，不要求逐位元一致。
# 資料為模擬：大小取 continuation_selection 一格的量級，玩家局數為重尾，含極小與等於 1 的修剪比例。

import time

import numpy as np
import weighted_trim_rs
from scipy import optimize


def trim_low(y, W, share):
    tot = W.sum(1)
    target = share * tot
    cw = np.cumsum(W, 1)
    take = np.clip(target[:, None] - (cw - W), 0, W)
    return (take * y).sum(1) / target


def trim_high(y, W, share):
    return trim_low(y[::-1], W[:, ::-1], share)


def tilt_mean(y, w, share, gamma):
    if gamma == 0 or share == 1:
        return (w * y).sum() / w.sum()
    f = lambda a: (w / (1 + np.exp(-(a + gamma * y)))).sum() / w.sum() - share
    a = optimize.brentq(f, -60 - abs(gamma), 60 + abs(gamma))
    pi = w / (1 + np.exp(-(a + gamma * y)))
    return (pi * y).sum() / pi.sum()


rng = np.random.default_rng(0)
B = 999
for P, n in [(500, 2_000), (20_000, 30_000), (60_000, 150_000)]:
    C = rng.multinomial(P, np.full(P, 1 / P), size=B).astype(np.float64)
    act = rng.pareto(1.2, P) + 1
    pi = np.sort(rng.choice(P, n, p=act / act.sum()))
    y = rng.normal(size=n)
    wr = 1 / np.bincount(pi, minlength=P)[pi]
    o = np.argsort(y)
    ys, ps, ws = y[o], pi[o], wr[o]
    shares = np.column_stack([rng.uniform(0, 1, B + 1), np.r_[1e-9, rng.uniform(0, 1, B - 1), 1.0]])

    t0 = time.perf_counter()
    W = np.vstack([ws[None, :], C[:, ps] * ws])
    ref = ((W * ys).sum(1) / W.sum(1), np.column_stack([trim_low(ys, W, s) for s in shares.T]), np.column_stack([trim_high(ys, W, s) for s in shares.T]))
    t_py = time.perf_counter() - t0
    t0 = time.perf_counter()
    rs = weighted_trim_rs.trim_means(ys, ps, ws, C, shares)
    t_rs = time.perf_counter() - t0
    for name, a, b in zip(["mean", "low", "high"], ref, rs):
        rel = np.abs(a - b) / np.maximum(np.abs(a), 1e-300)
        print(P, n, name, "max rel diff", rel.max())
        assert np.allclose(a, b, rtol=1e-10, atol=0)
    print(P, n, "numpy", t_py, "rust", t_rs)

    # K 為 0 時只算加權平均，y 不必排序（IPW 的用法）
    W = np.vstack([wr[None, :], C[:, pi] * wr])
    m_rs = weighted_trim_rs.trim_means(y, pi, wr, C, np.zeros((B + 1, 0)))[0]
    assert np.allclose((W * y).sum(1) / W.sum(1), m_rs, rtol=1e-10, atol=0)

    # 傾斜平均：gamma 的格點同 continuation_selection；share 含 1（不修剪）與接近 0、1 的值。
    # 反應變數是得分殘差，落在 [-1, 1]；常態的尾端在 |gamma| = 40 時超出 brentq 的初始區間
    gammas = np.round(np.linspace(-40, 40, 81), 1)
    yr = rng.uniform(-1, 1, n)
    for share in [1.0, 0.999, 0.6, 0.05]:
        t0 = time.perf_counter()
        ref = np.array([tilt_mean(yr, wr, share, g) for g in gammas])
        t_py = time.perf_counter() - t0
        t0 = time.perf_counter()
        rs = weighted_trim_rs.tilt_means(yr, wr, share, gammas)
        t_rs = time.perf_counter() - t0
        print(P, n, "tilt share", share, "max abs diff", np.abs(ref - rs).max(), "numpy", t_py, "rust", t_rs)
        assert np.allclose(ref, rs, rtol=0, atol=1e-10)
