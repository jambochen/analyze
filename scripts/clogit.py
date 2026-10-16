# 以批次分層的 conditional logit。statsmodels 的 ConditionalLogit 逐組以 Python 迴圈計算，
# 數萬個批次時慢到不可用；這裡把同大小的組補齊成陣列，以 jax 一次算完，梯度與 Hessian 由自動微分取得。
# 條件概似的分母是「組內 m 個事件的所有組合」之和，即 exp(eta) 的 m 次基本對稱多項式，以 log 空間的遞推計算。

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
# XLA 的 CPU 執行緒數不受 OMP 等環境變數控制，以 affinity 限制在 8 核，與其他並行工作共用機器
os.sched_setaffinity(0, range(8))

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)
NEG = -1e10


def _bucket_ll(beta, X, y, mask, m, mmax):
    eta = jnp.where(mask, X @ beta, 0.0)
    G = X.shape[0]
    init = jnp.full((G, mmax + 1), NEG).at[:, 0].set(0.0)

    def step(le, j):
        shifted = jnp.concatenate([jnp.full((G, 1), NEG), le[:, :-1]], axis=1) + eta[:, j][:, None]
        return jnp.where(mask[:, j][:, None], jnp.logaddexp(le, shifted), le), None

    le, _ = jax.lax.scan(step, init, jnp.arange(X.shape[1]))
    return (y * eta).sum(1) - jnp.take_along_axis(le, m[:, None], 1)[:, 0]


def fit(y, X, groups, clusters, names):
    """回傳 dict：coef、cov（以 clusters 聚類的 sandwich）、cov_naive、names、groups_used、n_used。
    事件全為 0 或全為 1 的組對條件概似沒有貢獻，先剔除。"""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    _, gi = np.unique(groups, return_inverse=True)
    n = np.bincount(gi)
    m = np.bincount(gi, y).round().astype(int)
    keep_g = (m > 0) & (m < n)
    keep = keep_g[gi]
    y, X, gi, cl = y[keep], X[keep], gi[keep], np.asarray(clusters)[keep]
    _, gi = np.unique(gi, return_inverse=True)
    n = np.bincount(gi)
    m = np.bincount(gi, y).round().astype(int)
    order = np.argsort(gi, kind="stable")
    y, X, gi, cl = y[order], X[order], gi[order], cl[order]
    pos = np.arange(len(gi)) - np.r_[0, np.cumsum(n)[:-1]][gi]
    gcl = cl[np.r_[0, np.cumsum(n)[:-1]]]
    p = X.shape[1]
    buckets = []
    width = 2 ** np.ceil(np.log2(np.maximum(n, 2))).astype(int)
    for w in np.unique(width):
        gs = np.flatnonzero(width == w)
        loc = np.full(len(n), -1)
        loc[gs] = np.arange(len(gs))
        rows = loc[gi] >= 0
        Xb = np.zeros((len(gs), w, p))
        yb = np.zeros((len(gs), w))
        mb = np.zeros((len(gs), w), bool)
        Xb[loc[gi[rows]], pos[rows]] = X[rows]
        yb[loc[gi[rows]], pos[rows]] = y[rows]
        mb[loc[gi[rows]], pos[rows]] = True
        mm = m[gs]
        buckets.append((jnp.array(Xb), jnp.array(yb), jnp.array(mb), jnp.array(mm), int(mm.max()), gs))

    def ll(beta):
        return sum(_bucket_ll(beta, Xb, yb, mb, mm, mx).sum() for Xb, yb, mb, mm, mx, _ in buckets)

    grad = jax.jit(jax.grad(ll))
    hess = jax.jit(jax.hessian(ll))
    beta = jnp.zeros(p)
    for _ in range(50):
        H = hess(beta)
        step = jnp.linalg.solve(H, grad(beta))
        beta = beta - step
        if float(jnp.abs(step).max()) < 1e-9:
            break
    H = np.asarray(hess(beta))
    scores = np.zeros((len(n), p))
    for Xb, yb, mb, mm, mx, gs in buckets:
        s = jax.jacfwd(lambda b: _bucket_ll(b, Xb, yb, mb, mm, mx))(beta)
        scores[gs] = np.asarray(s)
    _, ci = np.unique(gcl, return_inverse=True)
    S = np.zeros((ci.max() + 1, p))
    np.add.at(S, ci, scores)
    C = S.shape[0]
    Hinv = np.linalg.inv(-H)
    cov = Hinv @ (S.T @ S) @ Hinv * C / (C - 1)
    return dict(coef=np.asarray(beta), cov=cov, cov_naive=Hinv, names=list(names), groups_used=len(n), n_used=len(y))
