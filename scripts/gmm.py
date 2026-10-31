# 完整共變異的 Gaussian mixture，以 EM 估計。
import numpy as np
from scipy.special import logsumexp


def logp(Z, pi, mu, cov):
    d = Z.shape[1]
    D = Z[None] - mu[:, None]
    q = ((D @ np.linalg.inv(cov)) * D).sum(2)
    return (np.log(pi)[:, None] - 0.5 * (q + np.linalg.slogdet(cov)[1][:, None] + d * np.log(2 * np.pi))).T


def predict(Z, f):
    return np.argmax(logp(Z, f["pi"], f["mu"], f["cov"]), 1)


def gmm(Z, K, rng, starts, max_iter=1000):
    best = None
    d = Z.shape[1]
    for _ in range(starts):
        mu = Z[rng.choice(len(Z), K, replace=False)]
        cov = np.stack([np.cov(Z.T)] * K)
        pi = np.full(K, 1 / K)
        ll_old = -np.inf
        for it in range(max_iter):
            lp = logp(Z, pi, mu, cov)
            lse = logsumexp(lp, 1, keepdims=True)
            ll = lse.sum()
            r = np.exp(lp - lse)
            nk = r.sum(0)
            pi = nk / len(Z)
            mu = (r.T @ Z) / nk[:, None]
            D = Z[None] - mu[:, None]
            cov = np.matmul((D * r.T[:, :, None]).transpose(0, 2, 1), D) / nk[:, None, None] + 1e-6 * np.eye(d)
            if ll - ll_old < 1e-8 * len(Z):
                break
            ll_old = ll
        fit = {"pi": pi, "mu": mu, "cov": cov, "ll": ll, "iters": it + 1, "converged": it < max_iter - 1, "min_weight": pi.min()}
        if best is None or fit["ll"] > best["ll"]:
            best = fit
    return best


def task(args):
    # 第 rep 個工作：rep = 0 為全樣本，其餘為重抽玩家；種子只由 (速度, K, rep) 決定，與排程無關
    Z, K, seed, rep, starts = args
    rng = np.random.default_rng(seed)
    Zb = Z if rep == 0 else Z[rng.integers(0, len(Z), len(Z))]
    f = gmm(Zb, K, rng, starts)
    return K, rep, f, predict(Z, f)
