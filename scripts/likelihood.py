# 對局資料的概似推論：逐塊最大概似、dyadic 的 sandwich 共變異、修正的概似比檢定。
# 一局有兩位玩家，同一玩家又下很多局，觀察值不獨立，一般概似比統計量的虛無分布不是卡方。

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
from scipy import integrate, optimize, stats

jax.config.update("jax_enable_x64", True)

CHUNK = 250_000


# 任兩局只要有一位共同玩家，分數的乘積就計入 meat。sum_p S_p S_p' 對同一對手組合的對局
# 重複計了兩次，減去以無序對手組合為群的一次
def dyad_meat(S, w, b):
    S = S.reshape(len(w), -1)
    P = int(max(w.max(), b.max())) + 1
    Sp = np.stack([np.bincount(w, S[:, j], P) + np.bincount(b, S[:, j], P) for j in range(S.shape[1])], 1)
    pair = np.unique(np.minimum(w, b).astype(np.int64) * P + np.maximum(w, b), return_inverse=True)[1]
    Sq = np.stack([np.bincount(pair, S[:, j]) for j in range(S.shape[1])], 1)
    return Sp.T @ Sp - Sq.T @ Sq


# ll(t, *args) 回傳逐局的對數概似；free 以外的參數固定在 t0。概似、梯度與 Hessian 逐塊加總，
# 避免 jax 在數百萬列上一次展開中間陣列。回傳 (完整參數, 對數概似, sandwich 共變異, H^-1, 逐局分數, H)，
# 後四者只含 free 的參數
def fit(ll, t0, free, args, w, b):
    free = np.asarray(free)
    t0 = jnp.asarray(t0, dtype=jnp.float64)
    full = lambda x: t0.at[free].set(x)
    nll = lambda x, *a: -jnp.sum(ll(full(x), *a))
    cs = [tuple(a[k : k + CHUNK] for a in args) for k in range(0, len(w), CHUNK)]
    # 受限模型可能把參數全部固定（例如 Elo 完全校準），此時只需算固定參數下的對數概似
    if len(free) == 0:
        f0 = jax.jit(lambda *a: jnp.sum(ll(t0, *a)))
        z = np.zeros((0, 0))
        return np.asarray(t0), float(sum(f0(*c) for c in cs)), z, z, np.zeros((len(w), 0)), z
    f_, g_, h_ = jax.jit(nll), jax.jit(jax.grad(nll)), jax.jit(jax.hessian(nll))
    sc = jax.jit(jax.jacfwd(lambda x, *a: ll(full(x), *a)))
    r = optimize.minimize(
        lambda x: float(sum(f_(x, *c) for c in cs)), np.asarray(t0[free]),
        jac=lambda x: np.asarray(sum(g_(x, *c) for c in cs)),
        hess=lambda x: np.asarray(sum(h_(x, *c) for c in cs)),
        method="trust-exact",
    )
    H = np.asarray(sum(h_(r.x, *c) for c in cs))
    S = np.concatenate([np.asarray(sc(r.x, *c)) for c in cs])
    Hi = np.linalg.inv(H)
    return np.asarray(full(r.x)), -r.fun, Hi @ dyad_meat(S, w, b) @ Hi, Hi, S, H


# 以得分為反應的 logistic 迴歸（fractional logit）的 Newton 法，numpy。jax 的 fit 每次都要 jit 與逐塊加總，
# 參數少、要重估上千次的 bootstrap 與 profile 用這個。回傳 (參數, 配適機率, 負 Hessian, quasi 對數概似)
def frac_newton(X, s, t):
    t = np.array(t, dtype=float)
    for _ in range(50):
        p = 1 / (1 + np.exp(-X @ t))
        H = X.T @ (X * (p * (1 - p))[:, None])
        step = np.linalg.solve(H, X.T @ (s - p))
        t += step
        if np.abs(step).max() < 1e-10:
            break
    p = 1 / (1 + np.exp(-X @ t))
    return t, p, X.T @ (X * (p * (1 - p))[:, None]), float(np.sum(s * np.log(p) + (1 - s) * np.log1p(-p)))


# 工作概似下，受檢參數 psi 的概似比統計量漸近為 sum_j lambda_j chi2_1，
# lambda 為 [H^-1]_psi^-1 [V]_psi 的特徵值（Rotnitzky、Jewell 1990）。idx 是受檢參數在 free 中的位置
def lr_weights(Hi, V, idx):
    idx = np.asarray(idx)
    return np.sort(np.real(np.linalg.eigvals(np.linalg.solve(Hi[np.ix_(idx, idx)], V[np.ix_(idx, idx)]))))


# 受檢的是線性限制 R theta = r 時，同理取 (R H^-1 R')^-1 (R V R') 的特徵值
def lr_weights_linear(Hi, V, R):
    R = np.atleast_2d(R)
    return np.sort(np.real(np.linalg.eigvals(np.linalg.solve(R @ Hi @ R.T, R @ V @ R.T))))


# Imhof (1961) 的反轉公式。被積函數是 sin(A(u) - x u / 2) / (u rho(u))，在無窮區間上振盪；
# 以 sin(A - w u) = sin A cos wu - cos A sin wu 拆開，第一個週期以後交給 Fourier 積分（QAWF），
# 否則一般的 quad 會耗盡細分次數且失去精度。cos A / u 在 0 附近發散，所以第一個週期照原式積
def wchisq_sf_imhof(x, lam):
    lam = np.asarray(lam, dtype=float)
    om = 0.5 * x
    A = lambda u: 0.5 * np.arctan(lam * u).sum()
    rho = lambda u: np.prod((1 + (lam * u) ** 2) ** 0.25)
    a = 2 * np.pi / om
    head, _ = integrate.quad(lambda u: np.sin(A(u) - om * u) / (u * rho(u)), 0, a, limit=200)
    c, _ = integrate.quad(lambda u: np.sin(A(u)) / (u * rho(u)), a, np.inf, weight="cos", wvar=om)
    s, _ = integrate.quad(lambda u: np.cos(A(u)) / (u * rho(u)), a, np.inf, weight="sin", wvar=om)
    return 0.5 + (head + c - s) / np.pi


# 以 c chi2_nu 配前兩階動差（Satterthwaite）
def wchisq_sf_satt(x, lam):
    lam = np.asarray(lam, dtype=float)
    c = (lam**2).sum() / lam.sum()
    nu = lam.sum() ** 2 / (lam**2).sum()
    return stats.chi2.sf(x / c, nu)


# df = 1 時虛無分布就是 lambda chi2_1，有精確解。df > 1 時用 Imhof：它的振盪積分在統計量大時
# 失去精度，甚至算出負值，所以尾端改報上界：每個 lambda_j <= lambda_max，
# 故 sum lambda_j chi2_1 隨機地小於 lambda_max chi2_df，P(lambda_max chi2_df > lr) 是嚴格的上界
def adjusted_lrt(lr, lam):
    df = len(lam)
    p_upper = stats.chi2.sf(lr / lam.max(), df)
    if df == 1:
        p, method = stats.chi2.sf(lr / lam[0], 1), "exact"
    elif p_upper < 1e-6:
        p, method = p_upper, "upper_bound"
    else:
        p, method = wchisq_sf_imhof(lr, lam), "imhof"
    return {"lr": lr, "df": df, "p_naive": stats.chi2.sf(lr, df), "p": p, "p_method": method, "p_upper": p_upper, "p_satt": wchisq_sf_satt(lr, lam), "lam_min": lam.min(), "lam_max": lam.max()}


# 以局加權的修正概似比檢定：lambda 以依玩家分群（w 為群）與 dyadic 的 meat 各算一次，取較大者（較保守）。
# 回傳 (完整模型的參數, 完整模型的 dyadic 共變異, 檢定結果)
def lrt_player_dyad(ll, t0, f0, f1, args, w, b):
    th1, ll1, V, Hi, S, _ = fit(ll, t0, f1, args, w, b)
    _, ll0, *_ = fit(ll, t0, f0, args, w, b)
    idx = np.flatnonzero(~np.isin(f1, f0))
    pc = np.unique(w, return_inverse=True)[1]
    Sp = np.stack([np.bincount(pc, S[:, j]) for j in range(S.shape[1])], 1)
    Vp = Hi @ (Sp.T @ Sp) @ Hi
    lam_d, lam_p = lr_weights(Hi, V, idx), lr_weights(Hi, Vp, idx)
    lam = lam_d if lam_d.max() >= lam_p.max() else lam_p
    return th1, V, adjusted_lrt(2 * (ll1 - ll0), lam) | {"lam_max_dyadic": lam_d.max(), "lam_max_player": lam_p.max()}
