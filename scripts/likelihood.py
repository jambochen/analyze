# 對局資料的推論：逐塊最大概似、dyadic 的 sandwich 共變異與標準誤、修正的概似比檢定、Wald 檢定、Holm 校正。
# 一局有兩位玩家，同一玩家又下很多局，觀察值不獨立，一般概似比統計量的虛無分布不是卡方。

import functools
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import likelihood_rs
import numpy as np
from scipy import integrate, optimize, stats

jax.config.update("jax_enable_x64", True)

CHUNK = 2**18


# 任兩局只要有一位共同玩家，分數的乘積就計入 meat。sum_p S_p S_p' 對同一對手組合的對局
# 重複計了兩次，減去以無序對手組合為群的一次。Rust 版（rust/likelihood）；numpy 參考實作見
# scripts/likelihood_rust_matches_python.py
def dyad_meat(S, w, b):
    return likelihood_rs.dyad_meat(np.ascontiguousarray(np.reshape(S, (len(w), -1)), dtype=np.float64), np.asarray(w, dtype=np.int64), np.asarray(b, dtype=np.int64))


# 純量估計量的 dyadic 標準誤，phi 為逐局的影響函數。平均的影響函數是 (x - 平均) / n
def dyad_se(phi, w, b):
    return float(np.sqrt(dyad_meat(phi, w, b)[0, 0]))


# 線性限制 R theta = r 的 Wald 檢定，回傳 (統計量, 自由度, p 值)
def wald(theta, V, R, r):
    R = np.atleast_2d(R)
    diff = R @ theta - r
    stat = float(diff @ np.linalg.solve(R @ V @ R.T, diff))
    return stat, len(diff), stats.chi2.sf(stat, len(diff))


# Holm 法控制族錯誤率。family 為布林遮罩時只校正族內，族外為 NaN；不給即全部為一族
def holm(p, family=None):
    p = np.asarray(p, dtype=float)
    fi = np.arange(len(p)) if family is None else np.flatnonzero(family)
    o = fi[np.argsort(p[fi])]
    out = np.full(len(p), np.nan)
    out[o] = np.minimum(1, np.maximum.accumulate(p[o] * (len(o) - np.arange(len(o)))))
    return out


# 以得分為反應、線性預測式 X theta 的 logistic 迴歸（fractional logit）的逐局準對數概似。
# fit 遇到它時改由 Rust（rust/likelihood）計算；這裡的 jax 寫法即其參考實作
def logit_ll(th, X, y):
    eta = X @ th
    return y * jax.nn.log_sigmoid(eta) + (1 - y) * jax.nn.log_sigmoid(-eta)


# 以 ll 為鍵快取 jit 過的函式，t0、free 當作輸入而非閉包：每次 fit 都建新閉包時，每次呼叫都要重新編譯，
# 編譯時間與資料量無關，小樣本的 fit 幾乎全花在編譯上。值、梯度與 Hessian 在同一個 jit 內一次算出
@functools.cache
def _compiled(ll):
    def nll(x, t0, free, mask, *a):
        return -jnp.sum(mask * ll(t0.at[free].set(x), *a))

    def vgh(x, t0, free, mask, *a):
        v, g = jax.value_and_grad(nll)(x, t0, free, mask, *a)
        return v, g, jax.hessian(nll)(x, t0, free, mask, *a)

    score = lambda x, t0, free, *a: jax.jacfwd(lambda x: ll(t0.at[free].set(x), *a))(x)
    return jax.jit(vgh), jax.jit(score)


# ll(t, *args) 回傳逐局的對數概似；free 以外的參數固定在 t0。回傳 (完整參數, 對數概似, sandwich 共變異, H^-1, 逐局分數, H)，
# 後四者只含 free 的參數
def fit(ll, t0, free, args, w, b):
    free = np.asarray(free, dtype=np.int64)
    t0 = jnp.asarray(t0, dtype=jnp.float64)
    full = lambda x: np.asarray(t0.at[free].set(x))
    if ll is logit_ll:
        # XLA 在 CPU 上對逐元素運算與加總只平行到兩三個核，Rust 版各列以 rayon 平行
        X, y = (np.ascontiguousarray(a, dtype=np.float64) for a in args)
        evaluate = lambda x: likelihood_rs.vgh(X, y, full(x), free)
        scores = lambda x: likelihood_rs.score(X, y, full(x), free)
    else:
        # 概似、梯度與 Hessian 逐塊加總，避免 jax 在數百萬列上一次展開中間陣列。每塊補齊到 2 的冪次長度
        # （以第一列填充、mask 為 0），形狀只有少數幾種，不同樣本大小的 fit 共用編譯結果
        vgh_, score_ = _compiled(ll)
        cs = []
        for k in range(0, len(w), CHUNK):
            m = min(CHUNK, len(w) - k)
            idx = np.r_[k : k + m, np.full((1 << (m - 1).bit_length()) - m, k)]
            cs.append((m, (np.arange(len(idx)) < m).astype(np.float64), tuple(a[idx] for a in args)))

        def evaluate(x):
            out = [vgh_(x, t0, free, mk, *c) for _, mk, c in cs]
            return float(sum(o[0] for o in out)), np.asarray(sum(o[1] for o in out)), np.asarray(sum(o[2] for o in out))

        scores = lambda x: np.concatenate([np.asarray(score_(x, t0, free, *c))[:m] for m, _, c in cs])
    # 受限模型可能把參數全部固定（例如 Elo 完全校準），此時只需算固定參數下的對數概似
    if len(free) == 0:
        z = np.zeros((0, 0))
        return np.asarray(t0), -evaluate(np.zeros(0))[0], z, z, np.zeros((len(w), 0)), z
    # trust-exact 在同一點依序要值、梯度與 Hessian，只記住最後一點即可
    memo = {}

    def at(x):
        key = np.asarray(x).tobytes()
        if key not in memo:
            memo.clear()
            memo[key] = evaluate(x)
        return memo[key]

    r = optimize.minimize(lambda x: at(x)[0], np.asarray(t0[free]), jac=lambda x: at(x)[1], hess=lambda x: at(x)[2], method="trust-exact")
    H = at(r.x)[2]
    S = scores(r.x)
    Hi = np.linalg.inv(H)
    return full(r.x), -r.fun, Hi @ dyad_meat(S, w, b) @ Hi, Hi, S, H


# 以得分為反應的 logistic 迴歸（fractional logit）的 Newton 法，numpy。jax 的 fit 每次都要逐塊加總、經 scipy 迭代，
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


# sigmoid(alpha + beta L) 的 score 統計量 U' I^-1 U，只需在虛無假設下計算。各局當成獨立、quasi-likelihood 取 phi = 1；
# 和局使 phi < 1，統計量偏小、區間偏寬
def frac_score(L, s, t):
    X = np.column_stack([np.ones_like(L), L])
    p = 1 / (1 + np.exp(-(t[0] + t[1] * L)))
    U = X.T @ (s - p)
    return U @ np.linalg.solve(X.T @ (X * (p * (1 - p))[:, None]), U)


# 固定第 j 個參數為 v，另一個以牛頓法解虛無假設下的估計，再算 score 統計量
def frac_score_at(L, s, j, v, t0):
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
    return frac_score(L, s, t)


# 反轉 score test 的 95% 區間；th 為估計值，se 只用來框出 brentq 的搜尋範圍
def frac_score_ci(L, s, j, th, se):
    f = lambda v: frac_score_at(L, s, j, v, th) - stats.chi2.ppf(0.95, 1)
    return optimize.brentq(f, th[j] - 4 * se, th[j]), optimize.brentq(f, th[j], th[j] + 4 * se)


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
    Vp = Hi @ likelihood_rs.group_outer(np.ascontiguousarray(S, dtype=np.float64), np.asarray(w, dtype=np.int64)) @ Hi
    lam_d, lam_p = lr_weights(Hi, V, idx), lr_weights(Hi, Vp, idx)
    lam = lam_d if lam_d.max() >= lam_p.max() else lam_p
    return th1, V, adjusted_lrt(2 * (ll1 - ll0), lam) | {"lam_max_dyadic": lam_d.max(), "lam_max_player": lam_p.max()}
