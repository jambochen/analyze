# 修正的概似比檢定在 dyadic 相依下的檢定水準：模擬資料上，虛無為真時拒絕率是否接近名目水準。
# 對局結果受未觀測的玩家實力影響，受檢的共變數是玩家層的，所以同一玩家的對局相關，
# 一般的概似比檢定會過度拒絕；修正後應接近名目水準。
# sandwich 在有效群數少時低估變異數，所以分情境看：玩家活躍度越集中、玩家越少，有效群數越少。
# 每批 BATCH 次重複以 vmap 同時做 Newton 迭代：逐次呼叫最佳化器時，時間幾乎全花在呼叫開銷上

import os
import sys

# 先於 import jax 設定：float64 在這張 GPU 上很慢
os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, "scripts")

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl

import likelihood as lk
import ordinal

N_GAME, REPS, BATCH = 20_000, 1_000, 500
# (玩家數, 活躍度的 Pareto 形狀參數)；形狀參數越小，少數玩家佔的局數越多
SCENARIOS = [(2_000, 1.5), (2_000, 3.0), (10_000, 1.5), (10_000, 3.0)]
LEVELS = [0.01, 0.05, 0.10]
K = 5
T0 = np.array([0, 0.5, 0, 0, -1.0])


def ll(t, xd, z1, z2, y):
    # 與正式分析共用同一份概似，模擬檢驗的才是正式用的那一份
    return ordinal.ll(t[0] + t[1] * xd + t[2] * z1 + t[3] * z2, jnp.exp(t[4]), y)


# 受限參數固定為 0；回傳 (參數, 對數概似, H, 逐局分數)，皆帶批的維度
def batch_fit(free, data):
    free = np.asarray(free)
    t0 = jnp.asarray(T0).at[np.setdiff1d(np.arange(K), free)].set(0.0)
    full = lambda x: t0.at[free].set(x)
    nll = lambda x, *a: -jnp.sum(ll(full(x), *a))
    f = jax.jit(jax.vmap(nll))
    g = jax.jit(jax.vmap(jax.grad(nll)))
    h = jax.jit(jax.vmap(jax.hessian(nll)))
    x = jnp.tile(t0[free], (data[0].shape[0], 1))
    fx = f(x, *data)
    for _ in range(50):
        step = jnp.linalg.solve(h(x, *data), g(x, *data)[..., None])[..., 0]
        # 逐次減半步長，直到每一次重複的負對數概似都不上升
        a = jnp.ones(x.shape[0])
        for _ in range(20):
            bad = f(x - a[:, None] * step, *data) > fx + 1e-9
            if not bool(bad.any()):
                break
            a = jnp.where(bad, a / 2, a)
        x = x - a[:, None] * step
        fx = f(x, *data)
        if float(jnp.abs(a[:, None] * step).max()) < 1e-9:
            break
    S = jax.vmap(jax.jacfwd(lambda x, *a: ll(full(x), *a)))(x, *data)
    return np.asarray(x), -np.asarray(fx), np.asarray(h(x, *data)), np.asarray(S)


# 每次重複各自的 dyadic meat：玩家代碼與對手組合加上重複次數的偏移，一次 bincount 算完整批
def batch_meat(S, w, b, n_player):
    R, n, k = S.shape
    rep = np.repeat(np.arange(R), n)
    wo, bo = (w + rep.reshape(R, n) * n_player).ravel(), (b + rep.reshape(R, n) * n_player).ravel()
    Sf = S.reshape(R * n, k)
    Sp = np.stack([np.bincount(wo, Sf[:, j], R * n_player) + np.bincount(bo, Sf[:, j], R * n_player) for j in range(k)], 1).reshape(R, n_player, k)
    pair = (np.minimum(w, b).astype(np.int64) * n_player + np.maximum(w, b)).ravel() + rep * n_player**2
    _, inv = np.unique(pair, return_inverse=True)
    Sq = np.stack([np.bincount(inv, Sf[:, j]) for j in range(k)], 1)
    pair_rep = np.zeros(inv.max() + 1, np.int64)
    pair_rep[inv] = rep
    meat = np.einsum("rpi,rpj->rij", Sp, Sp)
    np.subtract.at(meat, pair_rep, np.einsum("qi,qj->qij", Sq, Sq))
    return meat


rng = np.random.default_rng(7)
rows = []
for n_player, shape in SCENARIOS:
    for start in range(0, REPS, BATCH):
        W, B, data, eff = [], [], [], []
        for _ in range(BATCH):
            r = rng.normal(0, 1, n_player)
            hidden = rng.normal(0, 1, n_player)
            zp = rng.normal(0, 1, (n_player, 2))
            act = rng.pareto(shape, n_player) + 1
            w = rng.choice(n_player, N_GAME, p=act / act.sum())
            b = rng.choice(n_player, N_GAME, p=act / act.sum())
            # 每次重複的局數相同，整批才能疊成一個陣列
            b = np.where(w == b, (b + 1) % n_player, b)
            eta = 0.1 + 0.8 * (r[w] - r[b]) + 1.0 * (hidden[w] - hidden[b])
            e = rng.logistic(size=N_GAME)
            y = np.where(eta - e > 0.4, 2, np.where(eta - e > -0.4, 1, 0))
            # 有效玩家數：各玩家局數占比平方和的倒數
            share = np.bincount(np.concatenate([w, b]), minlength=n_player) / (2 * N_GAME)
            eff.append(1 / (share**2).sum())
            W.append(w)
            B.append(b)
            data.append(np.stack([r[w] - r[b], zp[w, 0] - zp[b, 0], zp[w, 1] - zp[b, 1], y.astype(np.float64)]))
        W, B, data = np.stack(W), np.stack(B), np.stack(data)
        args = tuple(jnp.asarray(data[:, j]) for j in range(4))
        _, ll1, H, S = batch_fit(np.arange(K), args)
        Hi = np.linalg.inv(H)
        V = Hi @ batch_meat(S, W, B, n_player) @ Hi
        for name, idx in [("one_df", [2]), ("two_df", [2, 3])]:
            _, ll0, *_ = batch_fit(np.setdiff1d(np.arange(K), idx), args)
            for i in range(BATCH):
                res = lk.adjusted_lrt(2 * (ll1[i] - ll0[i]), lk.lr_weights(Hi[i], V[i], idx))
                rows.append({"n_player": n_player, "pareto_shape": shape, "effective_players": eff[i], "rep": start + i, "test": name, **res})
        print(n_player, shape, start + BATCH, "reps")

sim = pl.DataFrame(rows)
size = (
    sim.group_by("n_player", "pareto_shape", "test")
    .agg(
        reps=pl.len(), effective_players=pl.col("effective_players").mean(), lam_max_mean=pl.col("lam_max").mean(),
        *[(pl.col("p_naive") < a).mean().alias(f"naive_{a}") for a in LEVELS],
        *[(pl.col("p") < a).mean().alias(f"adjusted_{a}") for a in LEVELS],
        *[(pl.col("p_satt") < a).mean().alias(f"satterthwaite_{a}") for a in LEVELS],
    )
    .sort("n_player", "pareto_shape", "test")
)
size.write_csv("outputs/tables/adjusted_lrt_size.csv")
print(size)
