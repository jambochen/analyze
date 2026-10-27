# 隱藏狀態：玩家在同一場錦標賽內是否在幾個持續的狀態（例如失常）之間轉移，使結果偏離主基準。
# 規格在執行前寫定：
# - 序列：玩家 × 錦標賽的全部 arena 局，依開局時間排序；每場重新開始
# - 模型：K 個狀態，狀態 s 下本局為有序 logistic，線性預測式為主基準的 eta 加 mu_s，和局帶寬為主基準的 w；
#   狀態依一階 Markov 轉移，轉移矩陣與每場第一局的狀態分布由資料估計，全體玩家共用。K = 2 為主，K = 3 為對照，
#   K = 1 即主基準加一個常數平移。mu 依大小排序，避免標籤互換
# - 估計：上半年開始的錦標賽，以 forward algorithm 的對數概似做最大概似，數個起始值取概似最大者
# - 預測：forward filter 的一步預測 P(y_t | y_1..t-1)，只用對局前的資訊；參數固定為上半年估得的值
# - 評估：下半年開始的錦標賽中，同場有上一局的局，與 scripts/markov_order.py 的樣本相同。
#   對數分數與 RPS 對主基準及「上一局結果」（一階 Markov）的差，dyadic sandwich 標準誤；
#   判準同上：改善且 95% 區間不含 0
# - 狀態數不以概似比檢定決定：K 個狀態對 K - 1 的虛無假設在參數空間邊界上，概似比不是卡方，只看樣本外
# - 只用自身的序列，對手的狀態不進模型

import os
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, "scripts")

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
from scipy import optimize

import likelihood as lk
import ordinal
from arena_batches import player_sample_rd

UTC = "UTC"
SPEEDS = ["Bullet", "Blitz", "Classical"]
STARTS = 4

grp = ["player", "tournament_id"]
t = (
    player_sample_rd()
    .sort("player", "tournament_id", "start", "game_id")
    .with_columns(
        y=(pl.col("score") * 2).cast(pl.Int64),
        pos=pl.int_range(pl.len()).over(grp),
        seq_len=pl.len().over(grp),
        prev_result=pl.col("result").shift(1).over(grp),
        train=pl.col("start").first().over(grp) < pl.datetime(2014, 7, 1, time_zone=UTC),
    )
    .with_columns(
        prev_draw=(pl.col("prev_result") == "draw").cast(pl.Float64),
        prev_win=(pl.col("prev_result") == "win").cast(pl.Float64),
    )
)


# 依序列長度分到 2 的冪次的桶，每桶補齊到該桶的最大長度；補齊的位置 mask 為 0
def buckets(d):
    d = d.with_columns(seq=pl.struct(grp).rank("dense") - 1, b=pl.col("seq_len").log(2).ceil().cast(pl.Int64))
    out = []
    for _, z in d.group_by("b"):
        z = z.with_columns(s=pl.col("seq").rank("dense") - 1)
        S, T = z["s"].max() + 1, z["seq_len"].max()
        s, p = z["s"].to_numpy(), z["pos"].to_numpy()
        arr = {c: np.zeros((S, T)) for c in ["eta", "w", "y", "mask", "row"]}
        arr["w"][:] = 1.0
        for c in ["eta", "w", "y", "row"]:
            arr[c][s, p] = z[c].to_numpy()
        arr["mask"][s, p] = 1.0
        out.append(arr)
    return out


def unpack(x, K):
    mu = jnp.cumsum(jnp.concatenate([x[:1], jnp.exp(x[1:K])]))
    A = jax.nn.log_softmax(jnp.concatenate([x[K : K + K * (K - 1)].reshape(K, K - 1), jnp.zeros((K, 1))], 1), 1)
    pi = jax.nn.log_softmax(jnp.concatenate([x[K + K * (K - 1) :], jnp.zeros(1)]))
    return mu, A, pi


# forward filter；回傳總對數概似與每個位置的一步預測狀態分布（對數）
def forward(x, K, eta, w, y, mask):
    mu, A, pi = unpack(x, K)
    emis = ordinal.ll(eta[..., None] + mu, w[..., None], y[..., None].astype(jnp.int64))

    def step(prior, inp):
        e, m = inp
        lp = jax.scipy.special.logsumexp(prior + e, 1)
        post = prior + e - lp[:, None]
        nxt = jax.scipy.special.logsumexp(post[:, :, None] + A[None], 1)
        return nxt, (lp * m, prior)

    prior0 = jnp.broadcast_to(pi, (eta.shape[0], K))
    _, (lps, priors) = jax.lax.scan(step, prior0, (jnp.swapaxes(emis, 0, 1), mask.T))
    return lps.sum(), jnp.swapaxes(priors, 0, 1)


def total_ll(x, K, bs):
    return sum(forward(x, K, b["eta"], b["w"], b["y"], b["mask"])[0] for b in bs)


def fit_hmm(K, bs, rng):
    f = jax.jit(jax.value_and_grad(lambda x, bs: -total_ll(x, K, bs)))
    best = None
    for _ in range(STARTS):
        x0 = np.r_[rng.normal(-0.3, 0.1), np.log(rng.uniform(0.2, 0.6, K - 1)), rng.normal(2, 0.5, K * (K - 1)), np.zeros(K - 1)]
        r = optimize.minimize(lambda x: tuple(np.asarray(v, dtype=np.float64) for v in f(x, bs)), x0, jac=True, method="L-BFGS-B")
        if best is None or r.fun < best.fun:
            best = r
    return best


def predict(x, K, bs, n):
    mu = np.asarray(unpack(jnp.asarray(x), K)[0])
    P = np.zeros((n, 3))
    for b in bs:
        prior = np.exp(np.asarray(forward(jnp.asarray(x), K, b["eta"], b["w"], b["y"], b["mask"])[1]))
        m = b["mask"] > 0
        pk = np.stack([ordinal.probs(b["eta"][m] + mu[k], b["w"][m]) for k in range(K)], 1)
        P[b["row"][m].astype(np.int64)] = (prior[m][:, :, None] * pk).sum(1)
    return P


def losses(p, y):
    O = np.eye(3)[y]
    C, Co = np.cumsum(p, 1)[:, :2], np.cumsum(O, 1)[:, :2]
    return {"log_loss": -np.log(p[np.arange(len(y)), y]), "rps": ((C - Co) ** 2).sum(1) / 2}


def ll_prev(th, eta, w, X, y):
    return ordinal.ll(eta + X @ th, w, y)


rng = np.random.default_rng(0)
par_rows, pred_rows = [], []
for spd in SPEEDS:
    ts = t.filter(pl.col("speed") == spd)
    tr, te = (ts.filter(m).with_columns(row=pl.int_range(pl.len())) for m in [pl.col("train"), ~pl.col("train")])
    btr, bte = buckets(tr), buckets(te)
    ev = te.filter(pl.col("prev_result").is_not_null())
    idx, y_ev = ev["row"].to_numpy(), ev["y"].to_numpy()
    P = {"baseline": ordinal.probs(ev["eta"].to_numpy(), ev["w"].to_numpy())}
    trp = tr.filter(pl.col("prev_result").is_not_null())
    X = lambda d: d.select("prev_draw", "prev_win").to_numpy()
    th = lk.fit(ll_prev, np.zeros(2), np.array([0, 1]), (trp["eta"].to_numpy(), trp["w"].to_numpy(), X(trp), trp["y"].to_numpy()), trp["player"].to_numpy(), trp["opponent"].to_numpy())[0]
    P["prev_result"] = ordinal.probs(ev["eta"].to_numpy() + X(ev) @ th, ev["w"].to_numpy())
    for K in [1, 2, 3]:
        r = fit_hmm(K, btr, rng)
        mu, A, pi = (np.asarray(v) for v in unpack(jnp.asarray(r.x), K))
        A = np.exp(A)
        # 轉移矩陣的平穩分布：A' 的特徵值 1 對應的特徵向量
        ev_, vec = np.linalg.eig(A.T)
        st = np.real(vec[:, np.argmin(abs(ev_ - 1))])
        st = st / st.sum()
        for k in range(K):
            par_rows.append({"speed": spd, "K": K, "state": k, "mu": mu[k], "pi": np.exp(pi[k]), "stay": A[k, k], "stationary": st[k], "train_ll": -r.fun, "converged": r.success})
        P[f"hmm{K}"] = predict(r.x, K, bte, te.height)[idx]
        print(spd, K, "done", flush=True)
    L = {k: losses(v, y_ev) for k, v in P.items()}
    wte, bte_ = ev["player"].to_numpy(), ev["opponent"].to_numpy()
    for vs in ["baseline", "prev_result"]:
        for nm in P:
            for metric in ["log_loss", "rps"]:
                dlt = L[nm][metric] - L[vs][metric]
                se = lk.dyad_se((dlt - dlt.mean()) / len(dlt), wte, bte_)
                pred_rows.append({"speed": spd, "model": nm, "vs": vs, "metric": metric, "n": len(dlt), "mean": L[nm][metric].mean(), "diff": dlt.mean(), "diff_se_dyadic": se, "better": dlt.mean() + 1.96 * se < 0})

par = pl.DataFrame(par_rows)
pred = pl.DataFrame(pred_rows)
with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250):
    print(par)
    print(pred.filter(pl.col("metric") == "log_loss"))
par.write_csv("outputs/tables/hmm_tilt_params.csv")
pred.write_csv("outputs/tables/hmm_tilt_prediction.csv")
