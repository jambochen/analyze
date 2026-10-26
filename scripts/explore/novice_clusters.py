# 新手期的偏差收斂是否因人而異：以前 10 局的等級分軌跡把新帳號分群，再看各群第 11 局以後的校準斜率。
# 規格見 docs/methods.md「新手期的界定」的「收斂速度是否因人而異」。推論把對局當成獨立。

import sys

sys.path.insert(0, "scripts")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import stats
from scipy.special import logsumexp

import likelihood as lk
import novice
import ordinal

UTC = "UTC"
CLASSES = ["Bullet", "Blitz", "Classical"]
N_FEAT = 10
KS = range(2, 7)
B_STAB = 20
STARTS = 5
K_BINS = [10, 15, 20, 30, 50, 100]
K_LABELS = [f"{a}-{b - 1}" for a, b in zip(K_BINS[:-1], K_BINS[1:])]

g = (
    pl.scan_parquet("data/processed/lichess_games.parquet")
    .filter(
        pl.col("speed").is_in(CLASSES), pl.col("base_seconds").is_not_null(),
        pl.col("start") >= pl.datetime(2014, 1, 1, time_zone=UTC), pl.col("start") < pl.datetime(2015, 1, 1, time_zone=UTC),
        pl.col("white_elo").is_not_null(), pl.col("black_elo").is_not_null(), pl.col("result").is_in(["1-0", "0-1", "1/2-1/2"]),
    )
    .select("game_id", "start", "speed", "white_id", "black_id", "white_elo", "black_elo", s=pl.col("result").replace_strict({"0-1": 0.0, "1/2-1/2": 0.5, "1-0": 1.0}))
    .join(pl.scan_parquet("data/processed/glicko_replay.parquet").select("game_id", "perf", "white_rd", "black_rd", "white_left_truncated", "black_left_truncated", "white_prior_games", "black_prior_games"), on="game_id")
    .filter(pl.col("perf") == pl.col("speed"))
)
# 玩家視角：每局兩列
side = pl.concat([
    g.select(
        "game_id", "start", "speed", p=f"{c}_id", o=f"{o}_id", elo=f"{c}_elo", opp_elo=f"{o}_elo", rd=f"{c}_rd", opp_rd=f"{o}_rd",
        lt=f"{c}_left_truncated", opp_lt=f"{o}_left_truncated", k=f"{c}_prior_games", score=pl.col("s") if c == "white" else 1 - pl.col("s"),
    )
    for c, o in [("white", "black"), ("black", "white")]
]).collect(engine="streaming")

# 沒有更新等級分的對局不增加此前局數，同一個 k 可能有多局；軌跡取每個 k 的第一局
first = side.filter(pl.col("k") <= N_FEAT).sort("start", "game_id").group_by("speed", "p", "k").first()
eligible = (
    first.group_by("speed", "p")
    .agg(
        n_k=pl.len(),
        start_ok=((pl.col("k") == 0) & (pl.col("elo") == 1500) & ~pl.col("lt")).any(),
        elo=pl.col("elo").sort_by("k"),
        rd10=pl.col("rd").filter(pl.col("k") == N_FEAT).first(),
    )
    .filter(pl.col("n_k") == N_FEAT + 1, pl.col("start_ok"))
)
feat = eligible.with_columns(
    drift=pl.col("elo").list.get(N_FEAT) - 1500,
    late_drift=pl.col("elo").list.get(N_FEAT) - pl.col("elo").list.get(N_FEAT // 2),
    mean_abs_change=pl.col("elo").list.diff().list.drop_nulls().list.eval(pl.element().abs()).list.mean(),
).drop("elo", "n_k", "start_ok")
print(feat.group_by("speed").agg(n=pl.len(), rd10_med=pl.col("rd10").median(), rd10_share_ge110=(pl.col("rd10") >= novice.RD).mean()))
FEATS = ["drift", "late_drift", "mean_abs_change"]


# EM 的 Gaussian mixture，完整共變異；多個起點取概似最大者
def gmm(Z, K, rng):
    best = None
    for _ in range(STARTS):
        mu = Z[rng.choice(len(Z), K, replace=False)]
        cov = np.stack([np.cov(Z.T)] * K)
        pi = np.full(K, 1 / K)
        ll_old = -np.inf
        for it in range(1000):
            lp = np.stack([np.log(pi[j]) + stats.multivariate_normal.logpdf(Z, mu[j], cov[j]) for j in range(K)], 1)
            ll = logsumexp(lp, 1).sum()
            r = np.exp(lp - logsumexp(lp, 1, keepdims=True))
            nk = r.sum(0)
            pi = nk / len(Z)
            mu = (r.T @ Z) / nk[:, None]
            cov = np.stack([((Z - mu[j]).T * r[:, j]) @ (Z - mu[j]) / nk[j] + 1e-6 * np.eye(Z.shape[1]) for j in range(K)])
            if ll - ll_old < 1e-8 * len(Z):
                break
            ll_old = ll
        fit = {"pi": pi, "mu": mu, "cov": cov, "ll": ll, "iters": it + 1, "converged": it < 999, "min_weight": pi.min()}
        if best is None or fit["ll"] > best["ll"]:
            best = fit
    return best


def predict(Z, f):
    return np.argmax(np.stack([np.log(f["pi"][j]) + stats.multivariate_normal.logpdf(Z, f["mu"][j], f["cov"][j]) for j in range(len(f["pi"]))], 1), 1)


def ari(a, b):
    n = len(a)
    ct = np.bincount(a * (b.max() + 1) + b, minlength=(a.max() + 1) * (b.max() + 1)).reshape(a.max() + 1, b.max() + 1)
    c2 = lambda v: (v * (v - 1) / 2).sum()
    idx, ra, rb = c2(ct), c2(ct.sum(1)), c2(ct.sum(0))
    exp = ra * rb / (n * (n - 1) / 2)
    return (idx - exp) / ((ra + rb) / 2 - exp)


stab_rows, clus_rows, cal_rows, test_rows = [], [], [], []
labels_all = []
for k_tc, tc in enumerate(CLASSES):
    f = feat.filter(pl.col("speed") == tc).sort("p")
    Z = f.select(FEATS).to_numpy().astype(float)
    Z = (Z - Z.mean(0)) / Z.std(0)
    rng = np.random.default_rng([7, k_tc])
    fits = {}
    for K in KS:
        fits[K] = gmm(Z, K, rng)
        lab = predict(Z, fits[K])
        aris, jac = [], []
        for _ in range(B_STAB):
            ib = rng.integers(0, len(Z), len(Z))
            lb = predict(Z, gmm(Z[ib], K, rng))
            aris.append(ari(lab, lb))
            # Hennig 的群內 Jaccard：每個原群取與重抽各群的最大 Jaccard
            jac.append([max(((lab == j) & (lb == h)).sum() / ((lab == j) | (lb == h)).sum() for h in range(K)) for j in range(K)])
        jac = np.array(jac).mean(0)
        stab_rows.append({"speed": tc, "K": K, "n_players": len(Z), "ll": fits[K]["ll"], "converged": fits[K]["converged"], "iters": fits[K]["iters"], "min_weight": fits[K]["min_weight"], "ari_mean": np.mean(aris), "ari_p10": np.quantile(aris, 0.1), "jaccard_min": jac.min(), "jaccard_by_cluster": ",".join(f"{v:.3f}" for v in jac)})
        print(stab_rows[-1], flush=True)
    st = pl.DataFrame(stab_rows).filter(pl.col("speed") == tc)
    K = st.sort(["ari_mean", "K"], descending=[True, False])["K"][0]
    lab = predict(Z, fits[K])
    # 群的編號依漂移的平均排序，讓各速度類別的圖例可對照
    order = np.argsort([f["drift"].to_numpy()[lab == j].mean() for j in range(K)])
    lab = np.argsort(order)[lab]
    f = f.with_columns(cluster=pl.Series(lab))
    labels_all.append(f.select("speed", "p", "cluster"))
    clus_rows.append(f.group_by("speed", "cluster").agg(n=pl.len(), *[pl.col(c).mean().alias(c + "_mean") for c in FEATS], *[pl.col(c).std().alias(c + "_sd") for c in FEATS], rd10_med=pl.col("rd10").median()).sort("cluster").with_columns(K=pl.lit(K)))
    print(clus_rows[-1])

labels = pl.concat(labels_all)
clus = pl.concat(clus_rows)
stab = pl.DataFrame(stab_rows)
stab.write_csv("outputs/tables/novice_clusters_stability.csv")
clus.write_csv("outputs/tables/novice_clusters_summary.csv")

# 評估：第 11 局以後（此前局數 >= 10）、對手已非新手且非左截斷的對局，依群與此前局數的分段估校準截距與斜率
ev = (
    side.filter(pl.col("k") >= N_FEAT, pl.col("k") < K_BINS[-1], pl.col("opp_rd") < novice.RD, ~pl.col("opp_lt"))
    .join(labels, on=["speed", "p"])
    .with_columns(L=(pl.col("elo") - pl.col("opp_elo")) * ordinal.Q, kbin=pl.col("k").cut(K_BINS[1:-1], labels=K_LABELS, left_closed=True).cast(pl.String))
)
del side
for tc in CLASSES:
    e = ev.filter(pl.col("speed") == tc)
    for kb in e["kbin"].unique().sort().to_list():
        m = e.filter(pl.col("kbin") == kb)
        c = m["cluster"].to_numpy()
        K = c.max() + 1
        L, s = m["L"].to_numpy(), m["score"].to_numpy()
        D = np.eye(K)[c]
        # 各群自己的截距與斜率對「各群截距、共同斜率」
        t1, _, H1, ll1 = lk.frac_newton(np.column_stack([D, D * L[:, None]]), s, [0] * K + [1] * K)
        _, _, _, ll0 = lk.frac_newton(np.column_stack([D, L]), s, [0] * K + [1])
        Hi = np.linalg.inv(H1)
        for j in range(K):
            cal_rows.append({"speed": tc, "kbin": kb, "cluster": j, "n_games": int((c == j).sum()), "n_players": m.filter(pl.col("cluster") == j)["p"].n_unique(), "rd_med": float(np.median(m["rd"].to_numpy()[c == j])), "intercept": t1[j], "intercept_se_iid": np.sqrt(Hi[j, j]), "slope": t1[K + j], "slope_se_iid": np.sqrt(Hi[K + j, K + j])})
        test_rows.append({"speed": tc, "kbin": kb, "lr": 2 * (ll1 - ll0), "df": K - 1, "p_naive": stats.chi2.sf(2 * (ll1 - ll0), K - 1), "n_games": m.height})
cal = pl.DataFrame(cal_rows)
tests = pl.DataFrame(test_rows).sort("p_naive")
tests = tests.with_columns(p_holm=(pl.col("p_naive") * (pl.len() - pl.int_range(pl.len()))).cum_max().clip(upper_bound=1))
cal.write_csv("outputs/tables/novice_clusters_calibration.csv")
tests.write_csv("outputs/tables/novice_clusters_tests.csv")
print(cal)
print(tests)

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARK = ["o", "s", "^", "D", "v", "P"]
INK, MUTE = "#2b2b2b", "#8a8a85"
mid = {lab: (a + b - 1) / 2 for lab, a, b in zip(K_LABELS, K_BINS[:-1], K_BINS[1:])}
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
for ax, tc in zip(axes, CLASSES):
    c = cal.filter(pl.col("speed") == tc).with_columns(x=pl.col("kbin").replace_strict(mid, return_dtype=pl.Float64))
    sm = clus.filter(pl.col("speed") == tc)
    for j in range(c["cluster"].max() + 1):
        cj = c.filter(pl.col("cluster") == j).sort("x")
        dr = sm.filter(pl.col("cluster") == j)["drift_mean"][0]
        ax.errorbar(cj["x"] * (1 + 0.03 * (j - 1)), cj["slope"], yerr=1.96 * cj["slope_se_iid"], color=C[j], marker=MARK[j], ms=5, lw=1.5, capsize=2, label=f"cluster {j + 1} (mean drift {dr:+.0f})")
    ax.axhline(1, color=MUTE, lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_title(tc, color=INK)
    ax.set_xlabel("player's prior games (bin midpoint, log scale)")
    ax.grid(axis="y", color="#e6e6e3", lw=0.6)
    ax.legend(fontsize=7, frameon=False)
axes[0].set_ylabel("calibration slope of Elo E (player view)")
fig.suptitle("Calibration slope after the first 10 games, by early rating-trajectory cluster (iid 95% CI)")
fig.tight_layout()
fig.savefig("outputs/figures/novice_clusters_calibration.png", dpi=150)
