# 新手期的偏差收斂是否因人而異：以前 10 局的等級分軌跡把新帳號分群，再看各群第 11 局以後的校準斜率。
# 規格見 docs/methods.md「新手期的界定」的「收斂速度是否因人而異」，結論見 docs/results/novice-clusters.md。

import multiprocessing as mp
import sys

sys.path.insert(0, "scripts")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import game_source
from gmm import task
import likelihood as lk
import novice
import ordinal

N_PROC = 14
# 行程池在 polars 做任何運算之前 fork：polars 的執行緒池啟動後再 fork 可能死結
pool = mp.get_context("fork").Pool(N_PROC)

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
    .select("game_id", "start", "speed", "white_id", "black_id", "white_elo", "black_elo", arena=pl.col("tournament_id").is_not_null(), s=pl.col("result").replace_strict({"0-1": 0.0, "1/2-1/2": 0.5, "1-0": 1.0}))
    .join(pl.scan_parquet("data/processed/glicko_replay.parquet").select("game_id", "perf", "white_rd", "black_rd", "white_left_truncated", "black_left_truncated", "white_prior_games", "black_prior_games"), on="game_id")
    .filter(pl.col("perf") == pl.col("speed"))
)
# 軌跡要含邀戰與無 source 的對局，因為它們同樣更新等級分；只有評估的對局限 arena 與大廳
g = game_source.with_source(g).with_columns(keep=game_source.KEEP.fill_null(False))
# 玩家視角：每局兩列
side = pl.concat([
    g.select(
        "game_id", "start", "speed", "keep", p=f"{c}_id", o=f"{o}_id", elo=f"{c}_elo", opp_elo=f"{o}_elo", rd=f"{c}_rd", opp_rd=f"{o}_rd",
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
    # 全部 K 與重抽的配適一次送進行程池；工作量隨 K 增加，大的先送，最後一波才不會只剩少數行程
    jobs = [(Z, K, [7, k_tc, K, rep], rep, STARTS) for K in reversed(KS) for rep in range(B_STAB + 1)]
    res = {}
    for n, (K, rep, fit, lb) in enumerate(pool.imap_unordered(task, jobs), 1):
        res[K, rep] = (fit, lb)
        print(tc, n, "of", len(jobs), flush=True)
    fits = {K: res[K, 0][0] for K in KS}
    for K in KS:
        lab = res[K, 0][1]
        aris, jac = [], []
        for rep in range(1, B_STAB + 1):
            lb = res[K, rep][1]
            aris.append(ari(lab, lb))
            # Hennig 的群內 Jaccard：每個原群取與重抽各群的最大 Jaccard
            jac.append([max(((lab == j) & (lb == h)).sum() / ((lab == j) | (lb == h)).sum() for h in range(K)) for j in range(K)])
        jac = np.array(jac).mean(0)
        stab_rows.append({"speed": tc, "K": K, "n_players": len(Z), "ll": fits[K]["ll"], "converged": fits[K]["converged"], "iters": fits[K]["iters"], "min_weight": fits[K]["min_weight"], "ari_mean": np.mean(aris), "ari_p10": np.quantile(aris, 0.1), "jaccard_min": jac.min(), "jaccard_by_cluster": ",".join(f"{v:.3f}" for v in jac)})
        print(stab_rows[-1], flush=True)
    st = pl.DataFrame(stab_rows).filter(pl.col("speed") == tc)
    K = st.sort(["ari_mean", "K"], descending=[True, False])["K"][0]
    lab = res[K, 0][1]
    # 群的編號依漂移的平均排序，讓各速度類別的圖例可對照
    order = np.argsort([f["drift"].to_numpy()[lab == j].mean() for j in range(K)])
    lab = np.argsort(order)[lab]
    f = f.with_columns(cluster=pl.Series(lab))
    labels_all.append(f.select("speed", "p", "cluster"))
    jac_k = np.array(st.filter(pl.col("K") == K)["jaccard_by_cluster"][0].split(","), dtype=float)[order]
    clus_rows.append(f.group_by("speed", "cluster").agg(n=pl.len(), *[pl.col(c).mean().alias(c + "_mean") for c in FEATS], *[pl.col(c).std().alias(c + "_sd") for c in FEATS], rd10_med=pl.col("rd10").median()).sort("cluster").with_columns(K=pl.lit(K), jaccard=pl.Series(jac_k), exists=pl.Series(jac_k >= 0.5)))
    print(clus_rows[-1])

labels = pl.concat(labels_all)
clus = pl.concat(clus_rows)
stab = pl.DataFrame(stab_rows)
stab.write_csv("outputs/tables/novice_clusters_stability.csv")
clus.write_csv("outputs/tables/novice_clusters_summary.csv")

# 評估：第 11 局以後（此前局數 >= 10）、對手已非新手且非左截斷的對局，依群與此前局數的分段估校準截距與斜率
ev = (
    side.filter(pl.col("keep"), pl.col("k") >= N_FEAT, pl.col("k") < K_BINS[-1], pl.col("opp_rd") < novice.RD, ~pl.col("opp_lt"))
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
        # 各群自己的截距與斜率對「各群截距、共同斜率」。同一局可能以雙方各一列進來，dyadic 的 meat 一併涵蓋
        X1 = np.column_stack([D, D * L[:, None]])
        t1, p1, H1, ll1 = lk.frac_newton(X1, s, [0] * K + [1] * K)
        _, _, _, ll0 = lk.frac_newton(np.column_stack([D, L]), s, [0] * K + [1])
        Hi = np.linalg.inv(H1)
        V = Hi @ lk.dyad_meat(X1 * (s - p1)[:, None], m["p"].to_numpy(), m["o"].to_numpy()) @ Hi
        # 斜率相同等價於 K - 1 個斜率差為 0
        R = np.zeros((K - 1, 2 * K))
        R[np.arange(K - 1), K + np.arange(1, K)] = 1
        R[:, K] = -1
        for j in range(K):
            cal_rows.append({"speed": tc, "kbin": kb, "cluster": j, "n_games": int((c == j).sum()), "n_players": m.filter(pl.col("cluster") == j)["p"].n_unique(), "rd_med": float(np.median(m["rd"].to_numpy()[c == j])), "intercept": t1[j], "intercept_se": np.sqrt(V[j, j]), "slope": t1[K + j], "slope_se": np.sqrt(V[K + j, K + j]), "slope_se_iid": np.sqrt(Hi[K + j, K + j])})
        test_rows.append({"speed": tc, "kbin": kb, "n_games": m.height} | lk.adjusted_lrt(2 * (ll1 - ll0), lk.lr_weights_linear(Hi, V, R)))
cal = pl.DataFrame(cal_rows)
tests = pl.DataFrame(test_rows)
tests = tests.with_columns(p_holm=pl.Series(lk.holm(tests["p"])))
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
        ax.errorbar(cj["x"] * (1 + 0.03 * (j - 1)), cj["slope"], yerr=1.96 * cj["slope_se"], color=C[j], marker=MARK[j], ms=5, lw=1.5, capsize=2, label=f"cluster {j + 1} (mean drift {dr:+.0f})")
    ax.axhline(1, color=MUTE, lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_title(tc, color=INK)
    ax.set_xlabel("player's prior games (bin midpoint, log scale)")
    ax.grid(axis="y", color="#e6e6e3", lw=0.6)
    ax.legend(fontsize=7, frameon=False)
axes[0].set_ylabel("calibration slope of Elo E (player view)")
fig.suptitle("Calibration slope after the first 10 games, by early rating-trajectory cluster (dyadic 95% CI)")
fig.tight_layout()
fig.savefig("outputs/figures/novice_clusters_calibration.png", dpi=150)
