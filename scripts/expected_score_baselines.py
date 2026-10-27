# 期望勝率的三個基準：有序 logistic、Glicko 形式、加入 rating deviation 的有序 logistic。
# 三者巢狀於加入 RD 的模型，以 dyadic 相依修正的概似比檢定比較；另以 2014 年下半年作樣本外評估。
# 樣本為 2014 年 Bullet、Blitz、Classical 的有鐘對局，含新手，RD 取自重放。

import os
import sys

# 先於 import jax 設定：float64 在這張 GPU 上很慢
os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, "scripts")

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
from scipy import optimize, stats

import likelihood as lk
import novice
import ordinal

UTC = "UTC"
CLASSES = ["Bullet", "Blitz", "Classical"]
Q = ordinal.Q
N_FIT = 1_000_000

g_ = (
    pl.scan_parquet("data/processed/lichess_games.parquet")
    .filter(
        pl.col("speed").is_in(CLASSES), pl.col("base_seconds").is_not_null(),
        pl.col("start") >= pl.datetime(2014, 1, 1, time_zone=UTC), pl.col("start") < pl.datetime(2015, 1, 1, time_zone=UTC),
        pl.col("white_elo").is_not_null(), pl.col("black_elo").is_not_null(),
    )
    .select(
        "game_id", "start", "speed", "white_elo", "black_elo",
        y=pl.col("result").replace_strict({"0-1": 0, "1/2-1/2": 1, "1-0": 2}, default=None, return_dtype=pl.Int8),
        w="white_id", b="black_id",
        arena=pl.col("tournament_id").is_not_null(),
    )
    .filter(pl.col("y").is_not_null())
)
rp = pl.scan_parquet("data/processed/glicko_replay.parquet").select("game_id", "perf", "white_rd", "black_rd", "white_left_truncated", "black_left_truncated")
d = (
    g_.join(rp, on="game_id")
    .filter(pl.col("perf") == pl.col("speed"))
    .with_columns(
        diff=(pl.col("white_elo") - pl.col("black_elo")).cast(pl.Float64),
        gs=ordinal.g("white_rd", "black_rd"),
        rd_max=pl.max_horizontal("white_rd", "black_rd"),
        lt_any=pl.col("white_left_truncated") | pl.col("black_left_truncated"),
    )
    .with_columns(
        xr=((pl.col("white_elo") + pl.col("black_elo")) / 2 - 1500) / 100,
        xd=pl.col("diff") / 100,
        gxd=pl.col("gs") * pl.col("diff") / 100,
        score=pl.col("y") / 2,
        L_elo=Q * pl.col("diff"),
        L_glicko=Q * pl.col("gs") * pl.col("diff"),
        novice_rd=novice.novice_rd("white_rd", "black_rd"),
        train=pl.col("start") < pl.datetime(2014, 7, 1, time_zone=UTC),
        # 分層與 calibration.py 相同；重算前後以 8 月 1 日為界
        source=pl.when(pl.col("arena")).then(pl.lit("arena")).otherwise(pl.lit("lobby")),
        period=pl.when(pl.col("start") < pl.datetime(2014, 8, 1, time_zone=UTC)).then(pl.lit("Jan-Jul")).otherwise(pl.lit("Aug-Dec")),
        band=((pl.col("white_elo") + pl.col("black_elo")) / 2).cut([1200, 1400, 1600, 1800, 2000], labels=["<1200", "1200-1400", "1400-1600", "1600-1800", "1800-2000", ">=2000"]).cast(pl.String),
    )
    .select("speed", "w", "b", "y", "score", "xr", "xd", "gxd", "L_elo", "L_glicko", "rd_max", "novice_rd", "lt_any", "train", "source", "period", "band")
    .collect(engine="streaming")
)
print(d.group_by("speed").agg(n=pl.len(), novice_rd=pl.col("novice_rd").mean(), lt_any=pl.col("lt_any").mean()).sort("speed"))


# 與 calibration.py 的有序模型相同，另加 t8 g(sigma) x_d。t8 = 0 即有序 logistic；
# 只留 t0、t5、t8 即 Glicko 形式：分差依雙方合成的 RD 往 0 收縮，切點自由估計
def ll_ord(t, xr, xd, gxd, y):
    eta = t[0] + t[1] * xr + t[2] * xr**2 + (t[3] + t[4] * xr) * xd + t[8] * gxd
    return ordinal.ll(eta, jnp.exp(t[5] + t[6] * xr + t[7] * xr**2), y)


def ll_frac(t, L, s):
    z = t[0] + t[1] * L
    return s * jax.nn.log_sigmoid(z) + (1 - s) * jax.nn.log_sigmoid(-z)


def probs(t, xr, xd, gxd):
    eta = t[0] + t[1] * xr + t[2] * xr**2 + (t[3] + t[4] * xr) * xd + t[8] * gxd
    return ordinal.probs(eta, np.exp(t[5] + t[6] * xr + t[7] * xr**2))


PARAMS = ["white_adv", "white_adv_x_rating", "white_adv_x_rating2", "slope_diff", "slope_diff_x_rating", "log_draw_width", "log_draw_width_x_rating", "log_draw_width_x_rating2", "slope_g_diff"]
T0 = np.array([0.1, 0, 0, 0.6, 0, -2.5, 0, 0, 0])
MODELS = {
    "rd_ordinal": np.arange(9),
    "ordinal": np.arange(8),
    "glicko": np.array([0, 5, 8]),
}
# Glicko 形式的受限參數必須是 0，不是 T0 的起始值。t8 從純 Glicko 的斜率起算：
# 分差每 100 分的累積 logit 約為 100 q 的兩倍（有和局時累積 logit 的斜率較陡）
T0_GLICKO = np.zeros(9)
T0_GLICKO[[0, 5, 8]] = [0.1, -2.5, 2 * 100 * Q]

coef_rows, test_rows, cox_rows, pred_rows, curve_parts, strata_rows, score_rows = [], [], [], [], [], [], []
for tc in CLASSES:
    dt = d.filter(pl.col("speed") == tc)
    for variant, keep in [("all", pl.lit(True)), ("no_left_truncated", ~pl.col("lt_any"))]:
        m = dt.filter(keep).sample(N_FIT, seed=1)
        w, b = m["w"].to_numpy(), m["b"].to_numpy()
        args = tuple(m[c].to_numpy().astype(np.float64) for c in ["xr", "xd", "gxd", "y"])
        res = {}
        for nm, free in MODELS.items():
            res[nm] = lk.fit(ll_ord, T0_GLICKO if nm == "glicko" else T0, free, args, w, b)
            th, llk, V, Hi, *_ = res[nm]
            for k, j in enumerate(free):
                coef_rows.append({"speed": tc, "sample": variant, "model": nm, "param": PARAMS[j], "est": th[j], "se_dyadic": np.sqrt(V[k, k]), "se_iid": np.sqrt(Hi[k, k]), "loglik": llk, "n": m.height})
        th1, ll1, V1, Hi1, *_ = res["rd_ordinal"]
        for nm, idx in [("ordinal", [8]), ("glicko", [1, 2, 3, 4, 6, 7])]:
            r = lk.adjusted_lrt(2 * (ll1 - res[nm][1]), lk.lr_weights(Hi1, V1, idx))
            test_rows.append({"speed": tc, "sample": variant, "restricted": nm, "full": "rd_ordinal", **r})
        print(tc, variant, "done")

    # 平均得分的校準：Elo 與 Glicko 的期望得分本身是否校準（Cox 截距 0、斜率 1），依新手與否分層
    for nm in ["L_elo", "L_glicko"]:
        for grp, keep in [("all", pl.lit(True)), ("novice", pl.col("novice_rd")), ("established", ~pl.col("novice_rd"))]:
            m = dt.filter(keep)
            th, _, V, Hi, *_ = lk.fit(ll_frac, [0.0, 1.0], np.arange(2), (m[nm].to_numpy(), m["score"].to_numpy()), m["w"].to_numpy(), m["b"].to_numpy())
            cox_rows.append({"speed": tc, "expected_score": nm.removeprefix("L_"), "group": grp, "n": m.height, "intercept": th[0], "intercept_se": np.sqrt(V[0, 0]), "intercept_se_iid": np.sqrt(Hi[0, 0]), "slope": th[1], "slope_se": np.sqrt(V[1, 1]), "slope_se_iid": np.sqrt(Hi[1, 1])})

    # 老手對局的 Elo 期望得分：score test 檢定 alpha = 0、beta = 1 與 Elo 本身（兩者皆是），區間為反轉 score test。
    # 只需在虛無假設下計算，Elo 本身不用估計。各局當成獨立，quasi-likelihood 取 phi = 1；和局使 phi < 1，統計量偏小、區間偏寬
    m = dt.filter(~pl.col("novice_rd"))
    L, S = m["L_elo"].to_numpy(), m["score"].to_numpy()
    X = np.column_stack([np.ones_like(L), L])

    def score_stat(t):
        p = 1 / (1 + np.exp(-(t[0] + t[1] * L)))
        U = X.T @ (S - p)
        return U @ np.linalg.solve(X.T @ (X * (p * (1 - p))[:, None]), U)

    # 固定第 j 個參數為 v，另一個以牛頓法解虛無假設下的估計，再算 score 統計量
    def score_at(j, v, t0):
        t = np.array(t0, dtype=float)
        t[j] = v
        k = 1 - j
        for _ in range(50):
            p = 1 / (1 + np.exp(-(t[0] + t[1] * L)))
            step = (X[:, k] @ (S - p)) / (X[:, k] ** 2 @ (p * (1 - p)))
            t[k] += step
            if abs(step) < 1e-12:
                break
        return score_stat(t)

    th, _, _, Hi, *_ = lk.fit(ll_frac, [0.0, 1.0], np.arange(2), (L, S), m["w"].to_numpy(), m["b"].to_numpy())
    crit = stats.chi2.ppf(0.95, 1)
    row = {"speed": tc, "n": m.height, "score_elo": score_stat([0.0, 1.0])}
    for j, nm, h0 in [(0, "alpha", 0.0), (1, "beta", 1.0)]:
        se = np.sqrt(Hi[j, j])
        f = lambda v: score_at(j, v, th) - crit
        row |= {nm: th[j], f"{nm}_lo": optimize.brentq(f, th[j] - 4 * se, th[j]), f"{nm}_hi": optimize.brentq(f, th[j], th[j] + 4 * se), f"score_{nm}_h0": score_at(j, h0, th)}
    score_rows.append(row)
    print(row)

    # 老手對局依兩種期望得分分組的平均殘差，分組與 calibration.py 的校準曲線相同。
    # 標準誤另列 dyadic：同一位玩家的多局、同一局的兩位玩家都相關
    for nm in ["L_elo", "L_glicko"]:
        m = (
            dt.filter(~pl.col("novice_rd"))
            .with_columns(E=1 / (1 + (-pl.col(nm)).exp()))
            .with_columns(e_bin=(pl.col("E") * 20).floor().clip(0, 19) / 20 + 0.025, r=pl.col("score") - pl.col("E"))
        )
        for c in m.partition_by("e_bin"):
            r = c["r"].to_numpy()
            curve_parts.append({
                "e_bin": c["e_bin"][0], "n": len(r), "e_mean": c["E"].mean(), "resid_mean": r.mean(),
                "resid_se_iid": r.std(ddof=1) / np.sqrt(len(r)),
                "resid_se_dyadic": lk.dyad_se((r - r.mean()) / len(r), c["w"].to_numpy(), c["b"].to_numpy()),
                "speed": tc, "expected_score": nm.removeprefix("L_"),
            })

    # 老手對局的 Elo 期望得分依分層估截距與斜率，看偏差是否只出在某些子樣本
    for k in ["band", "source", "period"]:
        for v in dt[k].unique().sort().to_list():
            m = dt.filter(~pl.col("novice_rd"), pl.col(k) == v)
            th, _, V, Hi, *_ = lk.fit(ll_frac, [0.0, 1.0], np.arange(2), (m["L_elo"].to_numpy(), m["score"].to_numpy()), m["w"].to_numpy(), m["b"].to_numpy())
            strata_rows.append({"speed": tc, "stratum_type": k, "stratum": v, "n": m.height, "intercept": th[0], "slope": th[1], "slope_se": np.sqrt(V[1, 1]), "slope_se_iid": np.sqrt(Hi[1, 1])})

    # 樣本外評估：1-6 月估計，7-12 月評估
    tr = dt.filter(pl.col("train")).sample(N_FIT, seed=2)
    te = dt.filter(~pl.col("train"))
    a = tuple(tr[c].to_numpy().astype(np.float64) for c in ["xr", "xd", "gxd", "y"])
    xte = [te[c].to_numpy() for c in ["xr", "xd", "gxd"]]
    yte = te["y"].to_numpy()
    O = np.eye(3)[yte]
    loss = {}
    for nm, free in MODELS.items():
        th, *_ = lk.fit(ll_ord, T0_GLICKO if nm == "glicko" else T0, free, a, tr["w"].to_numpy(), tr["b"].to_numpy())
        p = probs(th, *xte)
        C, Co = np.cumsum(p, 1)[:, :2], np.cumsum(O, 1)[:, :2]
        loss[nm] = {"log_loss": -np.log(p[np.arange(len(yte)), yte]), "rps": ((C - Co) ** 2).sum(1) / 2}
    for grp in ["all", "novice", "established"]:
        msk = np.ones(len(yte), bool) if grp == "all" else (te["novice_rd"] == (grp == "novice")).to_numpy()
        wg, bg = te["w"].to_numpy()[msk], te["b"].to_numpy()[msk]
        for nm in MODELS:
            for metric in ["log_loss", "rps"]:
                x = loss[nm][metric][msk]
                dlt = x - loss["ordinal"][metric][msk]
                se = lk.dyad_se((dlt - dlt.mean()) / len(dlt), wg, bg)
                pred_rows.append({"speed": tc, "group": grp, "model": nm, "metric": metric, "n": int(msk.sum()), "mean": x.mean(), "diff_vs_ordinal": dlt.mean(), "diff_se_dyadic": se})
    print(tc, "prediction done")

coef = pl.DataFrame(coef_rows)
coef.write_csv("outputs/tables/expected_score_baselines_coef.csv")
tests = pl.DataFrame(test_rows)
# Holm 法對全部修正的概似比檢定控制族錯誤率
tests = tests.with_columns(p_holm=lk.holm(tests["p"]))
tests.write_csv("outputs/tables/expected_score_baselines_tests.csv")
cox = pl.DataFrame(cox_rows)
cox.write_csv("outputs/tables/expected_score_baselines_cox.csv")
pl.DataFrame(curve_parts).sort("speed", "expected_score", "e_bin").write_csv("outputs/tables/expected_score_baselines_curve.csv")
pl.DataFrame(strata_rows).write_csv("outputs/tables/expected_score_baselines_strata.csv")
pl.DataFrame(score_rows).write_csv("outputs/tables/expected_score_baselines_calibration_score.csv")
pred = pl.DataFrame(pred_rows)
pred.write_csv("outputs/tables/expected_score_baselines_prediction.csv")
print(coef.filter(pl.col("sample") == "all"))
print(tests)
print(cox)
print(pred)
