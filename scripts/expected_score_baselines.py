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

import likelihood as lk

UTC = "UTC"
CLASSES = ["Bullet", "Blitz", "Classical"]
Q = np.log(10) / 400
N_FIT = 1_000_000
RD_NOVICE = 110

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
    )
    .filter(pl.col("y").is_not_null())
)
rp = pl.scan_parquet("data/processed/glicko_replay.parquet").select("game_id", "perf", "white_rd", "black_rd", "white_left_truncated", "black_left_truncated")
d = (
    g_.join(rp, on="game_id")
    .filter(pl.col("perf") == pl.col("speed"))
    .with_columns(
        diff=(pl.col("white_elo") - pl.col("black_elo")).cast(pl.Float64),
        sigma=(pl.col("white_rd").cast(pl.Float64) ** 2 + pl.col("black_rd").cast(pl.Float64) ** 2).sqrt(),
        rd_max=pl.max_horizontal("white_rd", "black_rd"),
        lt_any=pl.col("white_left_truncated") | pl.col("black_left_truncated"),
    )
    .with_columns(gs=1 / (1 + 3 * Q**2 * pl.col("sigma") ** 2 / np.pi**2).sqrt())
    .with_columns(
        xr=((pl.col("white_elo") + pl.col("black_elo")) / 2 - 1500) / 100,
        xd=pl.col("diff") / 100,
        gxd=pl.col("gs") * pl.col("diff") / 100,
        score=pl.col("y") / 2,
        L_elo=Q * pl.col("diff"),
        L_glicko=Q * pl.col("gs") * pl.col("diff"),
        novice=pl.col("rd_max") >= RD_NOVICE,
        train=pl.col("start") < pl.datetime(2014, 7, 1, time_zone=UTC),
    )
    .select("speed", "w", "b", "y", "score", "xr", "xd", "gxd", "L_elo", "L_glicko", "rd_max", "novice", "lt_any", "train")
    .collect(engine="streaming")
)
print(d.group_by("speed").agg(n=pl.len(), novice=pl.col("novice").mean(), lt_any=pl.col("lt_any").mean()).sort("speed"))


# 與 calibration.py 的有序模型相同，另加 t8 g(sigma) x_d。t8 = 0 即有序 logistic；
# 只留 t0、t5、t8 即 Glicko 形式：分差依雙方合成的 RD 往 0 收縮，切點自由估計
def ll_ord(t, xr, xd, gxd, y):
    eta = t[0] + t[1] * xr + t[2] * xr**2 + (t[3] + t[4] * xr) * xd + t[8] * gxd
    w = jnp.exp(t[5] + t[6] * xr + t[7] * xr**2)
    u, v = eta + w, eta - w
    ldraw = jax.nn.log_sigmoid(u) + jax.nn.log_sigmoid(-v) + jnp.log(-jnp.expm1(-2 * w))
    return jnp.where(y == 2, jax.nn.log_sigmoid(v), jnp.where(y == 0, jax.nn.log_sigmoid(-u), ldraw))


def ll_frac(t, L, s):
    z = t[0] + t[1] * L
    return s * jax.nn.log_sigmoid(z) + (1 - s) * jax.nn.log_sigmoid(-z)


def probs(t, xr, xd, gxd):
    eta = t[0] + t[1] * xr + t[2] * xr**2 + (t[3] + t[4] * xr) * xd + t[8] * gxd
    w = np.exp(t[5] + t[6] * xr + t[7] * xr**2)
    pw, pn = 1 / (1 + np.exp(-(eta - w))), 1 / (1 + np.exp(-(eta + w)))
    return np.stack([1 - pn, pn - pw, pw], 1)


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

coef_rows, test_rows, cox_rows, pred_rows = [], [], [], []
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
        for grp, keep in [("all", pl.lit(True)), ("novice", pl.col("novice")), ("established", ~pl.col("novice"))]:
            m = dt.filter(keep)
            th, _, V, *_ = lk.fit(ll_frac, [0.0, 1.0], np.arange(2), (m[nm].to_numpy(), m["score"].to_numpy()), m["w"].to_numpy(), m["b"].to_numpy())
            cox_rows.append({"speed": tc, "expected_score": nm.removeprefix("L_"), "group": grp, "n": m.height, "intercept": th[0], "intercept_se": np.sqrt(V[0, 0]), "slope": th[1], "slope_se": np.sqrt(V[1, 1])})

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
        msk = np.ones(len(yte), bool) if grp == "all" else (te["novice"] == (grp == "novice")).to_numpy()
        wg, bg = te["w"].to_numpy()[msk], te["b"].to_numpy()[msk]
        for nm in MODELS:
            for metric in ["log_loss", "rps"]:
                x = loss[nm][metric][msk]
                dlt = x - loss["ordinal"][metric][msk]
                se = np.sqrt(lk.dyad_meat((dlt - dlt.mean())[:, None], wg, bg)[0, 0]) / len(dlt)
                pred_rows.append({"speed": tc, "group": grp, "model": nm, "metric": metric, "n": int(msk.sum()), "mean": x.mean(), "diff_vs_ordinal": dlt.mean(), "diff_se_dyadic": se})
    print(tc, "prediction done")

coef = pl.DataFrame(coef_rows)
coef.write_csv("outputs/tables/expected_score_baselines_coef.csv")
tests = pl.DataFrame(test_rows)
# Holm 法對全部修正的概似比檢定控制族錯誤率
pv = tests["p"].to_numpy()
order = np.argsort(pv)
adj = np.empty(len(pv))
adj[order] = np.minimum(np.maximum.accumulate(pv[order] * (len(pv) - np.arange(len(pv)))), 1)
tests = tests.with_columns(p_holm=adj)
tests.write_csv("outputs/tables/expected_score_baselines_tests.csv")
cox = pl.DataFrame(cox_rows)
cox.write_csv("outputs/tables/expected_score_baselines_cox.csv")
pred = pl.DataFrame(pred_rows)
pred.write_csv("outputs/tables/expected_score_baselines_prediction.csv")
print(coef.filter(pl.col("sample") == "all"))
print(tests)
print(cox)
print(pred)
