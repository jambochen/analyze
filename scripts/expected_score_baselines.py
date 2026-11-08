# 期望勝率的三個基準：有序 logistic、Glicko 形式、加入 rating deviation 的有序 logistic。
# 三者巢狀於加入 RD 的模型，以 dyadic 相依修正的概似比檢定比較；另以 2014 年下半年作樣本外評估。
# 樣本為 2014 年 Bullet、Blitz、Classical 的有鐘對局，含新手，RD 取自重放。

import os
import sys

# 先於 import jax 設定：float64 在這張 GPU 上很慢
os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, "scripts")

import jax
import numpy as np
import polars as pl

import likelihood as lk
import ordinal
from baseline_sample import ll_ord, probs, sample as baseline_sample

UTC = "UTC"
CLASSES = ["Bullet", "Blitz", "Classical"]
Q = ordinal.Q
EXP = ["0", "1-2", "3-5", "6-10", "11-20", "21-50", "51-100", "101-200", ">200"]

d = baseline_sample()
print(d.group_by("speed").agg(n=pl.len(), novice_rd=pl.col("novice_rd").mean(), lt_any=pl.col("lt_any").mean()).sort("speed"))


# 與 calibration.py 的有序模型相同，另加 t8 g(sigma) x_d。t8 = 0 即有序 logistic；
# 只留 t0、t5、t8 即 Glicko 形式：分差依雙方合成的 RD 往 0 收縮，切點自由估計
def ll_frac(t, L, s):
    z = t[0] + t[1] * L
    return s * jax.nn.log_sigmoid(z) + (1 - s) * jax.nn.log_sigmoid(-z)


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

coef_rows, test_rows, cox_rows, pred_rows, curve_parts, strata_rows, score_rows, hold_rows, hold_cox_rows = [], [], [], [], [], [], [], [], []
for tc in CLASSES:
    dt = d.filter(pl.col("speed") == tc)
    for variant, keep in [("all", pl.lit(True)), ("no_left_truncated", ~pl.col("lt_any"))]:
        m = dt.filter(keep)
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
    # 只需在虛無假設下計算，Elo 本身不用估計
    m = dt.filter(~pl.col("novice_rd"))
    L, S = m["L_elo"].to_numpy(), m["score"].to_numpy()
    th, _, _, Hi, *_ = lk.fit(ll_frac, [0.0, 1.0], np.arange(2), (L, S), m["w"].to_numpy(), m["b"].to_numpy())
    row = {"speed": tc, "n": m.height, "score_elo": lk.frac_score(L, S, [0.0, 1.0])}
    for j, nm, h0 in [(0, "alpha", 0.0), (1, "beta", 1.0)]:
        lo, hi = lk.frac_score_ci(L, S, j, th, np.sqrt(Hi[j, j]))
        row |= {nm: th[j], f"{nm}_lo": lo, f"{nm}_hi": hi, f"score_{nm}_h0": lk.frac_score_at(L, S, j, h0, th)}
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
    tr = dt.filter(pl.col("train"))
    te = dt.filter(~pl.col("train"))
    a = tuple(tr[c].to_numpy().astype(np.float64) for c in ["xr", "xd", "gxd", "y"])
    xte = [te[c].to_numpy() for c in ["xr", "xd", "gxd"]]
    yte = te["y"].to_numpy()
    O = np.eye(3)[yte]
    loss, P = {}, {}
    for nm, free in MODELS.items():
        th, *_ = lk.fit(ll_ord, T0_GLICKO if nm == "glicko" else T0, free, a, tr["w"].to_numpy(), tr["b"].to_numpy())
        p = probs(th, *xte)
        P[nm] = p
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

    # 選出的模型在 7-12 月是否校準，對照 Elo：得分殘差依期望得分分組，Cox 斜率依累計局數分組。
    # 區間與本次其餘推論相同，各局當成獨立
    s_te = te["score"].to_numpy()
    eb_te = te["exp_bin"].to_numpy()
    for nm, E in [("elo", 1 / (1 + np.exp(-te["L_elo"].to_numpy()))), ("rd_ordinal", P["rd_ordinal"][:, 2] + 0.5 * P["rd_ordinal"][:, 1])]:
        k = np.clip(np.floor(E * 20), 0, 19)
        for j in np.unique(k):
            r = (s_te - E)[k == j]
            hold_rows.append({"speed": tc, "model": nm, "kind": "score", "bin": j, "n": len(r), "pred_mean": E[k == j].mean(), "resid_mean": r.mean(), "resid_se_iid": r.std(ddof=1) / np.sqrt(len(r))})
        L = np.log(E / (1 - E))
        for v in EXP:
            msk = eb_te == v
            th, _, H, _ = lk.frac_newton(np.column_stack([np.ones(msk.sum()), L[msk]]), s_te[msk], [0.0, 1.0])
            lo, hi = lk.frac_score_ci(L[msk], s_te[msk], 1, th, np.sqrt(np.linalg.inv(H)[1, 1]))
            hold_cox_rows.append({"speed": tc, "model": nm, "exp_bin": v, "n": int(msk.sum()), "intercept": th[0], "slope": th[1], "slope_lo": lo, "slope_hi": hi})
    # 和局機率依預測值的 20 分位分組；Elo 沒有和局機率，只看選出的模型
    pd_ = P["rd_ordinal"][:, 1]
    k = np.clip(np.searchsorted(np.quantile(pd_, np.linspace(0, 1, 21)), pd_, side="right") - 1, 0, 19)
    for j in range(20):
        r = (yte == 1)[k == j] - pd_[k == j]
        hold_rows.append({"speed": tc, "model": "rd_ordinal", "kind": "draw", "bin": j, "n": len(r), "pred_mean": pd_[k == j].mean(), "resid_mean": r.mean(), "resid_se_iid": r.std(ddof=1) / np.sqrt(len(r))})
    print(tc, "prediction done")

coef = pl.DataFrame(coef_rows)
coef.write_csv("outputs/tables/expected_score_baselines_coef.csv")
# 有序 logistic 與 Glicko 形式彼此不巢狀，概似比檢定只能各自對完整模型，兩者之間以 AIC 直接比較。
# 懲罰項 2k 假設各局獨立，與本次其餘推論相同
aic = (
    coef.group_by("speed", "sample", "model")
    .agg(k=pl.len(), loglik=pl.col("loglik").first(), n=pl.col("n").first())
    .with_columns(aic=2 * pl.col("k") - 2 * pl.col("loglik"))
    .sort("speed", "sample", "aic")
)
aic.write_csv("outputs/tables/expected_score_baselines_aic.csv")
print(aic)
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
pl.DataFrame(hold_rows).write_csv("outputs/tables/expected_score_baselines_holdout_curve.csv")
hold_cox = pl.DataFrame(hold_cox_rows)
hold_cox.write_csv("outputs/tables/expected_score_baselines_holdout_cox.csv")
print(hold_cox)
print(coef.filter(pl.col("sample") == "all"))
print(tests)
print(cox)
print(pred)
