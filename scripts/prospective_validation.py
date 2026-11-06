# 前瞻驗證：依 docs/preregistration.md，在 2015 年 1 月 1 日至 4 月 19 日的 arena 上只跑一次。
# 登記文件沒有寫死的細節，在這裡寫定、commit 之後才執行：
#
# 1. Blitz 剛被分到強對手之後的總效果
#    - 統計量、虛無分布、加權與 2014 年相同：arena_crossover.py 的 beta_lag1（主基準 offset 的有序模型在 beta = 0 下的一步估計），
#      主虛無分布、以玩家加權、全部兩局以上的批。效果取置中值（實際值減虛無分布的平均），B 與種子同 2014 年
#    - 單尾檢定：方向取 2014 年置中值的正負號；p = (1 + #{重抽的置中值在該方向上不小於實際值}) / (B + 1)，水準 0.05
#    - 同質性：z = (b2014 - b2015) / sqrt(sd2014^2 + sd2015^2)，sd 為各年 permutation 虛無分布的標準差，雙尾 0.05；
#      兩年差值的 95% 區間為差值加減 1.96 倍合成的 sd
#    - 兩條都成立才算通過。Bullet 一併算出，只作描述
# 2. 依狀態調整的勝率的對數分數
#    - 模型為 state_adjusted_probs.py 以 2014 上半年估計、在 2014 下半年評估過的那一組（data/processed/state_adjusted_models），不重估
#    - Bullet 與 Blitz 各自判定：與主基準的每局 log loss 之差，dyadic 標準誤，95% 區間的上端小於 0 才算改善。Classical 只作描述
# 3. 停止規則
#    - stop_rule_sample 的 2015 年版：一天的序列從 2014 年 12 月 1 日切起，評估 2015 年 1 月 1 日之後開始的天；回測同 folk_stop_rules.py
#    - 勝過從不停：Bullet 與 Blitz 的兩條規則共四格為一族，刪去減保留的差為負，且 Holm 校正後 p < 0.05
#    - 不必停：差值的玩家 block bootstrap 95% 區間完整落在 (-10, 10) 等級分內。換算用 2014 年主基準的 score_per_elo
#      （各速度類別的平均，preregistration_mde.py 所用的同一個值）
#    - 兩者皆不成立時記為樣本不足以判斷
# 4. 事後型態，只看方向（Bullet 與 Blitz 各自）
#    - 上一局等級分變動的係數為正：rating_filter_error.py 的校準迴歸 logit E[得分] = alpha + beta L + gamma dr，gamma > 0
#    - 剛贏之後好於期望：rating_filter_sample 中上一局為勝的局，主基準殘差的平均 > 0
#    - 高 RD 剛輸之後差於期望：自身 RD 在 2014 年最高五分之一（2014 年該等分的最小 RD 以上），上一局為負，主基準殘差的平均 < 0
#    - LightGBM、XGBoost 與 MLP 在新手組的對數分數好於依狀態調整的勝率：三者各自與修正後的有序模型比，log loss 之差 < 0
#    - 每天第一局低於期望：停止規則樣本中每位玩家每天的第一局 arena 局，主基準殘差的平均 < 0

import os
import sys

# 先於 import jax 設定：float64 在這張 GPU 上很慢
os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, "scripts")

import jax.numpy as jnp
import lightgbm
import numpy as np
import polars as pl
import xgboost
from scipy.stats import norm

import likelihood as lk
import ordinal
import pairing_null
from arena_batches import player_sample_rd, rating_filter_sample, stop_rule_sample
from crossover_direct import SPEEDS, load, null_score_info, weights
from state_adjusted import GBM_FEATURES, MODEL_DIR, XCOLS, diff_row, losses, mlp_out, mlp_x, probs, sample
from stop_rules import backtest

B = 2000
DELTA = 10
MAIN = ["Bullet", "Blitz"]


# 1. 總效果。平行池 fork 之前不啟動 jax 的計算
t, m, ns = load("2015")
s_t, v_t, _ = null_score_info(t)
sp_t, pl_t = t["sp"].to_numpy(), t["player"].to_numpy()
rr_m = (m["rule"] == "random").to_numpy()
mk = t["midx_lag1"].fill_null(-1).to_numpy()
rows = np.flatnonzero(mk >= 0)
mk, g = mk[rows], sp_t[rows]
wv = weights(pl_t[rows], g, rr_m[mk])[2]
e_bar_m = m["e_bar"].to_numpy()


def beta(e):
    h = (e_bar_m - e)[mk]
    return np.bincount(g, h * s_t[rows] * wv, len(SPEEDS)) / np.bincount(g, h * h * v_t[rows] * wv, len(SPEEDS))


act = beta(ns["e_act"])
nd = np.array(pairing_null.run(ns, beta, B)["main"])
ri14 = pl.read_csv("outputs/tables/arena_crossover_ri.csv").filter(
    pl.col("null") == "main", pl.col("stat") == "beta_lag1", pl.col("weighting") == "player", pl.col("batches") == "all"
)
slope_elo = dict(t.group_by("speed").agg(pl.col("slope_elo").mean()).iter_rows())
total_rows = []
for spd in MAIN:
    j = SPEEDS.index(spd)
    r14 = ri14.filter(pl.col("speed") == spd).row(0, named=True)
    sign = np.sign(r14["est_centered"])
    x = nd[:, j] - nd[:, j].mean()
    est = act[j] - nd[:, j].mean()
    sd = nd[:, j].std()
    sd_diff = np.sqrt(r14["null_sd"] ** 2 + sd**2)
    z = (r14["est_centered"] - est) / sd_diff
    p_one = (1 + (sign * x >= sign * est).sum()) / (B + 1)
    p_hom = 2 * norm.sf(abs(z))
    total_rows.append({
        "speed": spd, "rows": int((g == j).sum()), "players": len(np.unique(pl_t[rows][g == j])),
        "est_2014": r14["est_centered"], "sd_2014": r14["null_sd"], "est_2015": est, "sd_2015": sd,
        "p_one_sided_2015": p_one, "diff": r14["est_centered"] - est, "diff_lo": r14["est_centered"] - est - 1.96 * sd_diff, "diff_hi": r14["est_centered"] - est + 1.96 * sd_diff,
        "z_homogeneity": z, "p_homogeneity": p_hom,
        "elo_per_0_1_2014": r14["elo_per_0_1"], "elo_per_0_1_2015": est * 0.1 / slope_elo[spd],
        "registered": spd == "Blitz", "passed": bool(p_one < 0.05 and p_hom >= 0.05),
    })
total = pl.DataFrame(total_rows)
with pl.Config(tbl_cols=-1, tbl_width_chars=250):
    print(total)
total.write_csv("outputs/tables/prospective_validation_total_effect.csv")
del t, m, ns


# 2. 依狀態調整的勝率，與事後型態中新手組的 ML 對照
d = sample("2015")
pred_rows, ml_rows = [], []
for spd in SPEEDS:
    te = d.filter(pl.col("speed") == spd)
    eta, w, X, y = te["eta"].to_numpy(), te["w"].to_numpy(), te.select(XCOLS).to_numpy().astype(np.float64), te["y"].to_numpy()
    wte, bte = te["player"].to_numpy(), te["opponent"].to_numpy()
    mz = np.load(f"{MODEL_DIR}/{spd}.npz")
    loss = {"baseline": losses(probs(np.zeros_like(mz["th"]), eta, w, X), y), "adjusted": losses(probs(mz["th"], eta, w, X), y)}
    Xg = te.select(GBM_FEATURES).to_numpy().astype(np.float64)
    loss["lightgbm"] = losses(lightgbm.Booster(model_file=f"{MODEL_DIR}/lightgbm_{spd}.txt").predict(Xg), y)
    xgb = xgboost.Booster()
    xgb.load_model(f"{MODEL_DIR}/xgboost_{spd}.json")
    loss["xgboost"] = losses(xgb.predict(xgboost.DMatrix(Xg)), y)
    c, k = mlp_x(te)
    e_mlp, w_mlp = mlp_out([jnp.asarray(mz[f"mlp_{i}"]) for i in range(6)], np.c_[(c - mz["mlp_mu"]) / mz["mlp_sd"], k])
    loss["mlp"] = losses(ordinal.probs(np.asarray(e_mlp, dtype=np.float64), np.asarray(w_mlp, dtype=np.float64)), y)
    r = diff_row(spd, "adjusted", "baseline", loss["adjusted"], loss["baseline"], np.ones(te.height, bool), wte, bte, "all")
    pred_rows += [x | {"registered": spd in MAIN, "improved": x["diff"] + 1.96 * x["diff_se_dyadic"] < 0} for x in r]
    nov = te["novice_rd"].to_numpy()
    for nm in ["lightgbm", "xgboost", "mlp"]:
        ml_rows += diff_row(spd, nm, "adjusted", loss[nm], loss["adjusted"], nov, wte, bte, "novice")
pred = pl.DataFrame(pred_rows)
with pl.Config(tbl_cols=-1, tbl_width_chars=250):
    print(pred)
pred.write_csv("outputs/tables/prospective_validation_log_score.csv")


# 3. 停止規則
spe = dict(player_sample_rd("2014").group_by("speed").agg(pl.col("score_per_elo").mean()).iter_rows())
sd_, del_days = stop_rule_sample(period="2015")
res, reset, est_, boot, col, rcol = backtest(sd_, del_days, B)
stop = (
    res.filter(pl.col("group") == "all", pl.col("speed").is_in(MAIN))
    .with_columns(spe=pl.col("speed").replace_strict(spe, return_dtype=pl.Float64))
    .with_columns(diff_elo=pl.col("diff") / pl.col("spe"), lo_elo=pl.col("lo") / pl.col("spe"), hi_elo=pl.col("hi") / pl.col("spe"))
    .with_columns(
        verdict=pl.when((pl.col("diff") < 0) & (pl.col("p_holm") < 0.05)).then(pl.lit("beats_never_stop"))
        .when((pl.col("lo_elo") > -DELTA) & (pl.col("hi_elo") < DELTA)).then(pl.lit("negligible"))
        .otherwise(pl.lit("inconclusive"))
    )
)
with pl.Config(tbl_cols=-1, tbl_width_chars=250):
    print(stop)
    print(reset)
stop.write_csv("outputs/tables/prospective_validation_stop_rules.csv")
res.write_csv("outputs/tables/prospective_validation_stop_rules_all.csv")
reset.write_csv("outputs/tables/prospective_validation_stop_rules_reset.csv")


# 4. 事後型態
post_rows = [{"pattern": "ml_beats_adjusted_novice", "speed": r["speed"], "detail": r["model"], "n": r["n"], "value": r["diff"], "expected_sign": -1} for r in ml_rows if r["metric"] == "log_loss"]
rf = rating_filter_sample(period="2015")
cut = pl.read_csv("outputs/tables/rating_filter_error_sample.csv").filter(pl.col("rdq") == 4).group_by("speed").agg(pl.col("rd_min").min())
cut = dict(cut.iter_rows())
# 每天的第一局 arena 局，依該局的速度類別歸類
first_day = sd_.group_by("player", "day").agg(pl.col("resid").first(), pl.col("speed").first())
for spd in MAIN:
    s = rf.filter(pl.col("speed") == spd)
    th, *_ = lk.fit(lk.logit_ll, np.array([0.0, 1.0, 0.0]), np.arange(3), (s.select("const", "L", "dr").to_numpy().astype(np.float64), s["score"].to_numpy().astype(np.float64)), s["player"].to_numpy(), s["opponent"].to_numpy())
    post_rows.append({"pattern": "dr_coefficient", "speed": spd, "detail": "gamma", "n": s.height, "value": th[2], "expected_sign": 1})
    x = s.filter(pl.col("prev_result") == "win")
    post_rows.append({"pattern": "after_win", "speed": spd, "detail": "mean_resid", "n": x.height, "value": x["resid"].mean(), "expected_sign": 1})
    x = s.filter(pl.col("prev_result") == "loss", pl.col("rd") >= cut[spd])
    post_rows.append({"pattern": "high_rd_after_loss", "speed": spd, "detail": "mean_resid", "n": x.height, "value": x["resid"].mean(), "expected_sign": -1})
    x = first_day.filter(pl.col("speed") == spd)
    post_rows.append({"pattern": "first_game_of_day", "speed": spd, "detail": "mean_resid", "n": x.height, "value": x["resid"].mean(), "expected_sign": -1})
post = pl.DataFrame(post_rows).with_columns(direction_holds=np.sign(pl.col("value")) == pl.col("expected_sign"))
with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250):
    print(post)
post.write_csv("outputs/tables/prospective_validation_posthoc.csv")
