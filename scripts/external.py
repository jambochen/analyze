# 其他遊戲的外部驗證共用的估計：主基準的重估、系統與主基準的校準、依狀態調整的勝率的逐步加入、停止規則的判定。
# other_games_ogs.py 與 other_games_aoe2.py 共用；兩個遊戲若各寫一份，「沿用同一套方法」就無從保證。
#
# t 每局一列，取 A 方的視角：a_id、b_id、speed、train（估計段）、y（A 勝）、p_sys（系統給 A 的勝率）、novice，
# base 的各欄（主基準的線性預測式），與近況之差 d_*（A 減 B）。沒有和局，有序模型退化為 binary logistic。

import numpy as np
import polars as pl

import likelihood as lk
from state_adjusted import diff_row, losses
from stop_rules import apply_rules, backtest

STATE = [("prev_result", ["d_prev_win", "d_missing"]), ("dr", ["d_dr"])]
DELTA = 10


def _loss(p, y):
    return losses(np.c_[1 - p, np.zeros(len(p)), p], (y * 2).astype(int))


def evaluate(t, speeds, base, slope):
    """回傳 (tables, te)。tables 含 calibration、steps、prediction、coef；te 為評估段各局加上 p_base、p_adj。
    slope(th, d) 回傳主基準線性預測式對 A 方等級分（每 100 分）的斜率，用來把得分換算成等級分。"""
    cal_rows, step_rows, pred_rows, coef_rows, parts = [], [], [], [], []
    for spd in speeds:
        ts = t.filter(pl.col("speed") == spd)
        tr, te = ts.filter("train"), ts.filter(~pl.col("train"))
        wtr, btr, wte, bte = tr["a_id"].to_numpy(), tr["b_id"].to_numpy(), te["a_id"].to_numpy(), te["b_id"].to_numpy()
        th, *_ = lk.fit(lk.logit_ll, np.zeros(len(base)), np.arange(len(base)), (tr.select(base).to_numpy().astype(np.float64), tr["y"].to_numpy()), wtr, btr)
        eta = lambda d: d.select(base).to_numpy().astype(np.float64) @ th  # noqa: E731
        tr, te = tr.with_columns(L_base=pl.Series(eta(tr))), te.with_columns(L_base=pl.Series(eta(te)))
        te = te.with_columns(L_sys=(pl.col("p_sys") / (1 - pl.col("p_sys"))).log(), const=pl.lit(1.0))

        # 系統與主基準的校準：logit E[y] = alpha + beta L，評估段；完全校準為 alpha = 0、beta = 1
        for nm in ["L_sys", "L_base"]:
            X = te.select("const", nm).to_numpy().astype(np.float64)
            c, _, V, *_ = lk.fit(lk.logit_ll, np.array([0.0, 1.0]), np.arange(2), (X, te["y"].to_numpy()), wte, bte)
            cal_rows.append({"speed": spd, "expected": nm, "n": te.height, "alpha": c[0], "alpha_se": np.sqrt(V[0, 0]), "beta": c[1], "beta_se": np.sqrt(V[1, 1])})

        # 依狀態調整：主基準為 offset（係數固定為 1），各項依序加入
        def run(cols):
            X = ["L_base"] + cols
            t0 = np.r_[1.0, np.zeros(len(cols))]
            c, *_ = lk.fit(lk.logit_ll, t0, np.arange(1, len(X)), (tr.select(X).to_numpy().astype(np.float64), tr["y"].to_numpy()), wtr, btr) if cols else (t0,)
            p = 1 / (1 + np.exp(-(te.select(X).to_numpy().astype(np.float64) @ c)))
            return c, _loss(p, te["y"].to_numpy()), p

        _, loss_base, _ = run([])
        loss_sys = _loss(te["p_sys"].to_numpy(), te["y"].to_numpy())
        kept, cur, loss_cur = [], "baseline", loss_base
        for nm, cols in STATE:
            c, ls, _ = run(kept + cols)
            r = diff_row(spd, nm, cur, ls, loss_cur, np.ones(te.height, bool), wte, bte, "all")[0]
            keep = r["diff"] + 1.96 * r["diff_se_dyadic"] < 0
            step_rows.append({"speed": spd, "step": nm, "vs": cur, "diff": r["diff"], "diff_se_dyadic": r["diff_se_dyadic"], "kept": keep})
            if keep:
                kept, cur, loss_cur = kept + cols, nm, ls
        c, loss_adj, p_adj = run(kept)
        for k, col in enumerate(kept):
            coef_rows.append({"speed": spd, "param": col, "est": c[1 + k]})
        nov = te["novice"].to_numpy()
        for group, msk in [("all", np.ones(te.height, bool)), ("novice", nov), ("established", ~nov)]:
            pred_rows += diff_row(spd, "baseline", "system", loss_base, loss_sys, msk, wte, bte, group)
            pred_rows += diff_row(spd, "adjusted", "baseline", loss_adj, loss_base, msk, wte, bte, group)
        te = te.with_columns(p_adj=pl.Series(p_adj), p_base=1 / (1 + (-pl.col("L_base")).exp()))
        # 主基準下期望得分對自身等級分的斜率（每 1 分）
        spe = te.select(s=pl.col("p_base") * (1 - pl.col("p_base")) * slope(th, te) / 100)["s"].mean()
        coef_rows.append({"speed": spd, "param": "score_per_rating", "est": spe})
        parts.append(te)
        print(spd, "done", kept, flush=True)
    tables = {"calibration": pl.DataFrame(cal_rows), "steps": pl.DataFrame(step_rows), "prediction": pl.DataFrame(pred_rows), "coef": pl.DataFrame(coef_rows)}
    return tables, pl.concat(parts)


def player_rows(te):
    """評估段每局兩列（雙方視角）：game_id、speed、novice、player、opponent、score 與三個預報 e_sys、e_base、e_adj。"""
    return pl.concat([
        te.select("game_id", "speed", "novice", player="a_id", opponent="b_id", score="y", e_sys="p_sys", e_base="p_base", e_adj="p_adj"),
        te.select("game_id", "speed", "novice", player="b_id", opponent="a_id", score=1 - pl.col("y"), e_sys=1 - pl.col("p_sys"), e_base=1 - pl.col("p_base"), e_adj=1 - pl.col("p_adj")),
    ])


def stop_verdicts(d, keep, speeds, coef):
    """d 為玩家局（player、day、day_start、result、resid、speed、opponent、novice_rd，依玩家與時間排序），keep 為評估段的條件。
    回傳 (res, reset)：回測與 Lichess 相同，另加換算成等級分的差與三類判定。"""
    d, del_days = apply_rules(d, keep)
    res, reset, *_ = backtest(d, del_days, 2000, speeds=speeds, family=speeds)
    spe = dict(coef.filter(pl.col("param") == "score_per_rating").select("speed", "est").iter_rows())
    res = res.with_columns(spe=pl.col("speed").replace_strict(spe, return_dtype=pl.Float64)).with_columns(
        diff_rating=pl.col("diff") / pl.col("spe"), lo_rating=pl.col("lo") / pl.col("spe"), hi_rating=pl.col("hi") / pl.col("spe"),
    ).with_columns(
        verdict=pl.when((pl.col("diff") < 0) & (pl.col("p_holm") < 0.05)).then(pl.lit("beats_never_stop"))
        .when((pl.col("lo_rating") > -DELTA) & (pl.col("hi_rating") < DELTA)).then(pl.lit("negligible"))
        .otherwise(pl.lit("inconclusive"))
    )
    return res, reset
