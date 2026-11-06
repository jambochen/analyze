# OGS 圍棋的外部驗證：沿用 Lichess 定下的做法，檢驗系統的期望勝率在另一個遊戲與配對機制下是否校準、
# 加入近況能否在樣本外改善預測、民間停止規則有沒有用（規格見 docs/plan.md「其他遊戲的外部驗證」）。
# 沿用的是方法，不是係數：等級分的尺度與遊戲不同，各項在 OGS 自己的估計段重估。
#
# - 系統的期望勝率：scripts/ogs_replay.py 依 OGS 公開的規則重放所得（與參考實作逐局比對一致）
# - 樣本：2021 至 2023 年結束的 19 路、不讓子、非通訊棋（每手 1 至 3600 秒）、雙方都不是機器人、勝者為雙方之一的對局，
#   重放中未被跳過。速度類別依 OGS 的分法：每手 15 秒以下為 blitz，其餘為 live。2021、2022 年估計，2023 年評估
# - 一天：每位玩家的非通訊棋序列（含不在樣本中的對局，例如 9 路），與上一局的結束時間相隔超過 12 小時即切開。
#   OGS 沒有開局時間，間隔以結束時間計。「上一局」為同一天的上一局，不論是否在樣本中
# - 主基準：binary logistic，logit P(黑勝) = t0 + t1 xr + t2 xr^2 + (t3 + t4 xr + t5 g(sigma)) xd，即 Lichess 主基準去掉和局項。
#   沒有和局，有序模型退化成這個形式；t0 吸收黑先與貼目
# - 依狀態調整的勝率：主基準作 offset，依序加入雙方上一局結果之差（含「本日無上一局」之差）、上一局等級分變動之差，
#   2023 年的對數分數改善且 dyadic 區間不含 0 才保留。黑方視角每局一列，雙方共用係數即以差的形式進入
# - 新手：任一方對局前的 RD 不低於 160（OGS 程式中 provisional 的門檻）
# - 停止規則：2023 年開始的天，回測與判定同 Lichess（stop_rules.py）；邊界 10 等級分，以主基準的斜率換算

import os
import sqlite3
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, "scripts")

import numpy as np
import polars as pl

import likelihood as lk
import ordinal
from state_adjusted import diff_row, losses
from runs import runs_before
from stop_rules import apply_rules, backtest

SPEEDS = ["blitz", "live"]
DAY_GAP = 12 * 3600
NOVICE_RD = 160
DELTA = 10
T_LO, T_EVAL, T_HI = 1609459200, 1672531200, 1704067200  # 2021-01-01、2023-01-01、2024-01-01（UTC）

con = sqlite3.connect("file:data/raw/ogs/ogs-data.db?mode=ro", uri=True)
q = f"""
SELECT g.id AS game_id, size, handicap, black_id, white_id, time_per_move, winner_id, ended,
       coalesce(bp.is_bot, 1) AS black_bot, coalesce(wp.is_bot, 1) AS white_bot
FROM game_records g
LEFT JOIN players bp ON black_id = bp.id
LEFT JOIN players wp ON white_id = wp.id
WHERE ended >= {T_LO - 31 * 86400} AND ended < {T_HI} AND time_per_move > 0 AND time_per_move <= 3600
"""
g = pl.concat(pl.read_database(q, con, iter_batches=True, batch_size=1_000_000))
g = g.join(pl.scan_parquet("data/processed/ogs_replay.parquet").collect(), on="game_id").filter(~pl.col("skipped"), ~pl.col("overflow"))
g = g.with_columns(
    speed=pl.when(pl.col("time_per_move") <= 15).then(pl.lit("blitz")).otherwise(pl.lit("live")),
    sample=(pl.col("size") == 19) & (pl.col("handicap") == 0) & (pl.col("black_bot") == 0) & (pl.col("white_bot") == 0)
    & ((pl.col("winner_id") == pl.col("black_id")) | (pl.col("winner_id") == pl.col("white_id"))) & (pl.col("ended") >= T_LO),
)

# 每位玩家每局一列，依結束時間切出一天，取同一天的上一局
side = lambda me, op: g.select(  # noqa: E731
    "game_id", "ended", "speed", "sample",
    player=f"{me}_id", opponent=f"{op}_id", rating=f"{me}_rating", rd=f"{me}_rd",
    score=(pl.col("winner_id") == pl.col(f"{me}_id")).cast(pl.Float64),
)
pg = (
    pl.concat([side("black", "white"), side("white", "black")])
    .sort("player", "ended", "game_id")
    .with_columns(gap=pl.col("ended").diff().over("player"))
    .with_columns(day=(pl.col("gap").is_null() | (pl.col("gap") > DAY_GAP)).cum_sum().over("player"))
    .with_columns(
        has_prev=(pl.col("day") == pl.col("day").shift(1).over("player")).fill_null(False),
        prev_score=pl.col("score").shift(1).over("player"),
        dr=((pl.col("rating") - pl.col("rating").shift(1).over("player")) / 100),
        day_start=pl.col("ended").first().over("player", "day"),
        result=pl.when(pl.col("score") == 1).then(pl.lit("win")).otherwise(pl.lit("loss")),
    )
    .with_columns(
        prev_win=pl.when("has_prev").then(pl.col("prev_score")).otherwise(0.0),
        missing=(~pl.col("has_prev")).cast(pl.Float64),
        dr=pl.when("has_prev").then(pl.col("dr")).otherwise(0.0),
    )
)
pg = runs_before(pg, ["player", "day"])
st = pg.select("game_id", "player", "prev_win", "missing", "dr")
t = (
    g.filter("sample")
    .join(st.rename({c: f"b_{c}" for c in ["prev_win", "missing", "dr"]}), left_on=["game_id", "black_id"], right_on=["game_id", "player"])
    .join(st.rename({c: f"w_{c}" for c in ["prev_win", "missing", "dr"]}), left_on=["game_id", "white_id"], right_on=["game_id", "player"])
    .with_columns(
        y=(pl.col("winner_id") == pl.col("black_id")).cast(pl.Float64),
        xr=((pl.col("black_rating") + pl.col("white_rating")) / 2 - 1500) / 100,
        xd=(pl.col("black_rating") - pl.col("white_rating")) / 100,
        gs=ordinal.g("black_rd", "white_rd"),
        L_sys=(pl.col("expected_black") / (1 - pl.col("expected_black"))).log(),
        d_prev_win=pl.col("b_prev_win") - pl.col("w_prev_win"),
        d_missing=pl.col("b_missing") - pl.col("w_missing"),
        d_dr=pl.col("b_dr") - pl.col("w_dr"),
        novice=pl.max_horizontal("black_rd", "white_rd") >= NOVICE_RD,
        train=pl.col("ended") < T_EVAL,
        const=pl.lit(1.0),
    )
    .with_columns(xr2=pl.col("xr") ** 2, xdxr=pl.col("xd") * pl.col("xr"), gxd=pl.col("gs") * pl.col("xd"))
)
sample = t.group_by("speed", "train").agg(n=pl.len(), players=pl.concat_list("black_id", "white_id").explode().n_unique(), novice=pl.col("novice").mean(), black_win=pl.col("y").mean()).sort("speed", "train")
print(sample)
sample.write_csv("outputs/tables/other_games_ogs_sample.csv")

BASE = ["const", "xr", "xr2", "xd", "xdxr", "gxd"]
STATE = [("prev_result", ["d_prev_win", "d_missing"]), ("dr", ["d_dr"])]
cal_rows, step_rows, pred_rows, coef_rows, stop_parts, eval_parts = [], [], [], [], [], []
for spd in SPEEDS:
    ts = t.filter(pl.col("speed") == spd)
    tr, te = ts.filter("train"), ts.filter(~pl.col("train"))
    wtr, btr, wte, bte = tr["black_id"].to_numpy(), tr["white_id"].to_numpy(), te["black_id"].to_numpy(), te["white_id"].to_numpy()
    th, *_ = lk.fit(lk.logit_ll, np.zeros(len(BASE)), np.arange(len(BASE)), (tr.select(BASE).to_numpy().astype(np.float64), tr["y"].to_numpy()), wtr, btr)
    eta = lambda d: d.select(BASE).to_numpy().astype(np.float64) @ th  # noqa: E731
    tr, te = tr.with_columns(L_base=pl.Series(eta(tr))), te.with_columns(L_base=pl.Series(eta(te)))

    # 系統與主基準的校準：logit E[y] = alpha + beta L，2023 年；完全校準為 alpha = 0、beta = 1
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
        return c, losses(np.c_[1 - p, np.zeros(len(p)), p], (te["y"].to_numpy() * 2).astype(int)), p

    _, loss_base, _ = run([])
    p_sys = te["expected_black"].to_numpy()
    loss_sys = losses(np.c_[1 - p_sys, np.zeros(len(p_sys)), p_sys], (te["y"].to_numpy() * 2).astype(int))
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

    # 停止規則：玩家視角的殘差，殘差取主基準
    stop_parts += [
        te.select("game_id", player="black_id", e=1 / (1 + (-pl.col("L_base")).exp())),
        te.select("game_id", player="white_id", e=1 - 1 / (1 + (-pl.col("L_base")).exp())),
    ]
    # 評估段的三個預報，供連勝連敗（streaks.py）與 Brier 分解（forecast_decomposition.py）使用
    te = te.with_columns(p_adj=pl.Series(p_adj), p_base=1 / (1 + (-pl.col("L_base")).exp()))
    eval_parts += [
        te.select("game_id", "speed", "novice", player="black_id", opponent="white_id", score="y", e_sys="expected_black", e_base="p_base", e_adj="p_adj"),
        te.select("game_id", "speed", "novice", player="white_id", opponent="black_id", score=1 - pl.col("y"), e_sys=1 - pl.col("expected_black"), e_base=1 - pl.col("p_base"), e_adj=1 - pl.col("p_adj")),
    ]
    # 主基準下期望得分對自身等級分的斜率（每 1 分）：e(1 - e) (t3 + t4 xr + t5 g) / 100
    spe_te = te.select(s=(1 / (1 + (-pl.col("L_base")).exp())) * (1 - 1 / (1 + (-pl.col("L_base")).exp())) * (th[3] + th[4] * pl.col("xr") + th[5] * pl.col("gs")) / 100)["s"].mean()
    coef_rows.append({"speed": spd, "param": "score_per_rating", "est": spe_te})
    print(spd, "done", kept, flush=True)

cal, steps, pred, coef = pl.DataFrame(cal_rows), pl.DataFrame(step_rows), pl.DataFrame(pred_rows), pl.DataFrame(coef_rows)
with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250):
    print(cal)
    print(steps)
    print(pred.filter(pl.col("metric") == "log_loss"))
cal.write_csv("outputs/tables/other_games_ogs_calibration.csv")
steps.write_csv("outputs/tables/other_games_ogs_steps.csv")
pred.write_csv("outputs/tables/other_games_ogs_prediction.csv")
coef.write_csv("outputs/tables/other_games_ogs_coef.csv")
pl.concat(eval_parts).join(pg.select("game_id", "player", "day", "loss_run", "win_run"), on=["game_id", "player"]).write_parquet("data/processed/ogs_eval_rows.parquet")

# 停止規則的序列只含 2023 年的樣本局；一天的切分沿用全部非通訊棋的序列
d = (
    pg.join(pl.concat(stop_parts), on=["game_id", "player"])
    .join(pg.select("game_id", opponent="player", opp_rd="rd"), on=["game_id", "opponent"])
    .with_columns(
        resid=pl.col("score") - pl.col("e"),
        novice_rd=pl.max_horizontal("rd", "opp_rd") >= NOVICE_RD,
    )
    .sort("player", "ended", "game_id")
)
d, del_days = apply_rules(d, pl.col("day_start") >= T_EVAL)
res, reset, *_ = backtest(d, del_days, 2000, speeds=SPEEDS, family=SPEEDS)
spe = dict(coef.filter(pl.col("param") == "score_per_rating").select("speed", "est").iter_rows())
res = res.with_columns(spe=pl.col("speed").replace_strict(spe, return_dtype=pl.Float64)).with_columns(
    diff_rating=pl.col("diff") / pl.col("spe"), lo_rating=pl.col("lo") / pl.col("spe"), hi_rating=pl.col("hi") / pl.col("spe"),
).with_columns(
    verdict=pl.when((pl.col("diff") < 0) & (pl.col("p_holm") < 0.05)).then(pl.lit("beats_never_stop"))
    .when((pl.col("lo_rating") > -DELTA) & (pl.col("hi_rating") < DELTA)).then(pl.lit("negligible"))
    .otherwise(pl.lit("inconclusive"))
)
with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250):
    print(res.filter(pl.col("group") == "all"))
    print(reset)
res.write_csv("outputs/tables/other_games_ogs_stop_rules.csv")
reset.write_csv("outputs/tables/other_games_ogs_stop_rules_reset.csv")
