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
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, "scripts")

import polars as pl

from external import evaluate, player_rows, stop_verdicts
import external_sample

SPEEDS = ["blitz", "live"]
T_EVAL = 1672531200
NOVICE_RD = 160
t, pg = external_sample.ogs()
sample = t.group_by("speed", "train").agg(n=pl.len(), players=pl.concat_list("black_id", "white_id").explode().n_unique(), novice=pl.col("novice").mean(), black_win=pl.col("y").mean(), rating_diff_sd=(pl.col("xd") * 100).std()).sort("speed", "train")
print(sample)
sample.write_csv("outputs/tables/other_games_ogs_sample.csv")

BASE = ["const", "xr", "xr2", "xd", "xdxr", "gxd"]
# 黑方為 A 方；主基準線性預測式對黑方等級分（每 100 分）的斜率為 t3 + t4 xr + t5 g
tables, te = evaluate(
    t.rename({"black_id": "a_id", "white_id": "b_id", "expected_black": "p_sys"}), SPEEDS, BASE,
    lambda th, d: th[3] + th[4] * d["xr"] + th[5] * d["gs"],
)
with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250):
    print(tables["calibration"])
    print(tables["steps"])
    print(tables["prediction"].filter(pl.col("metric") == "log_loss"))
for k, v in tables.items():
    v.write_csv(f"outputs/tables/other_games_ogs_{k}.csv")
# 評估段的三個預報，供連勝連敗（streaks.py）與 Brier 分解（forecast_decomposition.py）使用
ev = player_rows(te)
ev.join(pg.select("game_id", "player", "day", "loss_run", "win_run"), on=["game_id", "player"]).write_parquet("data/processed/ogs_eval_rows.parquet")

# 停止規則的序列只含 2023 年的樣本局，殘差取主基準；一天的切分沿用全部非通訊棋的序列
d = (
    pg.join(ev.select("game_id", "player", e="e_base"), on=["game_id", "player"])
    .join(pg.select("game_id", opponent="player", opp_rd="rd"), on=["game_id", "opponent"])
    .with_columns(resid=pl.col("score") - pl.col("e"), novice_rd=pl.max_horizontal("rd", "opp_rd") >= NOVICE_RD)
    .sort("player", "ended", "game_id")
)
res, reset = stop_verdicts(d, pl.col("day_start") >= T_EVAL, SPEEDS, tables["coef"])
with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250):
    print(res.filter(pl.col("group") == "all"))
    print(reset)
res.write_csv("outputs/tables/other_games_ogs_stop_rules.csv")
reset.write_csv("outputs/tables/other_games_ogs_stop_rules_reset.csv")
