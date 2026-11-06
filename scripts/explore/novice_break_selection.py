# 探索：novice_break.py 的逐局診斷在 k = -1 出現階梯。RD 的下降取決於該局的資訊量，分差小的對局資訊最多，
# 所以讓 RD 跌破 110 的那一局（k = -1）可能偏向分差小的對局。這裡只看各 k 的分差分布，看過結果才做，不作檢定。
# 原註解：新手門檻（RD 110）處的斷點：以事件時間對齊，檢定 Elo 期望得分的校準截距與斜率在越過門檻時是否跳躍。
# 規格見 docs/methods.md「新手期的界定」的「斷點的檢定」。邊際模型：新手的偏差即帳號實力與等級分之差，
# 玩家隨機效果會把它吸收掉。

import os
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, "scripts")

import polars as pl

import game_source
import novice
import ordinal

UTC = "UTC"
SPEEDS = ["Bullet", "Blitz", "Classical"]
K_LO, K_HI = -5, 4

rp = pl.scan_parquet("data/processed/glicko_replay.parquet")
side = pl.concat([
    rp.select("game_id", color=pl.lit(c), rd=f"{c}_rd", lt=f"{c}_left_truncated", opp_rd=f"{o}_rd", opp_lt=f"{o}_left_truncated")
    for c, o in [("white", "black"), ("black", "white")]
])
pg = (
    pl.scan_parquet("data/processed/player_games.parquet")
    .filter(pl.col("speed").is_in(SPEEDS), pl.col("start") < pl.datetime(2015, 1, 1, time_zone=UTC))
    .select("game_id", "player", "opponent", "color", "start", "speed", "source", "elo", "opp_elo", "score")
    .join(side, on=["game_id", "color"])
    .sort("player", "speed", "start", "game_id")
    .collect(engine="streaming")
)
# 事件時間在玩家該速度類別的全部 rated 對局上計算：RD 隨每一局更新，不論來源
pg = pg.with_columns(
    idx=pl.int_range(pl.len()).over("player", "speed"),
    first_elo=pl.col("elo").first().over("player", "speed"),
    first_lt=pl.col("lt").first().over("player", "speed"),
)
cross = pg.filter(pl.col("rd") < novice.RD).group_by("player", "speed").agg(cross_idx=pl.col("idx").min())
t = (
    pg.join(cross, on=["player", "speed"])
    .with_columns(k=pl.col("idx") - pl.col("cross_idx"))
    .filter(
        pl.col("first_elo") == 1500, ~pl.col("first_lt"),
        pl.col("k").is_between(K_LO, K_HI),
        pl.col("opp_rd") < novice.RD, ~pl.col("opp_lt"),
        pl.col("start") >= pl.datetime(2014, 1, 1, time_zone=UTC),
        pl.col("elo").is_not_null(), pl.col("opp_elo").is_not_null(), pl.col("score").is_not_null(),
    )
)
t = t.filter(game_source.KEEP)
t = t.with_columns(
    L=(pl.col("elo") - pl.col("opp_elo")).cast(pl.Float64) * ordinal.Q,
    J=(pl.col("k") >= 0).cast(pl.Float64),
    kf=pl.col("k").cast(pl.Float64),
).with_columns(kL=pl.col("kf") * pl.col("L"), JL=pl.col("J") * pl.col("L"))

out = t.group_by("speed", "k").agg(n=pl.len(), abs_L=pl.col("L").abs().mean(), abs_L_med=pl.col("L").abs().median(), opp_rd=pl.col("opp_rd").mean()).sort("speed", "k")
with pl.Config(tbl_rows=-1):
    print(out)
