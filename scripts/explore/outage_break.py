# 可行性實測：全站停擺迫使正在連續對局的玩家中斷，能否據此估計「休息」對表現的效果。
# 受影響者：停擺前最後一局開局在停擺前 T 分鐘內、且主門檻下仍在連續對局中（停擺前一局到停擺開始的間隔未超過門檻）。
# 看受影響的人數、停擺後多久回來、回來第一局的表現殘差，以及與自願中斷後第一局比較時的標準誤。

import sys

sys.path.insert(0, "scripts")
import polars as pl

from calibrated import probs

T = 30 * 60
out = pl.read_csv("outputs/tables/sessions_outages.csv", try_parse_dates=True).with_columns(
    outage_end=pl.col("outage_start") + pl.duration(seconds=pl.col("outage_seconds"))
).filter(pl.col("outage_start") >= pl.datetime(2014, 1, 1, time_zone="UTC"))
print(out)

# 全部玩家局套校準模型會超出記憶體；對照組也取受影響者自己的其他中斷，故只讀受影響者的對局
near = pl.any_horizontal(
    pl.col("start").is_between(s - pl.duration(seconds=T), s) for s in out["outage_start"]
)
players = pl.scan_parquet("data/processed/player_games.parquet").filter(near).select("player").unique().collect()["player"].implode()
g = (
    pl.scan_parquet("data/processed/player_games.parquet")
    .filter(pl.col("player").is_in(players), pl.col("elo").is_not_null(), pl.col("opp_elo").is_not_null(), pl.col("score").is_not_null())
    .select("player", "start", "speed", "color", "elo", "opp_elo", "score", "prev_gap", "return_seconds", "session_threshold", "game_seq")
    .collect()
)
g = probs(g, color=True).with_columns(resid=pl.col("score") - pl.col("e_cal")).drop("p_win", "p_draw", "p_loss")

# 停擺前的最後一局：開局落在 [停擺開始 - T, 停擺開始]，下一局在停擺結束之後
last = (
    g.sort("start")
    .join_asof(out.select("outage_start", "outage_end", "outage_seconds").sort("outage_start"), left_on="start", right_on="outage_start", strategy="forward")
    .filter((pl.col("outage_start") - pl.col("start")).dt.total_seconds() <= T)
    .with_columns(back=pl.col("start") + pl.duration(seconds=pl.col("return_seconds")))
)
print("games started within T before an outage", last.height, "players", last["player"].n_unique())
# 停擺前 T 分鐘內的局不一定是最後一局；下一局仍在停擺之前者不是被停擺打斷的
back = last.with_columns(wait_after=(pl.col("back") - pl.col("outage_end")).dt.total_seconds()).filter(pl.col("wait_after") >= 0)
print("interrupted", back.height, "players", back["player"].n_unique())
print(back.select(pl.col("wait_after").quantile(q).alias(str(q)) for q in [0.1, 0.25, 0.5, 0.75, 0.9]))

# 回來的第一局：停擺結束後 2 小時內回來者
first_back = (
    back.filter(pl.col("wait_after") <= 2 * 3600)
    .select("player", start="back", break_seconds="return_seconds")
    .join(g, on=["player", "start"])
)
# 對照：同一批玩家自願中斷後的第一局，中斷長度落在受影響者的範圍內；停擺本身的那次中斷不算
lo, hi = first_back["break_seconds"].quantile(0.05), first_back["break_seconds"].quantile(0.95)
ctrl = g.filter(pl.col("prev_gap").is_between(lo, hi), pl.col("game_seq") == 1).join(first_back.select("player", "start"), on=["player", "start"], how="anti")
res = pl.DataFrame({
    "group": ["outage_return", "voluntary_break"],
    "n": [first_back.height, ctrl.height],
    "players": [first_back["player"].n_unique(), ctrl["player"].n_unique()],
    "resid_mean": [first_back["resid"].mean(), ctrl["resid"].mean()],
    "resid_sd": [first_back["resid"].std(), ctrl["resid"].std()],
})
res = res.with_columns(se=pl.col("resid_sd") / pl.col("n").sqrt())
print(lo, hi)
print(res)
res.write_csv("outputs/tables/explore_outage_break.csv")
