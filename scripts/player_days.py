# 玩家日彙整：Heliyon 2024 的分析單位。每列為（玩家、日、切點、對局類型）。
# 對局類型 arena 與 lobby（一般對局）分開計算當日變數；day_pure 標記當日只下了該類型。
# 流失取當日（不分類型）最後一局的 churn_14d，因為「之後 14 天沒有對局」是對玩家整天而言。
# 依玩家 hash 分桶彙整：視窗運算與 group_by 的分組數是記憶體瓶頸，玩家序列不跨桶。

import sys

sys.path.insert(0, "scripts")
import polars as pl

from calibrated import probs

NB = 16
UTC = "UTC"
COLS = ["player", "color", "start", "time_class", "arena", "elo", "opp_elo", "score", "result", "player_game_no", "churn_14d"]

base = pl.scan_parquet("data/processed/player_games.parquet").filter(
    pl.col("start") >= pl.datetime(2014, 1, 1, time_zone=UTC),
    pl.col("start") < pl.datetime(2015, 1, 2, time_zone=UTC),
)

# Heliyon 以全日活動最少的時刻為切點；Lichess 的玩家跨時區，取全站 UTC 每小時開局數的最低點作平移切點
hour = (
    base.filter(pl.col("start") < pl.datetime(2015, 1, 1, time_zone=UTC))
    .group_by(hour=pl.col("start").dt.hour())
    .agg(games=pl.len() / 2)
    .sort("hour")
    .collect()
)
print(hour)
hour.write_csv("outputs/tables/player_days_hour.csv")
SHIFT = int(hour.sort("games")["hour"][0])
print("shift hours", SHIFT)

out = []
for b in range(NB):
    g = (
        base.filter(pl.col("player").hash(1) % NB == b)
        .select(COLS)
        .collect()
        .sort("player", "start")
    )
    g = (
        probs(g.with_columns(elo_ok=pl.col("elo").is_not_null() & pl.col("opp_elo").is_not_null()), color=False)
        .drop("p_win", "p_draw", "p_loss")
        .sort("player", "start")
    )
    for cut, h in [("utc", 0), ("shift", SHIFT)]:
        d = g.with_columns(day=(pl.col("start") - pl.duration(hours=h)).dt.date()).filter(pl.col("day").dt.year() == 2014)
        dayinfo = d.group_by("player", "day").agg(
            churn=pl.col("churn_14d").last(),
            n_day=pl.len(),
            n_arena_day=pl.col("arena").sum(),
        )
        # 缺等級分的局不計入分差與表現，但仍算當日有下（流失與 n_day 已含）
        e = d.filter(pl.col("elo_ok")).with_columns(
            gap=(pl.col("opp_elo") - pl.col("elo")).cast(pl.Float64),
            win=(pl.col("result") == "win").cast(pl.Float64),
        )
        # 連勝、連敗自連續第 3 局起算，連續段在日界與類型間切斷（原文的連勝率以當日對局為範圍）；和局中斷兩者
        e = e.with_columns(run=(pl.col("result") != pl.col("result").shift(1)).fill_null(True).cum_sum().over("player", "day", "arena"))
        e = e.with_columns(run_pos=pl.int_range(1, pl.len() + 1).over("player", "day", "arena", "run"))
        a = e.group_by("player", "day", "arena").agg(
            n=pl.len(),
            gap_mean=pl.col("gap").mean(),
            gap_sd=pl.col("gap").std(ddof=0),
            ecal_mean=pl.col("e_cal").mean(),
            ecal_sd=pl.col("e_cal").std(ddof=0),
            q_mean=(pl.col("e_cal") - 0.5).abs().mean(),
            elo_mean=pl.col("elo").cast(pl.Float64).mean(),
            win_rate=pl.col("win").mean() * 100,
            score_rate=pl.col("score").mean() * 100,
            win_streak_rate=((pl.col("result") == "win") & (pl.col("run_pos") >= 3)).mean() * 100,
            loss_streak_rate=((pl.col("result") == "loss") & (pl.col("run_pos") >= 3)).mean() * 100,
            draw_rate=(pl.col("result") == "draw").mean() * 100,
            game_no=pl.col("player_game_no").last(),
            first_game_no=pl.col("player_game_no").first(),
            bullet=(pl.col("time_class") == "Bullet").mean(),
            classical=(pl.col("time_class") == "Classical").mean(),
        )
        a = a.join(dayinfo, on=["player", "day"]).with_columns(
            cut=pl.lit(cut),
            day_pure=pl.when(pl.col("arena")).then(pl.col("n_arena_day") == pl.col("n_day")).otherwise(pl.col("n_arena_day") == 0),
        )
        out.append(a)
    print(b, sum(x.height for x in out))

pd_ = pl.concat(out).with_columns(pl.col("player").cast(pl.Categorical))
print(pd_.describe())
pd_.write_parquet("data/processed/player_days.parquet")

sample = (
    pd_.group_by("cut", "arena", "day_pure")
    .agg(player_days=pl.len(), players=pl.col("player").n_unique(), games=pl.col("n").sum(), churn_rate=pl.col("churn").mean())
    .sort("cut", "arena", "day_pure")
)
print(sample)
sample.write_csv("outputs/tables/player_days_sample.csv")
