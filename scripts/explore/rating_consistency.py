# 檢查表頭的等級分與 RatingDiff 是否同一口徑：本局等級分加 RatingDiff，
# 應等於同一玩家同一時間類別下一局的等級分。
# 2014 年 7 月以前線上存的 RatingDiff 是全站等級分的變動；若月檔已改寫為各速度
# 類別的值，各月的相符比例應一致。

import polars as pl

g = pl.scan_parquet("data/processed/lichess_games.parquet").filter(
    pl.col("month").is_in(["2013-06", "2014-02", "2014-06", "2014-09", "2015-03"])
)
p = (
    pl.concat([
        g.select("month", "start", "game_id", "time_class", player="white_id", elo="white_elo", diff="white_rating_diff"),
        g.select("month", "start", "game_id", "time_class", player="black_id", elo="black_elo", diff="black_rating_diff"),
    ])
    .collect()
    .sort("player", "time_class", "start", "game_id")
    .with_columns(nxt=pl.col("elo").shift(-1).over("player", "time_class"))
    .drop_nulls(["diff", "nxt"])
    .with_columns(ok=(pl.col("elo") + pl.col("diff") - pl.col("nxt")).abs() <= 1)
)
print(p.group_by("month").agg(n=pl.len(), share_exact=pl.col("ok").mean()).sort("month"))
