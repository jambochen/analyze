# 速度類別規則變更的實際上線時間。commit 時間只是下限，上線時間要從表頭看：
# 一局的對局前等級分應等於同一玩家在「計分所用的速度類別」上一局的等級分加
# RatingDiff。對規則有歧義的對局，比對它接上哪一條鏈。
# 歧義一：est = base + 30 * inc 與 base + 40 * inc 落在不同類別的時間制（2015-04）。
# 歧義二：無限時對局（"-"）計入 Classical 或通信棋（2014-12）。

import polars as pl

g = (
    pl.scan_parquet("data/processed/lichess_games.parquet")
    .filter(
        pl.col("start").is_between(pl.datetime(2015, 4, 10, time_zone="UTC"), pl.datetime(2015, 5, 5, time_zone="UTC"))
        | pl.col("start").is_between(pl.datetime(2014, 11, 15, time_zone="UTC"), pl.datetime(2015, 1, 15, time_zone="UTC"))
    )
    .select("start", "white_id", "black_id", "white_elo", "black_elo", "white_rating_diff", "black_rating_diff", "base_seconds", "increment_seconds")
    .with_columns(
        e30=pl.col("base_seconds") + 30 * pl.col("increment_seconds"),
        e40=pl.col("base_seconds") + 40 * pl.col("increment_seconds"),
    )
    .with_columns(
        [
            pl.when(pl.col(e).is_null()).then(pl.lit("nc")).when(pl.col(e) < 180).then(pl.lit("bullet")).when(pl.col(e) < 480).then(pl.lit("blitz")).otherwise(pl.lit("classical")).alias(e.replace("e", "c"))
            for e in ["e30", "e40"]
        ]
    )
)
p = pl.concat([
    g.select("start", "c30", "c40", player="white_id", elo="white_elo", diff="white_rating_diff"),
    g.select("start", "c30", "c40", player="black_id", elo="black_elo", diff="black_rating_diff"),
]).collect().sort("player", "start")

# 每位玩家在各個非歧義類別上一局的「局後」等級分，往後填到每一列
post = pl.col("elo") + pl.col("diff")
cands = ["bullet", "blitz", "classical", "nc"]
p = p.with_columns(
    [
        pl.when((pl.col("c30") == c) & (pl.col("c40") == c)).then(post).otherwise(None).shift(1).forward_fill().over("player").alias(f"last_{c}")
        for c in cands
    ]
)

def last_of(col):
    e = pl.lit(None, dtype=pl.Int32)
    for c in cands:
        e = pl.when(pl.col(col) == c).then(pl.col(f"last_{c}")).otherwise(e)
    return e


amb = p.filter(pl.col("c30") != pl.col("c40")).with_columns(
    m30=pl.col("elo") == last_of("c30"),
    m40=pl.col("elo") == last_of("c40"),
)
print(
    amb.filter(pl.col("start").is_between(pl.datetime(2015, 4, 24, 18, time_zone="UTC"), pl.datetime(2015, 4, 25, 18, time_zone="UTC")))
    .group_by(h=pl.col("start").dt.truncate("1h"))
    .agg(pl.len(), pl.col("m30").mean(), pl.col("m40").mean())
    .sort("h")
)

nc = p.filter(pl.col("c30") == "nc").with_columns(
    m_cl=pl.col("elo") == pl.col("last_classical"),
    m_nc=pl.col("elo") == pl.col("last_nc"),
)
print(nc.filter(pl.col("start").is_between(pl.datetime(2014, 11, 29, time_zone="UTC"), pl.datetime(2014, 12, 8, time_zone="UTC"))).group_by(d=pl.col("start").dt.truncate("12h")).agg(pl.len(), pl.col("m_cl").mean(), pl.col("m_nc").mean(), first1500=(pl.col("elo") == 1500).mean()).sort("d"))
