# 把 API 回應攤平成每局一列：配對來源、建立與最後一步的時間、結束方式、API 存的對局當時等級分。
# 第一次列出各來源在 2014 年各月與各時間制的分布，以及 API 等級分與月檔 Elo 是否一致。

import gzip
import json

import polars as pl

rows = []
with gzip.open("data/raw/lichess-game-meta/games.ndjson.gz", "rt") as f:
    for l in f:
        if not l.strip():
            continue
        d = json.loads(l)
        p = d["players"]
        rows.append((d["id"], d.get("source"), d["createdAt"], d["lastMoveAt"], d["status"], d.get("winner"),
                     p["white"].get("rating"), p["black"].get("rating"), d.get("clock", {}).get("initial"), d.get("clock", {}).get("increment")))
meta = pl.DataFrame(rows, schema=["game_id", "source", "created_ms", "last_move_ms", "status", "winner", "white_rating", "black_rating", "clock_initial", "clock_increment"], orient="row")
meta.write_parquet("data/processed/game_meta.parquet")

g = pl.read_parquet("data/processed/lichess_games.parquet").filter(pl.col("start").dt.year() == 2014).join(meta, on="game_id", how="left")
print(g.columns)
print("未回應", g.filter(pl.col("created_ms").is_null()).height)

g = g.with_columns(src=pl.col("source").fill_null("none"), month=pl.col("start").dt.month(), arena=pl.col("tournament_id").is_not_null())
by_month = g.group_by("month", "src").len().pivot(on="src", index="month", values="len").sort("month").fill_null(0)
print(by_month)
by_month.write_csv("outputs/tables/game_meta_source_by_month.csv")

# 月檔的錦標賽標記與 API 的 source 是否一致
print(g.group_by("arena", "src").len().sort("arena", "src"))

# source 缺值的對局：時間制與等級分是否與 pool 試行期相符
none = g.filter(pl.col("src") == "none")
print(none.group_by("month").len().sort("month"))
print(none.group_by("clock_initial", "clock_increment").len().sort("len", descending=True).head(8))
