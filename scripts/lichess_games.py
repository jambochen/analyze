# 把 Lichess 月檔的 PGN 表頭解析成單一 parquet，每局一列，依開始時間排序。
# 錦標賽的賽制由 API 查詢結果併入，每場錦標賽一種賽制。

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor

import polars as pl

MONTHS = (
    [f"2013-{m:02d}" for m in range(1, 13)]
    + [f"2014-{m:02d}" for m in range(1, 13)]
    + [f"2015-{m:02d}" for m in range(1, 7)]
)


def parse(m):
    # 棋步行不以 [ 開頭，只留表頭
    cmd = f"zstdcat data/raw/lichess/lichess_db_standard_rated_{m}.pgn.zst | grep '^\\['"
    out = subprocess.run(cmd, shell=True, capture_output=True, check=True).stdout
    return (
        pl.DataFrame({"line": out.decode().splitlines()})
        .select(pl.col("line").str.extract_groups(r'^\[(\w+) "(.*)"\]$').struct.rename_fields(["tag", "value"]).struct.unnest())
        # 每局的表頭欄位數不固定（頭銜、RatingDiff 可能缺），以 Event 標記每局的起點
        .with_columns(g=(pl.col("tag") == "Event").cum_sum())
        .pivot(on="tag", index="g", values="value", aggregate_function="first")
        .drop("g")
        .with_columns(month=pl.lit(m))
    )


# 解析的重活在 zstdcat、grep 與 polars，都不受 GIL 限制
with ThreadPoolExecutor(4) as ex:
    frames = list(ex.map(parse, MONTHS))

src = pl.read_csv("data/raw/lichess-tournament-source/tournament_game.csv").join(
    pl.DataFrame([json.loads(l) for l in open("data/raw/lichess-tournament-source/games.ndjson")]).select(
        game_id=pl.col("id"), tournament_system=pl.col("source")
    ),
    on="game_id",
    how="left",
)

games = (
    pl.concat(frames, how="diagonal")
    .with_columns(
        game_id=pl.col("Site").str.extract(r"lichess\.org/(\w+)"),
        tournament_id=pl.col("Event").str.extract(r"/tournament/(\w+)"),
        # 2015-03 的月檔把其他月份的 Classical 標為 Standard，只留最長的時間制為
        # Classical；時間制的分布與前後月份的 Classical 相符，故併回 Classical
        time_class=pl.col("Event").str.extract(r"^Rated (\w+) ").replace("Standard", "Classical"),
        start=(pl.col("UTCDate") + " " + pl.col("UTCTime")).str.to_datetime("%Y.%m.%d %H:%M:%S", time_zone="UTC"),
        # ? 表示無值
        white_elo=pl.col("WhiteElo").replace("?", None).cast(pl.Int32),
        black_elo=pl.col("BlackElo").replace("?", None).cast(pl.Int32),
        white_rating_diff=pl.col("WhiteRatingDiff").cast(pl.Int32),
        black_rating_diff=pl.col("BlackRatingDiff").cast(pl.Int32),
        base_seconds=pl.col("TimeControl").str.extract(r"^(\d+)\+").cast(pl.Int32),
        increment_seconds=pl.col("TimeControl").str.extract(r"\+(\d+)$").cast(pl.Int32),
    )
    .join(src.select("tournament_id", "tournament_system"), on="tournament_id", how="left")
    .select(
        "game_id", "start", "month", "time_class", "tournament_id", "tournament_system",
        white="White", black="Black", result="Result",
        white_elo="white_elo", black_elo="black_elo",
        white_rating_diff="white_rating_diff", black_rating_diff="black_rating_diff",
        white_title="WhiteTitle", black_title="BlackTitle",
        eco="ECO", opening="Opening", time_control="TimeControl",
        base_seconds="base_seconds", increment_seconds="increment_seconds",
        termination="Termination",
    )
    .sort("start", "game_id")
)

games.write_parquet("data/processed/lichess_games.parquet")
print(games.height)
print(games.group_by("month").len().sort("month"))
print(games.filter(pl.col("tournament_id").is_not_null()).group_by("month", "tournament_system").len().sort("month", "tournament_system"))
print(games.null_count())
