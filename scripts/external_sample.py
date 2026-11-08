# 外部遊戲的樣本建構；既有分析與時間外驗證共用，不在載入時執行回測。
import datetime as dt
import sqlite3
import polars as pl
import novice
import ordinal
from runs import runs_before

def ogs():
    """回傳既有口徑的對局樣本與玩家序列，供預測流程共用。"""
    DAY_GAP = 12 * 3600
    NOVICE_RD = 160
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

    # 自我對局不屬於雙人預測，且會使雙方視角的串接鍵重複。
    print({"ogs_self_games_excluded": g.filter(pl.col("black_id") == pl.col("white_id")).height}, flush=True)
    g = g.filter(pl.col("black_id") != pl.col("white_id"))

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
            d_prev_win=pl.col("b_prev_win") - pl.col("w_prev_win"),
            d_missing=pl.col("b_missing") - pl.col("w_missing"),
            d_dr=pl.col("b_dr") - pl.col("w_dr"),
            novice=pl.max_horizontal("black_rd", "white_rd") >= NOVICE_RD,
            train=pl.col("ended") < T_EVAL,
            const=pl.lit(1.0),
        )
        .with_columns(xr2=pl.col("xr") ** 2, xdxr=pl.col("xd") * pl.col("xr"), gxd=pl.col("gs") * pl.col("xd"))
    )
    return t, pg


def aoe2():
    """回傳既有口徑的對局樣本與玩家序列，供預測流程共用。"""
    D = "data/raw/aoe2/2026-11-06/"
    DAY_GAP = 12 * 3600
    T_EVAL = int(dt.datetime(2026, 6, 24, tzinfo=dt.timezone.utc).timestamp())

    m = pl.read_parquet(D + "matches.parquet").filter(pl.col("ladder") == "1v1").select("match_id", "start_time", "completion_time")
    p = pl.read_parquet(D + "players.parquet").join(m, on="match_id")
    p = p.filter((pl.col("won").cast(pl.Int32).sum().over("match_id") == 1) & (pl.len().over("match_id") == 2))
    pg = (
        p.select(
            "match_id", "start_time", "completion_time", "team", "old_rating", "new_rating",
            player="profile_id", score=pl.col("won").cast(pl.Float64), games_before=pl.col("career_wins") + pl.col("career_losses") - 1,
        )
        .sort("player", "start_time", "match_id")
        .with_columns(gap=pl.col("start_time") - pl.col("completion_time").shift(1).over("player"), chain=pl.col("old_rating") == pl.col("new_rating").shift(1).over("player"))
        .with_columns(day=(pl.col("gap").is_null() | (pl.col("gap") > DAY_GAP)).cum_sum().over("player"))
        .with_columns(
            first_of_day=pl.col("day") != pl.col("day").shift(1).over("player").fill_null(-1),
            day_start=pl.col("start_time").first().over("player", "day"),
            result=pl.when(pl.col("score") == 1).then(pl.lit("win")).otherwise(pl.lit("loss")),
        )
        .with_columns(
            has_prev=~pl.col("first_of_day") & pl.col("chain").fill_null(False) & (pl.col("gap") >= 0),
            day_complete=(pl.col("first_of_day") | pl.col("chain").fill_null(False)).all().over("player", "day"),
        )
        .with_columns(
            prev_win=pl.when("has_prev").then(pl.col("score").shift(1).over("player")).otherwise(0.0),
            missing=(~pl.col("has_prev")).cast(pl.Float64),
            dr=pl.when("has_prev").then((pl.col("old_rating") - pl.col("old_rating").shift(1).over("player")) / 100).otherwise(0.0),
            novice_side=pl.col("games_before") < novice.GAMES,
        )
    )
    pg = runs_before(pg, ["player", "day"])
    a, b = pg.filter(pl.col("team") == 0), pg.filter(pl.col("team") == 1)
    side = lambda x, s: x.select("match_id", *[pl.col(c).alias(f"{s}_{c}") for c in ["player", "old_rating", "score", "prev_win", "missing", "dr", "novice_side"]])  # noqa: E731
    t = (
        side(a, "a").join(side(b, "b"), on="match_id").join(m, on="match_id")
        .with_columns(
            game_id="match_id", a_id="a_player", b_id="b_player", y="a_score", speed=pl.lit("1v1"),
            train=pl.col("start_time") < T_EVAL, const=pl.lit(1.0),
            xr=((pl.col("a_old_rating") + pl.col("b_old_rating")) / 2 - 1500) / 100,
            xd=((pl.col("a_old_rating") - pl.col("b_old_rating")) / 100).cast(pl.Float64),
            p_sys=1 / (1 + 10 ** (-(pl.col("a_old_rating") - pl.col("b_old_rating")) / 400)),
            d_prev_win=pl.col("a_prev_win") - pl.col("b_prev_win"),
            d_missing=pl.col("a_missing") - pl.col("b_missing"),
            d_dr=pl.col("a_dr") - pl.col("b_dr"),
            novice=pl.col("a_novice_side") | pl.col("b_novice_side"),
        )
        .with_columns(xr2=pl.col("xr") ** 2, xdxr=pl.col("xd") * pl.col("xr"))
    )
    return t, pg


