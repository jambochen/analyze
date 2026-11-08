# 全站校準樣本與 RD 有序模型；正式基準比較與時間外驗證共用。
import jax.numpy as jnp
import numpy as np
import polars as pl
import game_source
import novice
import ordinal
UTC = "UTC"
CLASSES = ["Bullet", "Blitz", "Classical"]
Q = ordinal.Q
EXP = ["0", "1-2", "3-5", "6-10", "11-20", "21-50", "51-100", "101-200", ">200"]


def sample():
    """2014 年可辨識來源的全站樣本，保留時間供向前切分。"""
    g_ = (
        pl.scan_parquet("data/processed/lichess_games.parquet")
        .filter(
            pl.col("speed").is_in(CLASSES), pl.col("base_seconds").is_not_null(),
            pl.col("start") >= pl.datetime(2014, 1, 1, time_zone=UTC), pl.col("start") < pl.datetime(2015, 1, 1, time_zone=UTC),
            pl.col("white_elo").is_not_null(), pl.col("black_elo").is_not_null(),
        )
        .select(
            "game_id", "start", "speed", "white_elo", "black_elo",
            y=pl.col("result").replace_strict({"0-1": 0, "1/2-1/2": 1, "1-0": 2}, default=None, return_dtype=pl.Int8),
            w="white_id", b="black_id",
            arena=pl.col("tournament_id").is_not_null(),
        )
        .filter(pl.col("y").is_not_null())
    )
    rp = pl.scan_parquet("data/processed/glicko_replay.parquet").select("game_id", "perf", "white_rd", "black_rd", "white_left_truncated", "black_left_truncated", "white_prior_games", "black_prior_games")
    d = (
        game_source.with_source(g_.join(rp, on="game_id"))
        # 邀戰與無 source 的對局不進校準，理由見 game_source.py
        .filter(pl.col("perf") == pl.col("speed"), game_source.KEEP)
        .with_columns(
            diff=(pl.col("white_elo") - pl.col("black_elo")).cast(pl.Float64),
            gs=ordinal.g("white_rd", "black_rd"),
            rd_max=pl.max_horizontal("white_rd", "black_rd"),
            lt_any=pl.col("white_left_truncated") | pl.col("black_left_truncated"),
        )
        .with_columns(
            xr=((pl.col("white_elo") + pl.col("black_elo")) / 2 - 1500) / 100,
            xd=pl.col("diff") / 100,
            gxd=pl.col("gs") * pl.col("diff") / 100,
            score=pl.col("y") / 2,
            L_elo=Q * pl.col("diff"),
            L_glicko=Q * pl.col("gs") * pl.col("diff"),
            novice_rd=novice.novice_rd("white_rd", "black_rd"),
            train=pl.col("start") < pl.datetime(2014, 7, 1, time_zone=UTC),
            # 分層與 calibration.py 相同；重算前後以 8 月 1 日為界
            period=pl.when(pl.col("start") < pl.datetime(2014, 8, 1, time_zone=UTC)).then(pl.lit("Jan-Jul")).otherwise(pl.lit("Aug-Dec")),
            # 分組與 calibration.py 相同；局數取自重放，從帳號在該速度類別的第一局起算
            exp_bin=pl.min_horizontal("white_prior_games", "black_prior_games").cut([0, 2, 5, 10, 20, 50, 100, 200], labels=EXP, left_closed=False).cast(pl.String),
            band=((pl.col("white_elo") + pl.col("black_elo")) / 2).cut([1200, 1400, 1600, 1800, 2000], labels=["<1200", "1200-1400", "1400-1600", "1600-1800", "1800-2000", ">=2000"]).cast(pl.String),
        )
        # streaming 的 join 輸出列序每個行程不同，加總順序隨之變動；先排序才可重現
        .sort("game_id")
        .select("game_id", "start", "speed", "w", "b", "y", "score", "xr", "xd", "gxd", "L_elo", "L_glicko", "rd_max", "novice_rd", "lt_any", "train", "source", "period", "band", "exp_bin")
        .collect(engine="streaming")
    )
    return d


def ll_ord(t, xr, xd, gxd, y):
    eta = t[0] + t[1] * xr + t[2] * xr**2 + (t[3] + t[4] * xr) * xd + t[8] * gxd
    return ordinal.ll(eta, jnp.exp(t[5] + t[6] * xr + t[7] * xr**2), y)


def probs(t, xr, xd, gxd):
    eta = t[0] + t[1] * xr + t[2] * xr**2 + (t[3] + t[4] * xr) * xd + t[8] * gxd
    return ordinal.probs(eta, np.exp(t[5] + t[6] * xr + t[7] * xr**2))
