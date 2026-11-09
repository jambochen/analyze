# 時間外分析的局前特徵與固定基準預測；配適與成果解讀共用，避免重建口徑漂移。
import numpy as np
import polars as pl

import external_sample
import ordinal
import state_adjusted as sa

BASE = ["const", "xr", "xr2", "xd", "xdxr", "gxd"]


def sample(game):
    """既有時間外設計的玩家局／對局、固定時間切分及近況候選；不估計模型。"""
    if game == "Lichess":
        t = sa.sample("2014").with_columns(
            time=pl.col("start").dt.epoch("s"), a_id=pl.col("player"), b_id=pl.col("opponent"),
            novice=pl.col("novice_rd"), sign=pl.when(pl.col("white") == 1).then(1.0).otherwise(-1.0),
            xr=((pl.col("elo") + pl.col("opp_elo")) / 2 - 1500) / 100,
            xd=(pl.col("elo") - pl.col("opp_elo")) / 100,
            gxd=ordinal.g("rd", "opp_rd") * (pl.col("elo") - pl.col("opp_elo")) / 100,
            rd_dr=pl.col("dr") * pl.col("rd") / 100 - pl.col("opp_dr") * pl.col("opp_rd") / 100,
            rd_abs_dr=pl.col("dr").abs() * pl.col("rd") / 100 - pl.col("opp_dr").abs() * pl.col("opp_rd") / 100,
            p_sys=1 / (1 + 10 ** (-(pl.col("elo") - pl.col("opp_elo")) / 400)),
        )
        # 不讓 sample 內全年基準的欄位流入本次配適。
        t = t.select("game_id", "time", "speed", "a_id", "b_id", "y", "novice", "sign", "xr", "xd", "gxd", "p_sys", "d_prev_draw", "d_prev_win", "opp_missing", "d_dr", "rd_dr", "rd_abs_dr", "elo", "opp_elo", "rd", "opp_rd", "dr", "opp_dr", "prev_result", "opp_prev_result")
        folds = [("Jul-Sep", "2014-06-01", "2014-07-01", "2014-10-01"), ("Oct-Dec", "2014-09-01", "2014-10-01", "2015-01-01")]
        state_cols = ["d_prev_draw", "d_prev_win", "opp_missing"]
    else:
        raw, pg = external_sample.ogs() if game == "OGS" else external_sample.aoe2()
        del pg
        if game == "OGS":
            t = raw.rename({"black_id": "a_id", "white_id": "b_id", "expected_black": "p_sys"}).with_columns(
                time=pl.col("ended"),
                rd_dr=pl.col("b_dr") * pl.col("black_rd") / 100 - pl.col("w_dr") * pl.col("white_rd") / 100,
                rd_abs_dr=pl.col("b_dr").abs() * pl.col("black_rd") / 100 - pl.col("w_dr").abs() * pl.col("white_rd") / 100,
            )
            folds = [("2023", "2022-01-01", "2023-01-01", "2024-01-01")]
        else:
            t = raw.with_columns(time=pl.col("start_time"))
            folds = [("Jun24-Jul03", "2026-06-17", "2026-06-24", "2026-07-04")]
        t = t.with_columns(y=(pl.col("y") * 2).cast(pl.Int64))
        state_cols = ["d_prev_win", "d_missing"]
        del raw
    return t, folds, state_cols


def offset(d, th, game):
    """套用已給定的基準係數，覆寫 eta 與 width，不配適、不選項。"""
    if game == "Lichess":
        xr, xd, gx, sign = [d[c].to_numpy() for c in ["xr", "xd", "gxd", "sign"]]
        eta = sign * (th[0] + th[1] * xr + th[2] * xr**2) + (th[3] + th[4] * xr) * xd + th[8] * gx
        width = np.exp(th[5] + th[6] * xr + th[7] * xr**2)
    else:
        eta = d.select(BASE if game == "OGS" else BASE[:-1]).to_numpy().astype(float) @ th
        width = np.zeros(d.height)
    return d.with_columns(eta=pl.Series(eta), width=pl.Series(width))

