# 連勝連敗：每局之前同一序列中連續的負局與勝局數。Lichess（streaks.py）與 OGS（other_games_ogs.py）共用，
# 兩邊的連串若各寫一份，「前 k - 1 局全負」的定義會漂移而不報錯。

import polars as pl


def runs_before(d, group):
    """d 依 group 與時間排序、含 result 欄（loss、draw、win）。加上 loss_run、win_run：本局之前、同一 group 內連續的負局與勝局數。
    和局打斷兩者。"""
    for v in ["loss", "win"]:
        d = (
            d.with_columns(_run=(pl.col("result") != v).cum_sum().over(group))
            .with_columns(_n=(pl.col("result") == v).cast(pl.Int32).cum_sum().over(group + ["_run"]))
            .with_columns(pl.col("_n").shift(1).over(group).fill_null(0).alias(f"{v}_run"))
            .drop("_run", "_n")
        )
    return d
