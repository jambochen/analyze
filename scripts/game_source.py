# 配對來源：arena 由月檔的錦標賽標記即可分出，一般大廳對局與指定邀戰只能靠 API 的 source。
# API 只查了 2014 年，2015 年的非錦標賽對局 source 為缺值。
# 匹配品質與校準只用系統配對或玩家自選對手的對局：邀戰的對手是指名的，
# 缺值的 2014 年對局疑為 pool 試行期，配對方式與等級分口徑未確認，兩者都不進主分析。

import polars as pl

KEEP = pl.col("source").is_in(["arena", "lobby"])


def with_source(df):
    """依 game_id 併入 source（arena、lobby、friend 或缺值）；df 須含 game_id 與布林欄 arena。"""
    meta = pl.scan_parquet("data/processed/game_meta.parquet").select("game_id", api_source="source")
    meta = meta if isinstance(df, pl.LazyFrame) else meta.collect()
    return (
        df.join(meta, on="game_id", how="left")
        .with_columns(source=pl.when(pl.col("arena")).then(pl.lit("arena")).otherwise(pl.col("api_source")))
        .drop("api_source")
    )
