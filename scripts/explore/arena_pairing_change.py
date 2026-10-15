# arena 於 2015 年 4 月改依等級分配對：檢查改制在資料上是否看得到。
# 大廳對局不受此改制影響，作為同期的對照。

import subprocess

import polars as pl

rows = []
for m in ["2015-01", "2015-02", "2015-03", "2015-04", "2015-05", "2015-06"]:
    # 只取需要的四個表頭；PGN 中它們依 Event、UTCDate、WhiteElo、BlackElo 的順序出現
    cmd = f"zstdcat data/raw/lichess/lichess_db_standard_rated_{m}.pgn.zst | grep -E '^\\[(Event|UTCDate|WhiteElo|BlackElo) '"
    lines = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True).stdout.splitlines()
    vals = [l.split('"')[1] for l in lines]
    rows.append(pl.DataFrame({"event": vals[0::4], "date": vals[1::4], "w": vals[2::4], "b": vals[3::4]}))

df = (
    pl.concat(rows)
    .filter(pl.col("w") != "?", pl.col("b") != "?")
    .with_columns(
        tour=pl.col("event").str.contains("tournament"),
        date=pl.col("date").str.to_date("%Y.%m.%d"),
        gap=(pl.col("w").cast(pl.Int32) - pl.col("b").cast(pl.Int32)).abs(),
    )
)

daily = (
    df.group_by("date", "tour")
    .agg(n=pl.len(), gap_median=pl.col("gap").median(), gap_mean=pl.col("gap").mean())
    .sort("tour", "date")
)
with pl.Config(tbl_rows=-1):
    print(daily.filter(pl.col("date").is_between(pl.date(2015, 4, 10), pl.date(2015, 5, 10))))

weekly = (
    df.with_columns(week=pl.col("date").dt.truncate("1w"))
    .group_by("week", "tour")
    .agg(n=pl.len(), gap_median=pl.col("gap").median())
    .sort("tour", "week")
)
with pl.Config(tbl_rows=-1):
    print(weekly)
