# 檢查 arena 是否隨機配對：同一場錦標賽內打亂黑方，若配對不看等級分，
# 實際分差應與打亂後的分差分布一致。大廳對局在同一天內打亂作對照。

import subprocess

import numpy as np
import polars as pl

rng = np.random.default_rng(0)

for m in ["2014-08", "2015-03", "2015-05"]:
    cmd = f"zstdcat data/raw/lichess/lichess_db_standard_rated_{m}.pgn.zst | grep -E '^\\[(Event|UTCDate|WhiteElo|BlackElo) '"
    lines = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True).stdout.splitlines()
    vals = [l.split('"')[1] for l in lines]
    df = (
        pl.DataFrame({"event": vals[0::4], "date": vals[1::4], "w": vals[2::4], "b": vals[3::4]})
        .filter(pl.col("w") != "?", pl.col("b") != "?")
        .with_columns(pl.col("w").cast(pl.Int32), pl.col("b").cast(pl.Int32))
        .with_columns(
            tour=pl.col("event").str.contains("tournament"),
            # 錦標賽以場次分組；大廳以日期與時間類別分組
            grp=pl.when(pl.col("event").str.contains("tournament"))
            .then(pl.col("event"))
            .otherwise(pl.col("date") + pl.col("event")),
        )
    )
    for tour in [True, False]:
        d = df.filter(pl.col("tour") == tour)
        actual = (d["w"] - d["b"]).abs().median()
        g = d["grp"].to_numpy()
        b = d["b"].to_numpy()
        order = np.argsort(g, kind="stable")
        bounds = np.flatnonzero(np.r_[True, g[order][1:] != g[order][:-1], True])
        shuffled = []
        for _ in range(20):
            bs = b[order].copy()
            for i, j in zip(bounds[:-1], bounds[1:]):
                rng.shuffle(bs[i:j])
            perm = np.empty_like(bs)
            perm[order] = bs
            shuffled.append(np.median(np.abs(d["w"].to_numpy() - perm)))
        # 組內相關：扣掉組平均後，白方與黑方等級分的相關
        r = d.with_columns(
            wd=pl.col("w") - pl.col("w").mean().over("grp"),
            bd=pl.col("b") - pl.col("b").mean().over("grp"),
        ).select(pl.corr("wd", "bd")).item()
        print(m, "tournament" if tour else "lobby", "n", d.height, "actual median gap", actual,
              "shuffled", np.min(shuffled), np.max(shuffled), "within-group corr", round(r, 3))
