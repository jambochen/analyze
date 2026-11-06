# 預測改善的來源：Brier 分數的 Murphy 分解（規格見 docs/methods.md「預測改善的來源：Brier 分數的分解」）。
# 預報依寬 0.1 分組；Brier = 可靠度 - 解析度 + 不確定度 + 組內預報變異 + 組內共變異項（Stephenson 等人的廣義分解），
# 後兩項使等式在分組下精確成立。Lichess 的預報是期望得分、結果是得分（和局記 0.5），OGS 是勝率與勝負。只作描述。

import os
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, "scripts")

import numpy as np
import polars as pl

from state_adjusted import eval_rows

BASES = ["e_sys", "e_base", "e_adj"]

d = pl.concat([
    eval_rows().filter(pl.col("speed").is_in(["Bullet", "Blitz"])).with_columns(game=pl.lit("lichess")),
    pl.read_parquet("data/processed/ogs_eval_rows.parquet").with_columns(game=pl.lit("ogs")),
    pl.read_parquet("data/processed/aoe2_eval_rows.parquet").with_columns(game=pl.lit("aoe2")),
], how="diagonal_relaxed")

rows = []
for (game, spd), s in d.group_by("game", "speed", maintain_order=True):
    for group, sg in [("all", s), ("novice", s.filter("novice")), ("established", s.filter(~pl.col("novice")))]:
        y = sg["score"].to_numpy()
        ybar = y.mean()
        for b in BASES:
            f = sg[b].to_numpy()
            k = np.minimum((f * 10).astype(int), 9)
            n = np.bincount(k, minlength=10)
            ok = n > 0
            fk = np.bincount(k, f, 10)[ok] / n[ok]
            yk = np.bincount(k, y, 10)[ok] / n[ok]
            N = len(y)
            rel = (n[ok] * (fk - yk) ** 2).sum() / N
            res = (n[ok] * (yk - ybar) ** 2).sum() / N
            unc = ((y - ybar) ** 2).mean()
            wbv = ((f - (np.bincount(k, f, 10) / np.maximum(n, 1))[k]) ** 2).mean()
            wbc = -2 * ((f - (np.bincount(k, f, 10) / np.maximum(n, 1))[k]) * (y - (np.bincount(k, y, 10) / np.maximum(n, 1))[k])).mean()
            bs = ((y - f) ** 2).mean()
            assert abs(bs - (rel - res + unc + wbv + wbc)) < 1e-10
            rows.append({"game": game, "speed": spd, "group": group, "forecast": b, "n": N, "brier": bs, "reliability": rel, "resolution": res, "uncertainty": unc, "within_bin_variance": wbv, "within_bin_covariance": wbc})
out = pl.DataFrame(rows)
with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250):
    print(out)
out.write_csv("outputs/tables/forecast_decomposition.csv")
