# 連勝連敗是否超出期望（規格見 docs/methods.md「連勝連敗是否超出期望」）。
# 前 K - 1 局全負（全勝）的局，其得分減期望得分的平均。一局是否納入只取決於之前的結果，期望勝率正確時殘差是
# martingale 差分，玩家何時停手不影響期望為 0，不需要續下模型；直接平均殘差，不經過每條序列的比例，沒有 Miller–Sanjurjo 偏誤。
# 基準依序為系統的期望勝率、主基準、依狀態調整的勝率，看超額是否在最後一個基準下消失。

import os
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, "scripts")

import polars as pl

import likelihood as lk
from state_adjusted import eval_rows

K = 5
BASES = ["e_sys", "e_base", "e_adj"]

d = pl.concat([
    eval_rows().filter(pl.col("speed").is_in(["Bullet", "Blitz"])).with_columns(game=pl.lit("lichess")),
    pl.read_parquet("data/processed/ogs_eval_rows.parquet").with_columns(game=pl.lit("ogs")),
    pl.read_parquet("data/processed/aoe2_eval_rows.parquet").with_columns(game=pl.lit("aoe2")),
], how="diagonal_relaxed")
# AoE2 的快照漏收場次，連串只在一天內每一場都接得上的天有定義（other_games_aoe2.py）
d = d.filter(pl.col("day_complete").fill_null(True))

rows = []
for (game, spd), s in d.group_by("game", "speed", maintain_order=True):
    for streak, col in [("losses", "loss_run"), ("wins", "win_run")]:
        x = s.filter(pl.col(col) >= K - 1)
        for b in BASES:
            r = (x["score"] - x[b]).to_numpy()
            se = lk.dyad_se((r - r.mean()) / len(r), x["player"].to_numpy(), x["opponent"].to_numpy())
            rows.append({"game": game, "speed": spd, "after": f"{K - 1}_{streak}", "baseline": b, "n": len(r), "mean_resid": r.mean(), "se_dyadic": se, "lo": r.mean() - 1.96 * se, "hi": r.mean() + 1.96 * se,
                         # 連串的局中弱方的比例：系統的期望勝率不校準時，殘差的正負隨強弱方而定
                         "share_underdog_sys": (x["e_sys"] < 0.5).mean()})
out = pl.DataFrame(rows)
with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250):
    print(out)
out.write_csv("outputs/tables/streaks.csv")
