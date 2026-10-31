# GLMM 只對訓練期出現過的玩家有作用：未見過的玩家隨機截距取 0，等於 M1。這裡分開看已見過與未見過的列。
import sys

sys.path.insert(0, "scripts")

import polars as pl

import likelihood as lk

rows = []
for sp in ["blitz", "bullet"]:
    raw = pl.read_csv(f"data/processed/explore_gam_glmm_{sp}.csv", columns=["player", "train", "game_id"])
    seen_players = raw.filter(pl.col("train") == 1)["player"].unique()
    # R 輸出的 player 欄已把未見過的玩家指到同一個水準，真正的編號按原列順序從匯出檔取回
    test = pl.read_csv(f"data/processed/explore_gam_glmm_{sp}.csv", columns=["player", "train"]).filter(pl.col("train") == 0)
    d = pl.read_csv(f"data/processed/explore_gam_glmm_loss_{sp}.csv").with_columns(player=test["player"]).with_columns(seen=pl.col("player").is_in(seen_players.implode()))
    for grp, g in [("seen", d.filter(pl.col("seen"))), ("unseen", d.filter(~pl.col("seen")))]:
        w, b = g["player"].to_numpy(), g["opponent"].to_numpy()
        for m in ["M2", "M3", "M4"]:
            dlt = (g[f"ll_{m}"] - g["ll_M1"]).to_numpy()
            se = lk.dyad_se((dlt - dlt.mean()) / len(dlt), w, b)
            rows.append({"speed": sp, "group": grp, "model": m, "vs": "M1", "n": g.height, "diff": dlt.mean(), "se": se, "z": dlt.mean() / se})
res = pl.DataFrame(rows)
with pl.Config(tbl_rows=-1, tbl_width_chars=160):
    print(res)
res.write_csv("outputs/tables/explore_gam_glmm_seen.csv")
