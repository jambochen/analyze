# gam_decomp.R 的逐列損失對 M1 的差，分已見過與未見過的玩家。
import sys

sys.path.insert(0, "scripts")

import polars as pl

import likelihood as lk

rows = []
for sp in ["blitz", "bullet"]:
    raw = pl.read_csv(f"data/processed/explore_gam_glmm_{sp}.csv", columns=["player", "opponent", "train"])
    seen_players = raw.filter(pl.col("train") == 1)["player"].unique()
    test = raw.filter(pl.col("train") == 0)
    m1 = pl.read_csv(f"data/processed/explore_gam_glmm_loss_{sp}.csv")["ll_M1"]
    dec = pl.read_csv(f"data/processed/explore_gam_decomp_loss_{sp}.csv")
    d = test.with_columns(ll_M1=m1, seen=pl.col("player").is_in(seen_players.implode())).hstack(dec.drop("game_id"))
    for grp, g in [("all", d), ("seen", d.filter(pl.col("seen"))), ("unseen", d.filter(~pl.col("seen")))]:
        for m in [c for c in dec.columns if c.startswith("ll_D")]:
            dlt = (g[m] - g["ll_M1"]).to_numpy()
            se = lk.dyad_se((dlt - dlt.mean()) / len(dlt), g["player"].to_numpy(), g["opponent"].to_numpy())
            rows.append({"speed": sp, "group": grp, "model": m[3:], "vs": "M1", "n": g.height, "diff": dlt.mean(), "se": se, "z": dlt.mean() / se})
res = pl.DataFrame(rows)
with pl.Config(tbl_rows=-1, tbl_width_chars=160):
    print(res)
res.write_csv("outputs/tables/explore_gam_decomp.csv")
