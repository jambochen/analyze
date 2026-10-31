# gam_glmm.R 的逐列損失：各模型的下半年對數分數與 ranked probability score，差值的標準誤用 dyadic sandwich。
import sys

sys.path.insert(0, "scripts")

import polars as pl

import likelihood as lk

rows = []
for sp in ["blitz", "bullet"]:
    # R 輸出的 player 欄已把未見過的玩家指到同一個水準，真正的編號按原列順序從匯出檔取回
    test = pl.read_csv(f"data/processed/explore_gam_glmm_{sp}.csv", columns=["player", "train"]).filter(pl.col("train") == 0)
    d = pl.read_csv(f"data/processed/explore_gam_glmm_loss_{sp}.csv").with_columns(player=test["player"])
    w, b = d["player"].to_numpy(), d["opponent"].to_numpy()
    for metric in ["ll", "rps"]:
        for m in ["M0", "M1", "M2", "M3", "M4"]:
            x = d[f"{metric}_{m}"].to_numpy()
            for ref in ["M0", "M1"]:
                if m == ref:
                    continue
                dlt = x - d[f"{metric}_{ref}"].to_numpy()
                se = lk.dyad_se((dlt - dlt.mean()) / len(dlt), w, b)
                rows.append({"speed": sp, "metric": metric, "model": m, "vs": ref, "mean": x.mean(), "diff": dlt.mean(), "se": se, "z": dlt.mean() / se})
res = pl.DataFrame(rows)
with pl.Config(tbl_rows=-1, tbl_width_chars=200):
    print(res)
res.write_csv("outputs/tables/explore_gam_glmm.csv")
