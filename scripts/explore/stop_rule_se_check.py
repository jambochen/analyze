# 停止規則回測的玩家 block bootstrap 標準誤，與解析的玩家分群 sandwich、dyadic sandwich 對照。
# bootstrap 只以玩家為群，同一局的兩位玩家分在不同的群；兩人的殘差正負相反，
# 這個跨群的相關使差值的變異數變大或變小，取決於兩人的局各自被刪或保留，方向事先不知道

import sys

sys.path.insert(0, "scripts")

import numpy as np
import polars as pl

import likelihood as lk
from arena_batches import STOP_RULES, stop_rule_sample

SPEEDS = ["Bullet", "Blitz", "Classical"]

d, _ = stop_rule_sample()
ref = pl.read_csv("outputs/tables/folk_stop_rules_backtest.csv").filter(pl.col("group") == "all").select("rule", "speed", se_boot="se")
rows = []
for r in STOP_RULES:
    for s in SPEEDS:
        z = d.filter(pl.col("speed") == s)
        x, dl = z["resid"].to_numpy(), z[f"del_{r}"].to_numpy()
        nd, nk = dl.sum(), (~dl).sum()
        md, mk = x[dl].mean(), x[~dl].mean()
        # 兩個比率估計量之差的影響函數
        phi = np.where(dl, (x - md) / nd, -(x - mk) / nk)
        w, b = z["player"].to_numpy(), z["opponent"].to_numpy()
        _, inv = np.unique(w, return_inverse=True)
        se_player = np.sqrt((np.bincount(inv, phi) ** 2).sum())
        # 兩位玩家都在樣本內的局，佔全部局的比例
        both = z.select("game_id").group_by("game_id").len()["len"].eq(2).mean()
        rows.append({"rule": r, "speed": s, "diff": md - mk, "se_player": se_player, "se_dyadic": lk.dyad_se(phi, w, b), "games_both_sides": both})
out = pl.DataFrame(rows).join(ref, on=["rule", "speed"])
with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250):
    print(out)
out.write_csv("outputs/tables/explore_stop_rule_se_check.csv")
