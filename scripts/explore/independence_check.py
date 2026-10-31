# 主基準殘差的獨立性：玩家內的相關（ICC、相鄰局自相關）、同一局兩位玩家的相關，
# 以及把對局當獨立的標準誤相對玩家分群、dyadic sandwich 的設計效應。

import sys

sys.path.insert(0, "scripts")

import numpy as np
import polars as pl

import likelihood as lk
from arena_batches import player_sample_rd

d = player_sample_rd().sort("player", "start", "game_id")
rows = []
for s in ["Bullet", "Blitz", "Classical"]:
    z = d.filter(pl.col("speed") == s)
    x = z["resid"].to_numpy()
    n = len(x)
    _, inv = np.unique(z["player"].to_numpy(), return_inverse=True)
    k = inv.max() + 1
    cnt = np.bincount(inv)
    # 玩家視角的殘差在同一局兩人正負相反，平均恆為 0；變異數的比較改在每局一列的白方視角：
    # 白方殘差的平均（白方優勢的偏差）以局為單位，玩家分群時同一玩家兼執白與執黑
    g = z.filter(pl.col("color") == "white")
    xg = g["resid"].to_numpy()
    ng = len(xg)
    wg, bg = g["player"].to_numpy(), g["opponent"].to_numpy()
    phi = (xg - xg.mean()) / ng
    se_iid = np.sqrt((phi**2).sum())
    pl_all = np.unique(np.concatenate([wg, bg]), return_inverse=True)[1]
    wi, bi = pl_all[:ng], pl_all[ng:]
    kk = pl_all.max() + 1
    # 每局的影響量分給白、黑兩位玩家所屬的群（同一玩家兼執黑白時相加）
    se_player = np.sqrt(((np.bincount(wi, phi, kk) + np.bincount(bi, phi, kk)) ** 2).sum())
    se_dyad = lk.dyad_se(phi, wg, bg)
    # 單因子 ICC（ANOVA 估計，群大小不等時用 n0）
    m = np.bincount(inv, x) / cnt
    ssb = (cnt * (m - x.mean()) ** 2).sum()
    ssw = ((x - m[inv]) ** 2).sum()
    msb, msw = ssb / (k - 1), ssw / (n - k)
    n0 = (n - (cnt**2).sum() / n) / (k - 1)
    icc = (msb - msw) / (msb + (n0 - 1) * msw)
    # 同一玩家相鄰兩局殘差的相關
    same = (z["player"].to_numpy()[1:] == z["player"].to_numpy()[:-1])
    ac = np.corrcoef(x[1:][same], x[:-1][same])[0, 1]
    # 同一局兩位玩家殘差的相關
    pair = z.group_by("game_id").agg(a=pl.col("resid").first(), b=pl.col("resid").last(), c=pl.len()).filter(pl.col("c") == 2)
    rg = np.corrcoef(pair["a"].to_numpy(), pair["b"].to_numpy())[0, 1]
    rows.append(dict(speed=s, n=n, players=k, eff_players=1 / ((cnt / n) ** 2).sum(), white_resid=xg.mean(), icc=icc, lag1=ac, within_game=rg,
                     se_iid=se_iid, se_player=se_player, se_dyadic=se_dyad, deff_player=(se_player / se_iid) ** 2, deff_dyadic=(se_dyad / se_iid) ** 2))
out = pl.DataFrame(rows)
with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250):
    print(out)
out.write_csv("outputs/tables/explore_independence_check.csv")
