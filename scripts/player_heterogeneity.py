# 玩家間的差異（規格見 docs/methods.md「玩家間的差異」）。
# 母體平均沒有效果不表示每個人都沒有。每位玩家的殘差對 h 的斜率，以逆變異數（正比於 Sxx，殘差變異數視為共同）加權後的變異
# Q = sum Sxx_p (b_p - b_bar)^2，與批內重新配對的虛無分布比較；虛無假設是沒有任何玩家受影響，所以殘差固定、只有 h 隨重抽變動。
# 同一次重抽也算加權平均斜率 b_bar（= 玩家內合併的斜率），用來分辨 Q 拒絕時是全體都有還是少數有。
# 樣本：Bullet 與 Blitz 上一局屬於兩局以上的批的列（第一局的前局效果），反應為不含 h 的直接效果模型的殘差（crossover_direct_resid.py）。

import sys

sys.path.insert(0, "scripts")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import likelihood as lk
import pairing_null
from crossover_direct import load

B = 2000
MIN_ROWS = 2
SP = ["Bullet", "Blitz"]


t, m, ns = load()
e_bar_m = m["e_bar"].to_numpy()
# 直接效果的殘差由 crossover_direct_resid.py 先算好
t = t.join(pl.read_parquet("data/processed/arena_direct_resid.parquet"), on=["game_id", "player", "speed"], how="left")

samples = []
size_rows = []
for s in SP:
    ts = t.filter(pl.col("speed") == s)
    r = ts["r_dir"].to_numpy()
    mk = ts["midx_lag1"].to_numpy()
    keep = mk >= 0
    pidx = np.unique(ts["player"].to_numpy()[keep], return_inverse=True)[1]
    n = np.bincount(pidx)
    # 資格只看列數，不看 h：h 隨重抽變動，依 h 篩選會使虛無分布的樣本不同
    ok = (n >= MIN_ROWS)[pidx]
    mk, r, pidx = mk[keep][ok], r[keep][ok], np.unique(pidx[ok], return_inverse=True)[1]
    P = pidx.max() + 1
    n = np.bincount(pidx)
    share = n / n.sum()
    size_rows.append({"speed": s, "rows": len(mk), "players": P, "effective_players": 1 / (share**2).sum()})
    samples.append((mk, pidx, r, P, n, np.bincount(pidx, r, P)))
sizes = pl.DataFrame(size_rows)
print(sizes)
sizes.write_csv("outputs/tables/player_heterogeneity_sample.csv")


def stat(e):
    hm = e_bar_m - e
    out = []
    for mk, pidx, r, P, n, sr in samples:
        h = hm[mk]
        sh = np.bincount(pidx, h, P)
        sxx = np.bincount(pidx, h * h, P) - sh**2 / n
        sxy = np.bincount(pidx, h * r, P) - sh * sr / n
        pos = sxx > 1e-12
        out += [(sxy[pos] ** 2 / sxx[pos]).sum() - sxy[pos].sum() ** 2 / sxx[pos].sum(), sxy[pos].sum() / sxx[pos].sum()]
    return np.array(out)


act = stat(ns["e_act"])
nd = {nl: np.array(v) for nl, v in pairing_null.run(ns, stat, B).items()}

rows = []
for nl in pairing_null.NULLS:
    for j, s in enumerate(SP):
        q, qn = act[2 * j], nd[nl][:, 2 * j]
        b, bn = act[2 * j + 1], nd[nl][:, 2 * j + 1]
        rows.append({
            "null": nl, "speed": s, "q": q, "q_null_mean": qn.mean(), "q_null_sd": qn.std(), "p_q": (1 + (qn >= q).sum()) / (B + 1),
            "mean_slope": b, "mean_slope_null_mean": bn.mean(), "mean_slope_null_sd": bn.std(),
            "p_mean_slope": (1 + (np.abs(bn - bn.mean()) >= abs(b - bn.mean())).sum()) / (B + 1),
        })
res = pl.DataFrame(rows)
# Q 的兩個速度類別為一族，Holm 法；以主虛無分布為準
main = (res["null"] == "main").to_numpy()
res = res.with_columns(p_q_holm=lk.holm(res["p_q"], main))
with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250):
    print(res)
res.write_csv("outputs/tables/player_heterogeneity.csv")

fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
for ax, j, s in zip(axes, range(2), SP):
    ax.hist(nd["main"][:, 2 * j], bins=40, color="#c9c9c4")
    ax.axvline(act[2 * j], color="#2a78d6", lw=1.5)
    ax.set_title(s)
    ax.set_xlabel("Q: inverse-variance weighted dispersion of player slopes")
axes[0].set_ylabel("re-pairings under the null")
fig.tight_layout()
fig.savefig("outputs/figures/player_heterogeneity.png", dpi=150)
