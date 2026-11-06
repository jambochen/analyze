# 依狀態調整的勝率的修正量：各近況組合使線性預測式平移多少，換算成等級分並附區間。
# 修正項只以「自身減對手」與 opp_missing 進入線性預測式，所以修正量是係數的線性組合，
# 區間以 dyadic sandwich 的共變異數經 delta method 求得。主基準作 offset、視為已知：
# 它以全站對局估計，樣本遠大於 arena 的上半年，其估計誤差不另傳遞。
# 係數在 2014 上半年重估，與 state_adjusted_probs.py 存下的模型核對一致。

import os
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, "scripts")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import likelihood as lk
from state_adjusted import MODEL_DIR, PARAMS, XCOLS, ll, sample

SPEEDS = ["Bullet", "Blitz", "Classical"]
RESULTS = ["loss", "draw", "win"]
OPP = RESULTS + ["first"]

t = sample()
tr = t.filter(pl.col("start") < pl.datetime(2014, 7, 1, time_zone="UTC"))
te = t.filter(pl.col("start") >= pl.datetime(2014, 7, 1, time_zone="UTC"))
coef = pl.read_csv("outputs/tables/state_adjusted_probs_coef.csv")

rows = []
for spd in SPEEDS:
    kept = [PARAMS.index(p) for p in coef.filter(pl.col("speed") == spd)["param"]]
    d = tr.filter(pl.col("speed") == spd)
    args = (d["eta"].to_numpy(), d["w"].to_numpy(), d.select(XCOLS).to_numpy().astype(np.float64), d["y"].to_numpy())
    th, _, V, *_ = lk.fit(ll, np.zeros(len(PARAMS)), np.array(kept), args, d["player"].to_numpy(), d["opponent"].to_numpy())
    print(spd, "max diff vs saved", np.abs(th - np.load(f"{MODEL_DIR}/{spd}.npz")["th"]).max())
    names = [PARAMS[j] for j in kept]
    # 線性預測式每單位對應的等級分，取評估段各組 1 / slope_elo 的平均
    e = te.filter(pl.col("speed") == spd)
    per_eta = {nov: (1 / e.filter(pl.col("novice_rd") == nov)["slope_elo"]).mean() for nov in [False, True]}

    def add(label, own, opp, g):
        g = np.array([g.get(n, 0.0) for n in names])
        est, se = g @ th[kept], np.sqrt(g @ V @ g)
        for nov, k in per_eta.items():
            rows.append({"speed": spd, "contrast": label, "own_prev": own, "opp_prev": opp, "novice_rd": nov, "eta": est, "eta_se": se, "elo": est * k, "elo_se": se * k})

    for own in RESULTS:
        for opp in OPP:
            g = {"sym_prev_win": float((own == "win") - (opp == "win")), "sym_prev_draw": float((own == "draw") - (opp == "draw"))}
            if opp == "first":
                g["opp_missing"] = 1.0
            add("prev_result", own, opp, g)
    if "sym_dr" in names:
        # 雙方上一局等級分變動之差每 100 分；dr 以 100 分為單位
        add("dr_per_100", None, None, {"sym_dr": 1.0})

shift = pl.DataFrame(rows)
with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250):
    print(shift)
shift.write_csv("outputs/tables/state_adjusted_shift.csv")

fig, axes = plt.subplots(1, len(SPEEDS), figsize=(13, 4.5), sharex=True, sharey=True)
labels = [f"{o} / {p}" for o in RESULTS for p in OPP]
for ax, spd in zip(axes, SPEEDS):
    z = shift.filter(pl.col("speed") == spd)
    for nov, mk, col, dx in [(False, "o", "black", -0.15), (True, "s", "grey", 0.15)]:
        zz = z.filter(pl.col("novice_rd") == nov)
        zr = zz.filter(pl.col("contrast") == "prev_result")
        x = np.arange(zr.height) + dx
        ax.errorbar(zr["elo"], x, xerr=1.96 * zr["elo_se"], fmt=mk, color=col, ms=4, capsize=2, lw=1, label="novice (RD >= 110)" if nov else "established")
        zd = zz.filter(pl.col("contrast") == "dr_per_100")
        if zd.height:
            ax.errorbar(zd["elo"], [len(labels) + dx], xerr=1.96 * zd["elo_se"], fmt=mk, color=col, ms=4, capsize=2, lw=1)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_title(spd)
    ax.set_xlabel("adjustment to own rating (rating points)")
    ax.tick_params(labelbottom=True)
axes[0].set_yticks(range(len(labels) + 1), labels + ["rating change diff +100"])
axes[0].set_ylabel("own previous / opponent previous")
axes[0].invert_yaxis()
axes[0].legend(fontsize=8, loc="lower left")
fig.tight_layout()
fig.savefig("outputs/figures/state_adjusted_shift.png", dpi=150)
