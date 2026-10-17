# 民間停止規則的回測（規格見 docs/methods.md「停止規則」）。
# 一天之內，「一路下到第一次輸」與「連輸兩局才停」各自刪去觸發之後的局；比較刪去與保留的局的平均殘差。
# 期望勝率準確且無 carryover 時殘差是 martingale 差分，刪去與保留的局期望相同，任何停止規則都無用。
# 被刪去的局只出現在輸了還繼續下的人身上，正是採用規則後行為會改變的人，所以不需要續下的模型。

import sys

sys.path.insert(0, "scripts")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import norm

from arena_batches import STOP_RULES, stop_rule_sample

SPEEDS = ["Bullet", "Blitz", "Classical"]
RULES = STOP_RULES
B = 2000
pd_ = ["player", "day"]

d, del_days = stop_rule_sample()

sample = d.group_by("speed").agg(
    n=pl.len(), players=pl.col("player").n_unique(), days=pl.struct(pd_).n_unique(),
    novice_rd=pl.col("novice_rd").mean(), **{f"deleted_{r}": pl.col(f"del_{r}").sum() for r in RULES},
).sort("speed")
print(sample)
sample.write_csv("outputs/tables/folk_stop_rules_sample.csv")

# 每位玩家各格的殘差和與局數；bootstrap 以玩家為單位重抽，對這兩張矩陣加權即可
players = d["player"].unique().sort()
P = len(players)
pi = np.searchsorted(players.to_numpy(), d["player"].to_numpy())
cells = []
resid = d["resid"].to_numpy()
spd = d["speed"].to_numpy()
nov = d["novice_rd"].to_numpy()
for r in RULES:
    dl = d[f"del_{r}"].to_numpy()
    for s in SPEEDS:
        for g, gm in [("all", np.ones(len(d), bool)), ("novice", nov), ("established", ~nov)]:
            m = (spd == s) & gm
            cells += [(r, s, g, "deleted", m & dl), (r, s, g, "kept", m & ~dl), (r, s, g, "all", m)]
# 隔天重置：有刪去局的天，其刪去局對下一個 arena 天的第一局
dd = d.join(del_days, on=pd_)
dpi = np.searchsorted(players.to_numpy(), dd["player"].to_numpy())
nd = del_days.filter(pl.col("next_resid").is_not_null())
npi = np.searchsorted(players.to_numpy(), nd["player"].to_numpy())
reset_cells = []
for r in RULES:
    for s in SPEEDS:
        reset_cells.append((r, s, "deleted", dpi, dd["resid"].to_numpy(), (dd[f"del_{r}"] & (dd["speed"] == s)).to_numpy()))
        reset_cells.append((r, s, "next_day_first", npi, nd["next_resid"].to_numpy(), (nd[f"any_{r}"] & (nd["next_speed"] == s)).to_numpy()))

S = np.zeros((P, len(cells) + len(reset_cells)))
N = np.zeros_like(S)
for k, (*_, m) in enumerate(cells):
    S[:, k] = np.bincount(pi[m], resid[m], P)
    N[:, k] = np.bincount(pi[m], None, P)
for j, (_, _, _, idx, x, m) in enumerate(reset_cells):
    k = len(cells) + j
    S[:, k] = np.bincount(idx[m], x[m], P)
    N[:, k] = np.bincount(idx[m], None, P)

rng = np.random.default_rng(0)
boot = []
for c in range(0, B, 100):
    W = rng.multinomial(P, np.full(P, 1 / P), size=min(100, B - c)).astype(np.float64)
    boot.append((W @ S) / (W @ N))
boot = np.concatenate(boot)
est = S.sum(0) / N.sum(0)
col = {(r, s, g, k): i for i, (r, s, g, k, _) in enumerate(cells)}
rcol = {(r, s, k): len(cells) + j for j, (r, s, k, *_) in enumerate(reset_cells)}


def summarize(a, b, extra):
    diff, bd = est[a] - est[b], boot[:, a] - boot[:, b]
    se = np.nanstd(bd, ddof=1)
    return extra | {"diff": diff, "se": se, "lo": np.nanquantile(bd, 0.025), "hi": np.nanquantile(bd, 0.975), "p": 2 * norm.sf(abs(diff) / se)}


rows = []
for r in RULES:
    for s in SPEEDS:
        for g in ["all", "novice", "established"]:
            i_d, i_k, i_a = col[(r, s, g, "deleted")], col[(r, s, g, "kept")], col[(r, s, g, "all")]
            row = summarize(i_d, i_k, {"rule": r, "speed": s, "group": g, "n_deleted": int(N[:, i_d].sum()), "n_kept": int(N[:, i_k].sum()), "mean_deleted": est[i_d], "mean_kept": est[i_k]})
            bv = boot[:, i_k] - boot[:, i_a]
            row |= {"kept_minus_never": est[i_k] - est[i_a], "kept_minus_never_se": np.nanstd(bv, ddof=1)}
            rows.append(row)
res = pl.DataFrame(rows)
fam = (res["group"] == "all") & res["speed"].is_in(["Bullet", "Blitz"])
p = res["p"].to_numpy()
ph = np.full(len(p), np.nan)
fi = np.flatnonzero(fam.to_numpy())
o = fi[np.argsort(p[fi])]
ph[o] = np.minimum(1, np.maximum.accumulate(p[o] * (len(o) - np.arange(len(o)))))
res = res.with_columns(p_holm=pl.Series(ph))

reset = pl.DataFrame([
    summarize(rcol[(r, s, "next_day_first")], rcol[(r, s, "deleted")], {
        "rule": r, "speed": s, "n_deleted": int(N[:, rcol[(r, s, "deleted")]].sum()), "n_next_day_first": int(N[:, rcol[(r, s, "next_day_first")]].sum()),
        "mean_deleted": est[rcol[(r, s, "deleted")]], "mean_next_day_first": est[rcol[(r, s, "next_day_first")]],
    })
    for r in RULES for s in SPEEDS
])
with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250):
    print(res)
    print(reset)
res.write_csv("outputs/tables/folk_stop_rules_backtest.csv")
reset.write_csv("outputs/tables/folk_stop_rules_reset.csv")

fig, axes = plt.subplots(1, len(RULES), figsize=(11, 4), sharey=True)
for ax, r in zip(axes, RULES):
    for j, (k, lab) in enumerate([("kept", "kept"), ("deleted", "deleted"), ("next_day_first", "next day, first game")]):
        idx = [rcol[(r, s, k)] if k == "next_day_first" else col[(r, s, "all", k)] for s in SPEEDS]
        lo = np.nanquantile(boot[:, idx], 0.025, axis=0)
        hi = np.nanquantile(boot[:, idx], 0.975, axis=0)
        x = np.arange(len(SPEEDS)) + (j - 1) * 0.25
        ax.errorbar(x, est[idx], yerr=[est[idx] - lo, hi - est[idx]], fmt="o", capsize=3, label=lab)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xticks(range(len(SPEEDS)), SPEEDS)
    ax.set_title({"first_loss": "stop at first loss", "two_losses": "stop after two straight losses"}[r])
axes[0].set_ylabel("mean score minus expected (main baseline)")
axes[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig("outputs/figures/folk_stop_rules_backtest.png", dpi=150)
