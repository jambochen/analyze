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

from arena_batches import stop_rule_sample
from stop_rules import RULES, SPEEDS, backtest, pd_

B = 2000

d, del_days = stop_rule_sample()

sample = d.group_by("speed").agg(
    n=pl.len(), players=pl.col("player").n_unique(), days=pl.struct(pd_).n_unique(),
    novice_rd=pl.col("novice_rd").mean(), **{f"deleted_{r}": pl.col(f"del_{r}").sum() for r in RULES},
).sort("speed")
print(sample)
sample.write_csv("outputs/tables/folk_stop_rules_sample.csv")

res, reset, est, boot, col, rcol = backtest(d, del_days, B)
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

# 一天的切分門檻敏感度：6 與 24 小時各重跑一次，只看刪去減保留的方向與顯著與否（主結果為 12 小時）
gap_rows = []
for gap in [6, 12, 24]:
    dg, _ = stop_rule_sample(gap * 3600)
    pg = dg["player"].unique().sort().to_numpy()
    ig = np.searchsorted(pg, dg["player"].to_numpy())
    rg = dg["resid"].to_numpy()
    for r in RULES:
        for s in ["Bullet", "Blitz"]:
            sm = (dg["speed"] == s).to_numpy()
            dl = dg[f"del_{r}"].to_numpy()
            cols = [sm & dl, sm & ~dl]
            Sg = np.stack([np.bincount(ig[m], rg[m], len(pg)) for m in cols], 1)
            Ng = np.stack([np.bincount(ig[m], None, len(pg)) for m in cols], 1)
            rng = np.random.default_rng(0)
            W = rng.multinomial(len(pg), np.full(len(pg), 1 / len(pg)), size=B).astype(np.float64)
            bs = (W @ Sg) / (W @ Ng)
            diff = Sg.sum(0)[0] / Ng.sum(0)[0] - Sg.sum(0)[1] / Ng.sum(0)[1]
            bd = bs[:, 0] - bs[:, 1]
            se = np.nanstd(bd, ddof=1)
            gap_rows.append({"day_gap_hours": gap, "rule": r, "speed": s, "n_deleted": int(Ng[:, 0].sum()), "diff": diff, "se": se, "p": 2 * norm.sf(abs(diff) / se)})
gap_res = pl.DataFrame(gap_rows)
print(gap_res)
gap_res.write_csv("outputs/tables/folk_stop_rules_day_gap.csv")
