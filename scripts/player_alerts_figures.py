# 玩家提醒的頻率、後續偏差與提醒對象分開呈現，避免把預測關聯當休息效果。
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

s = pl.read_csv("outputs/tables/player_alerts_evaluation.csv").filter(pl.col("group") == "all", pl.col("model") == "recent")
o = pl.read_csv("outputs/tables/player_alerts_overlap.csv")
plt.rcParams.update({"svg.fonttype": "path", "font.size": 10})
keys = s.select("speed", "fold").unique().sort("speed", "fold").rows()
rules = ["loss3", "loss4", "adjusted_fixed", "adjusted_matched"]
labels = ["3+ losses", "4+ losses", "Adjusted: fixed cutoff", "Adjusted: matched train rate"]
colors = ["#c98530", "#865aa3", "#307f92", "#557b43"]
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
for j, (rule, label, color) in enumerate(zip(rules, labels, colors)):
    d = s.filter(pl.col("rule") == rule).sort("speed", "fold")
    y = np.arange(len(keys)) + (j - 1.5) * .17
    axes[0].plot(d["alert_rate"].to_numpy() * 100, y, "o", color=color, label=label)
    for ax, col, lo, hi in [(axes[1], "residual_alert", "residual_alert_lo", "residual_alert_hi"), (axes[2], "equal_player_residual_alert", "equal_player_alert_lo", "equal_player_alert_hi")]:
        x = d[col].to_numpy() * 100
        ax.errorbar(x, y, xerr=np.array([x - d[lo].to_numpy() * 100, d[hi].to_numpy() * 100 - x]), fmt="o", color=color, capsize=3)
for ax, label in zip(axes, ["Alerted eligible player-games (%)", "Next score minus recent-model expectation\n(game-weighted, percentage points)", "Next score minus recent-model expectation\n(equal player weights, percentage points)"]):
    ax.set(yticks=np.arange(len(keys)), yticklabels=[f"{sp}: {fo}" for sp, fo in keys], xlabel=label)
    ax.axvline(0, color="grey", linestyle="--", linewidth=.8)
    ax.grid(axis="x", alpha=.2)
axes[0].invert_yaxis()
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, .93), ncol=4, fontsize=9)
fig.suptitle("Player alerts: continuers with at least five prior arena games")
fig.tight_layout(rect=(0, 0, 1, .83))
for ext in ["svg", "png"]:
    fig.savefig(f"outputs/figures/player_alerts_evaluation.{ext}", dpi=160)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
left = np.zeros(len(keys))
for category, label, color in [("both", "Both alerts", "#865aa3"), ("loss_only", "Loss streak only", "#c98530"), ("adjusted_only", "Adjusted only", "#307f92"), ("neither", "Neither", "#bbbbbb")]:
    d = o.filter(pl.col("category") == category).sort("speed", "fold")
    axes[0].barh(np.arange(len(keys)), d["share"].to_numpy() * 100, left=left, label=label, color=color)
    left += d["share"].to_numpy() * 100
for j, (category, label, color) in enumerate([("loss_only", "Loss streak only", "#c98530"), ("adjusted_only", "Adjusted only", "#307f92")]):
    d = o.filter(pl.col("category") == category).sort("speed", "fold")
    axes[1].barh(np.arange(len(keys)) + (j - .5) * .3, d["past_expected"].to_numpy() * 100, height=.27, label=label, color=color)
axes[0].set(yticks=np.arange(len(keys)), yticklabels=[f"{sp}: {fo}" for sp, fo in keys], xlabel="Share of eligible player-games (%)", xlim=(0, 100))
axes[1].set(xlabel="Mean expected score in the prior five games (%)", xlim=(0, 100))
axes[0].invert_yaxis()
axes[0].legend(loc="upper center", bbox_to_anchor=(.5, -.2), ncol=2, fontsize=9)
axes[1].legend(loc="upper center", bbox_to_anchor=(.5, -.2), ncol=2, fontsize=9)
fig.suptitle("A loss streak and underperformance flag different situations")
fig.tight_layout()
for ext in ["svg", "png"]:
    fig.savefig(f"outputs/figures/player_alerts_overlap.{ext}", dpi=160)
plt.close(fig)
