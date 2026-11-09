# 新情境的增量與完整模型比較分開，適用範圍同時呈現覆蓋與校準。
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

s = pl.read_csv("outputs/tables/entry_context_scores.csv")
c = pl.read_csv("outputs/tables/entry_context_calibration.csv")
v = pl.read_csv("outputs/tables/entry_context_coverage.csv")
plt.rcParams.update({"svg.fonttype": "path", "font.size": 10})
keys = s.select("speed", "fold").unique().sort("speed", "fold").rows()
models = ["first", "interval", "combined"]
labels = ["Tournament entry", "Start-to-start interval", "Entry + interval"]
colors = ["#c98530", "#307f92", "#865aa3"]
for reference in ["recalibrated", "recent"]:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, weighting, title in zip(axes, ["games", "players"], ["Game weights", "Equal player weights"]):
        for j, (model, label, color) in enumerate(zip(models, labels, colors)):
            d = s.filter(pl.col("group") == "all", pl.col("metric") == "log_loss", pl.col("weighting") == weighting, pl.col("model") == model, pl.col("reference") == reference).sort("speed", "fold")
            x = d["diff"].to_numpy() * 1000
            ax.errorbar(x, np.arange(len(keys)) + (j - 1) * .2, xerr=np.array([x - d["lo"].to_numpy() * 1000, d["hi"].to_numpy() * 1000 - x]), fmt="o", color=color, capsize=3, label=label)
        ax.set(yticks=np.arange(len(keys)), yticklabels=[f"{sp}: {fo}" for sp, fo in keys], xlabel="Change in log loss (x 0.001; lower is better)", title=title)
        ax.axvline(0, color="grey", linestyle="--")
        ax.grid(axis="x", alpha=.2)
    axes[0].invert_yaxis()
    fig.suptitle("Entry context vs " + ("recalibration only" if reference == "recalibrated" else "the fixed recent model"))
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, .93), ncol=3)
    fig.tight_layout(rect=(0, 0, 1, .85))
    for ext in ["svg", "png"]:
        fig.savefig(f"outputs/figures/entry_context_{reference}.{ext}", dpi=160)
    plt.close(fig)

groups = ["all", "low_rd", "supported", "low_rd_supported"]
group_labels = ["All", "Low RD", "Supported", "Both"]
fig, axes = plt.subplots(2, len(keys), figsize=(15, 7), sharey="row")
for j, (speed, fold) in enumerate(keys):
    order = dict(zip(groups, range(len(groups))))
    d = v.filter(pl.col("speed") == speed, pl.col("fold") == fold, pl.col("group").is_in(groups), pl.col("target") == "any_first").with_columns(order=pl.col("group").replace_strict(order)).sort("order")
    axes[0, j].bar(np.arange(4) - .17, d["coverage"].to_numpy() * 100, width=.32, color="#307f92", label="All-game retention")
    axes[0, j].bar(np.arange(4) + .17, d["target_retention"].to_numpy() * 100, width=.32, color="#c98530", label="Entry-game retention")
    axes[0, j].set(title=f"{speed}: {fold}", ylim=(0, 105))
    for k, (model, label, color) in enumerate([("recent", "Fixed recent", "#888888"), ("combined", "Entry + interval", "#865aa3")]):
        d = c.filter(pl.col("speed") == speed, pl.col("fold") == fold, pl.col("group").is_in(groups), pl.col("weighting") == "games", pl.col("model") == model).with_columns(order=pl.col("group").replace_strict(order)).sort("order")
        axes[1, j].bar(np.arange(4) + (k - .5) * .34, d["common_bin_abs_gap"].to_numpy() * 100, width=.32, color=color, label=label)
    for ax in axes[:, j]:
        ax.set(xticks=np.arange(4), xticklabels=group_labels)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=.2)
axes[0, 0].set_ylabel("Retained games (%)")
axes[1, 0].set_ylabel("Common-bin absolute calibration gap\n(percentage points)")
h1, l1 = axes[0, -1].get_legend_handles_labels()
h2, l2 = axes[1, -1].get_legend_handles_labels()
fig.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(.5, .94), ncol=4, fontsize=9)
fig.suptitle("Restricting scope: coverage cost and descriptive calibration")
fig.tight_layout(rect=(0, 0, 1, .88))
for ext in ["svg", "png"]:
    fig.savefig(f"outputs/figures/entry_context_scope.{ext}", dpi=160)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, group, title in zip(axes, ["known_interval", "missing_interval"], ["Both intervals observed", "At least one interval missing"]):
    for j, (weighting, label, color) in enumerate([("games", "Game weights", "#888888"), ("players", "Equal player weights", "#307f92")]):
        d = s.filter(pl.col("group") == group, pl.col("metric") == "log_loss", pl.col("model") == "interval", pl.col("reference") == "recent", pl.col("weighting") == weighting).sort("speed", "fold")
        x = d["diff"].to_numpy() * 1000
        ax.errorbar(x, np.arange(len(keys)) + (j - .5) * .2, xerr=np.array([x - d["lo"].to_numpy() * 1000, d["hi"].to_numpy() * 1000 - x]), fmt="o", color=color, capsize=3, label=label)
    ax.set(yticks=np.arange(len(keys)), yticklabels=[f"{sp}: {fo}" for sp, fo in keys], xlabel="Log-loss change vs fixed recent model (x 0.001)", title=title)
    ax.axvline(0, color="grey", linestyle="--")
    ax.grid(axis="x", alpha=.2)
axes[0].invert_yaxis()
fig.suptitle("Interval extension: observed history and missing history")
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, .93), ncol=2)
fig.tight_layout(rect=(0, 0, 1, .85))
for ext in ["svg", "png"]:
    fig.savefig(f"outputs/figures/entry_context_history.{ext}", dpi=160)
plt.close(fig)
