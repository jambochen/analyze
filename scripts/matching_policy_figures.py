# 同時畫全體改善與可替換範圍，避免只展示有利的小群體。
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

s = pl.read_csv("outputs/tables/matching_policy_summary.csv")
c = pl.read_csv("outputs/tables/matching_policy_coverage.csv")
plt.rcParams.update({"svg.fonttype": "path", "font.size": 10})
keys = s.select("speed", "fold").unique().sort("speed", "fold").rows()
policies = ["observed", "rating", "baseline", "recent"]
colors = ["#888888", "#c98530", "#307f92", "#865aa3"]
fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
for ax, metric, scale, title in zip(axes, ["elo_gap", "base_outside25_75", "recent_outside25_75"], [1, 100, 100], ["Mean absolute rating gap", "Baseline score outside [0.25, 0.75] (%)", "Recent score outside [0.25, 0.75] (%)"]):
    for j, (policy, color) in enumerate(zip(policies, colors)):
        d = s.filter(pl.col("policy") == policy).sort("speed", "fold")
        ax.barh(np.arange(len(keys)) + (j - 1.5) * .18, d[metric].to_numpy() * scale, height=.16, color=color, label=policy.title())
    ax.set(yticks=np.arange(len(keys)), yticklabels=[f"{sp}: {fo}" for sp, fo in keys], xlabel=title)
    ax.grid(axis="x", alpha=.2)
axes[0].invert_yaxis()
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, .94), ncol=4, fontsize=9)
fig.suptitle("Fixed-batch reassignment: all observed games included")
fig.tight_layout(rect=(0, 0, 1, .87))
for ext in ["svg", "png"]:
    fig.savefig(f"outputs/figures/matching_policy_comparison.{ext}", dpi=160)
plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 5))
left = np.zeros(len(keys))
cats = ["single", "no_alternative", "ambiguous_rule", "observed_rematch", "first", "later_random", "later_smart"]
for category in cats:
    vals = []
    for speed, fold in keys:
        d = c.filter(pl.col("speed") == speed, pl.col("fold") == fold)
        vals.append(d.filter(pl.col("category") == category)["games"].sum() / d["games"].sum() * 100)
    ax.barh(np.arange(len(keys)), vals, left=left, label=category.replace("_", " ").title())
    left += vals
ax.set(yticks=np.arange(len(keys)), yticklabels=[f"{sp}: {fo}" for sp, fo in keys], xlabel="Share of all observed games (%)", xlim=(0, 100), title="Coverage: fixed colors, fixed players, fixed start times")
ax.invert_yaxis()
ax.legend(loc="upper center", bbox_to_anchor=(.5, -.17), ncol=3, fontsize=9)
fig.tight_layout()
for ext in ["svg", "png"]:
    fig.savefig(f"outputs/figures/matching_policy_coverage.{ext}", dpi=160)
plt.close(fig)
