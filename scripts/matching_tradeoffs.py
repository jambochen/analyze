# 比較實際配對目標的改善與玩家局代價；各項皆為固定歷史集合內的描述。
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import novice

p = pl.read_csv("outputs/tables/matching_policy_pooled.csv")
r = pl.read_csv("outputs/tables/matching_policy_players.csv")
reference = p.filter(pl.col("policy") == "observed").select("scope", original_gap="elo_gap", original_tail="gap_ge400", original_max="batch_max_gap")
rating = p.filter(pl.col("policy") == "rating").select("scope", rating_gap="elo_gap")
p = p.join(reference, on="scope").join(rating, on="scope").with_columns(
    mean_gap_gain=pl.col("original_gap") - pl.col("elo_gap"),
    tail400_reduction_pp=100 * (pl.col("original_tail") - pl.col("gap_ge400")),
    batch_max_gain=pl.col("original_max") - pl.col("batch_max_gap"),
    worse_player_pct=100 * pl.col("worse_gap"), better_player_pct=100 * pl.col("better_gap"),
    changed_player_pct=100 * pl.col("changed_games") / pl.col("games"),
).with_columns(mean_gain_retained_pct=100 * pl.col("mean_gap_gain") / (pl.col("original_gap") - pl.col("rating_gap")))
p.write_csv("outputs/tables/matching_tradeoffs_summary.csv")

player_cols = [c for c in r.columns if c not in ["speed", "fold", "category", "policy", "group", "player_games"]]
pooled = []
folded = []
for scope, d in [("all", r), ("modifiable", r.filter(pl.col("category").is_in(["first", "later_smart", "later_random"])) )]:
    for keys, destination in [(["policy", "group"], pooled), (["speed", "fold", "policy", "group"], folded)]:
        q = d.group_by(keys).agg(pl.col("player_games").sum(), *[(pl.col(c) * pl.col("player_games")).sum().truediv(pl.col("player_games").sum()).alias(c) for c in player_cols]).with_columns(scope=pl.lit(scope))
        destination.append(q.with_columns(mean_increase_among_worse=pl.when(pl.col("worse_gap") > 0).then(pl.col("positive_gap_change") / pl.col("worse_gap"))))
players = pl.concat(pooled)
players.write_csv("outputs/tables/matching_tradeoffs_players.csv")
pl.concat(folded).write_csv("outputs/tables/matching_tradeoffs_player_folds.csv")
# 玩家權重與對局權重應一致；高 RD 與低 RD 玩家人數不因替換對手改變。
check = players.filter(pl.col("group") == "all").join(p, on=["scope", "policy"], suffix="_pair")
assert (check["player_games"] == 2 * check["games"]).all()
for c in ["elo_gap", "base_imbalance", "recent_imbalance", "worse_gap", "better_gap", "positive_gap_change", "worse_base", "worse_recent"]:
    assert (check[c] - check[c + "_pair"]).abs().max() < 1e-10
assert players.group_by("scope", "group").agg(pl.col("player_games").n_unique())["player_games"].max() == 1
assert players.filter(pl.col("policy") == "cap0")["worse_gap"].max() == 0

plt.rcParams.update({"svg.fonttype": "path", "font.size": 10})
policies = ["rating", "baseline", "recent", "tail200", "tail400", "minimax", "cap0", "cap50", "cap100", "rating_guard"]
labels = ["Mean rating gap", "Baseline probability", "Recent probability", "Tail >=200 first", "Tail >=400 first", "Worst match first", "No gap increase", "Increase <=50", "Increase <=100", "Mean + fewer worsenings"]
colors = ["#888888"] * 3 + ["#c98530"] * 2 + ["#865aa3"] + ["#307f92"] * 3 + ["#557b43"]
for scope in ["all", "modifiable"]:
    fig, axes = plt.subplots(1, 4, figsize=(15, 5), sharey=True)
    d = p.filter(pl.col("scope") == scope).with_columns(order=pl.col("policy").replace_strict(dict(zip(policies, range(len(policies)))), default=-1)).filter(pl.col("order") >= 0).sort("order")
    for ax, metric, label in zip(axes, ["mean_gap_gain", "tail400_reduction_pp", "batch_max_gain", "worse_player_pct"], ["Mean gap reduction\n(rating points)", "Fewer games with gap >=400\n(percentage points)", "Lower batch maximum gap\n(game-weighted points)", "Player-games with larger gap\n(%)"]):
        ax.barh(np.arange(len(policies)), d[metric].to_numpy(), color=colors)
        ax.set(yticks=np.arange(len(policies)), yticklabels=labels, xlabel=label)
        ax.axvline(0, color="black", linewidth=.6)
        ax.grid(axis="x", alpha=.2)
    axes[0].invert_yaxis()
    fig.suptitle("Decision tradeoffs: " + ("all observed games" if scope == "all" else "batches with a legal alternative"))
    fig.tight_layout()
    for ext in ["svg", "png"]:
        fig.savefig(f"outputs/figures/matching_tradeoffs_{scope}.{ext}", dpi=160)
    plt.close(fig)

fig, ax = plt.subplots(figsize=(10, 5))
for j, (group, label, color) in enumerate([("high_rd", f"Own RD >={novice.RD}", "#c98530"), ("established", f"Own RD <{novice.RD}", "#307f92")]):
    d = players.filter(pl.col("scope") == "modifiable", pl.col("group") == group).with_columns(order=pl.col("policy").replace_strict(dict(zip(policies, range(len(policies)))), default=-1)).filter(pl.col("order") >= 0).sort("order")
    ax.barh(np.arange(len(policies)) + (j - .5) * .35, d["worse_gap"].to_numpy() * 100, height=.32, label=label, color=color)
ax.set(yticks=np.arange(len(policies)), yticklabels=labels, xlabel="Player-games with larger rating gap (%)", title="Cost by own rating uncertainty: modifiable batches")
ax.invert_yaxis()
ax.legend()
ax.grid(axis="x", alpha=.2)
fig.tight_layout()
for ext in ["svg", "png"]:
    fig.savefig(f"outputs/figures/matching_tradeoffs_uncertainty.{ext}", dpi=160)
plt.close(fig)
print(p.select("scope", "policy", "mean_gap_gain", "mean_gain_retained_pct", "tail400_reduction_pp", "batch_max_gain", "worse_player_pct").sort("scope", "policy"))
