# 將固定模型的相對改善、機率修正與情境分開呈現，避免把修正幅度當成效益。
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

scores = pl.read_csv("outputs/tables/prediction_value_scores.csv")
changes = pl.read_csv("outputs/tables/prediction_value_changes.csv")
calibration = pl.read_csv("outputs/tables/prediction_value_calibration.csv")
scenarios = pl.read_csv("outputs/tables/prediction_value_scenarios.csv")
checks = pl.read_csv("outputs/tables/prediction_value_checks.csv")
assert checks.select(pl.struct("game", "speed", "fold").n_unique()).item() == 9
assert checks["max_loss_error"].max() < 1e-10
plt.rcParams.update({"svg.fonttype": "path", "font.size": 10})
colors = {"all": "#555555", "novice": "#b05a16", "established": "#247a70"}

for game in ["Lichess", "OGS", "AoE2"]:
    data = scores.filter(pl.col("game") == game, pl.col("metric") == "log_loss")
    panels = data.select("speed", "fold").unique().sort("speed", "fold").rows()
    fig, axes = plt.subplots(1, 2, figsize=(13, max(4, len(panels) * .8)))
    for ax, model in zip(axes, ["recent", "novice_extension"]):
        if game == "AoE2" and model == "novice_extension":
            ax.axis("off")
            ax.text(.5, .5, "RD unavailable\nNo RD extension evaluated", transform=ax.transAxes, ha="center", va="center")
            continue
        for j, (group, color) in enumerate(colors.items()):
            d = data.filter(pl.col("group") == group, pl.col("model") == model).sort("speed", "fold")
            y = np.arange(len(panels)) + (j - 1) * .2
            x = d["relative_gain_pct"].to_numpy()
            ax.errorbar(x, y, xerr=np.array([x - d["relative_gain_lo_pct"].to_numpy(), d["relative_gain_hi_pct"].to_numpy() - x]), fmt="o", capsize=3, markersize=4, color=color, label=group)
        ax.axvline(0, color="grey", linestyle="--")
        ax.set(yticks=np.arange(len(panels)), yticklabels=[f"{speed}: {fold}" for speed, fold in panels], xlabel="Reduction in log loss (%)", title="Recent vs baseline" if model == "recent" else "RD extension vs recent")
        ax.invert_yaxis()
        ax.legend(fontsize=8)
    limitation = "end-time ordering only" if game == "OGS" else "one-month snapshot" if game == "AoE2" else "reused-data temporal evaluation"
    fig.suptitle(f"{game}: error reduction ({limitation})")
    fig.text(.5, .01, "Conditional 95% difference intervals scaled by fixed reference loss. Zero may mean no terms selected.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .95))
    for ext in ["svg", "png"]:
        fig.savefig(f"outputs/figures/prediction_value_{game.lower()}_gain.{ext}", dpi=160)
    plt.close(fig)

    d = changes.filter(pl.col("game") == game, pl.col("model") == "recent").sort("speed", "fold", "group")
    labels = [f"{r['speed']} {r['fold']} | {r['group']}" for r in d.iter_rows(named=True)]
    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, d.height * .32)))
    y = np.arange(d.height)
    axes[0].scatter(d["mean_abs_win_pp"], y, marker="o", label="Mean absolute change", color="#33414a")
    axes[0].scatter(d["p90_abs_win_pp"], y, marker="x", label="90th percentile", color="#b05a16")
    axes[0].set(yticks=y, yticklabels=labels, xlabel="Absolute win-probability change (percentage points)", title="Recent vs baseline: magnitude, not benefit")
    axes[0].invert_yaxis()
    axes[0].legend(fontsize=8)
    for model, marker in [("baseline", "o"), ("recent", "s"), ("novice_extension", "^")]:
        if game == "AoE2" and model == "novice_extension":
            continue
        c = calibration.filter(pl.col("game") == game, pl.col("target") == "score", pl.col("binning") == "baseline_prediction", pl.col("model") == model).sort("speed", "fold", "group")
        axes[1].plot(c["weighted_abs_gap_pp"], y, marker, label=model)
    axes[1].set(yticks=y, yticklabels=[], xlabel="Weighted absolute score gap (percentage points)", title="Same baseline-defined bins: descriptive bias")
    axes[1].invert_yaxis()
    axes[1].legend(fontsize=8)
    fig.suptitle(f"{game}: probability adjustments and residual bias")
    fig.text(.5, .01, "Bin summaries have sampling and binning error; no uncertainty intervals shown. Adjustments are not causal effects.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .95))
    for ext in ["svg", "png"]:
        fig.savefig(f"outputs/figures/prediction_value_{game.lower()}_magnitude.{ext}", dpi=160)
    plt.close(fig)

for speed in ["Blitz", "Bullet"]:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (group, previous) in zip(axes.flat, [("novice", "loss"), ("novice", "win"), ("established", "loss"), ("established", "win")]):
        d = scenarios.filter(pl.col("speed") == speed, pl.col("fold") == "Oct-Dec", pl.col("group") == group, pl.col("previous") == previous, pl.col("opponent_previous") == ("win" if previous == "loss" else "loss"))
        rows = [d.filter(pl.col("model") == model).row(0, named=True) for model in ["baseline", "recent", "novice_extension"]]
        bottom = np.zeros(3)
        for outcome, color in [("loss", "#83949b"), ("draw", "#ddd6c9"), ("win", "#335b70")]:
            vals = np.array([r[f"p_{outcome}"] * 100 for r in rows])
            ax.bar(np.arange(3), vals, bottom=bottom, label=outcome, color=color)
            for j, value in enumerate(vals):
                if value > 3:
                    ax.text(j, bottom[j] + value / 2, f"{value:.1f}", ha="center", va="center", fontsize=9, color="white" if outcome == "win" else "black")
            bottom += vals
        r = rows[0]
        ax.set(ylim=(0, 100), xticks=np.arange(3), xticklabels=["Baseline", "Recent", "RD extension"], ylabel="Outcome probability (%)", title=f"{group}: previous {previous} vs opponent {r['opponent_previous']}")
        ax.text(.5, -.18, f"Rating {r['elo']:.0f} vs {r['opp_elo']:.0f}; RD {r['rd']:.0f} vs {r['opp_rd']:.0f}\nPrior rating change {100*r['dr']:+.0f} vs {100*r['opp_dr']:+.0f}", transform=ax.transAxes, ha="center", va="top", fontsize=9)
    axes[0, 0].legend(loc="lower left", fontsize=8)
    fig.suptitle(f"Lichess {speed}: anonymous training-selected examples (Oct-Dec model)")
    fig.text(.5, .01, "White perspective; both histories present; similar ratings. Compare models within a panel, not outcomes across panels.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .95), h_pad=3)
    for ext in ["svg", "png"]:
        fig.savefig(f"outputs/figures/prediction_value_{speed.lower()}_scenarios.{ext}", dpi=160)
    plt.close(fig)
print("Prediction-value figures saved; all reconstruction checks passed")
