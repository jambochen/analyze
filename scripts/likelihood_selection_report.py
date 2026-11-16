# 報告只整理已固定的共同流程輸出；模型名稱、例子、損失與校準使用相同的訓練內選擇。
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

p = pl.read_csv("outputs/tables/likelihood_selection_prediction.csv")
c = pl.read_csv("outputs/tables/likelihood_selection_calibration.csv")
bins = pl.read_csv("outputs/tables/likelihood_selection_calibration_bins.csv")
export = json.loads(Path("outputs/tables/likelihood_selection_models.json").read_text())
selection = pl.read_csv("outputs/tables/likelihood_selection_selection.csv")
model_rows = []
labels = {"baseline": "評分基準", "results": "前局結果", "rating_change": "評分變動", "recent": "結果＋評分變動", "rd_change": "評分變動×RD", "interaction": "結果＋RD 交互作用", "rd_direction": "評分變動×RD＋不對稱", "asymmetric": "結果＋RD 不對稱", "spline_baseline": "RD 樣條", "spline_results": "結果＋RD 樣條", "spline_change": "評分變動＋RD 樣條", "spline_recent": "結果＋變動＋RD 樣條"}
for spec in export:
    choice = spec["models"][spec["selected"]]
    ss = selection.filter(pl.col("speed") == spec["speed"], pl.col("fold") == spec["fold"])
    selected = ss.filter(pl.col("model") == spec["selected"]).row(0, named=True)
    aic = ss.filter(pl.col("model") == spec["selected_aic"]).row(0, named=True)
    model_rows.append({"speed": spec["speed"], "fold": spec["fold"], "selected": spec["selected"], "label": labels[spec["selected"]], "aic_label": labels[spec["selected_aic"]], "columns": ",".join(choice["columns"]), "spline_parameters": len(choice["spline_coef"]), "k": selected["k"], "effective_k": selected["effective_k"], "cl_bic": selected["cl_bic_games"], "cl_aic": aic["cl_aic"], "aic_choice_delta_bic": aic["delta_cl_bic_games"], "bic_choice_delta_aic": selected["delta_cl_aic"], "player_scale_label": labels[spec["selected_bic_players"]], "row_scale_label": labels[spec["selected_bic_rows"]], "train_end_exclusive": spec["train_end_exclusive"]})
pl.DataFrame(model_rows).sort("speed", "fold").write_csv("outputs/tables/likelihood_selection_report_models.csv")
Path("outputs/tables/prediction_models.json").write_text(json.dumps(export, ensure_ascii=False, indent=2))

plt.rcParams.update({"svg.fonttype": "path", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
for filename, pairs in [("branches", [("interaction", "recent", "BIC within interaction candidates", "#247a70"), ("spline", "recent", "BIC within spline candidates", "#b05a16")]), ("selected", [("selected", "baseline", "Selected model (BIC)", "#33414a")])]:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, group, title in zip(axes, ["all", "novice", "established"], ["All games", "At least one high-RD player", "Both players low-RD"]):
        for j, (model, reference, label, color) in enumerate(pairs):
            d = p.filter(pl.col("group") == group, pl.col("metric") == "log_loss", pl.col("model") == model, pl.col("vs") == reference)
            if filename == "branches":
                d = d.filter(pl.col("speed").is_in(["Bullet", "Blitz"]))
            d = d.sort("speed", "fold")
            y = np.arange(d.height)+(j-(len(pairs)-1)/2)*.18
            ax.errorbar(d["diff"].to_numpy()*1000, y, xerr=d["se"].to_numpy()*1960, fmt="o", color=color, capsize=3, label=label)
        ax.set(yticks=np.arange(d.height), yticklabels=[f"{r['speed']} / {r['fold']}" for r in d.iter_rows(named=True)], title=title, xlabel="Log-loss difference (x 0.001)")
        ax.axvline(0, color="grey", linestyle="--")
        ax.grid(axis="x", alpha=.15)
    axes[0].invert_yaxis()
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=len(pairs), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .9))
    fig.savefig(f"outputs/figures/likelihood_selection_report_{filename}.svg", bbox_inches="tight")
    plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for ax, group, title in zip(axes, ["all", "novice", "established"], ["All games", "At least one high-RD player", "Both players low-RD"]):
    for j, (source, model, label, color) in enumerate([(p, "recent", "Simple recent", "#b05a16"), (p, "selected", "Selected model (BIC)", "#33414a")]):
        d = source.filter(pl.col("group") == group, pl.col("metric") == "log_loss", pl.col("model") == model, pl.col("vs") == "baseline").sort("speed", "fold")
        reference = d["mean"].to_numpy()-d["diff"].to_numpy()
        ax.errorbar(-100*d["diff"].to_numpy()/reference, np.arange(d.height)+(j-.5)*.18, xerr=196*d["se"].to_numpy()/reference, fmt="o", color=color, capsize=3, label=label)
    ax.set(yticks=np.arange(d.height), yticklabels=[f"{r['speed']} / {r['fold']}" for r in d.iter_rows(named=True)], title=title, xlabel="Relative log-loss reduction (%)")
    ax.axvline(0, color="grey", linestyle="--")
    ax.grid(axis="x", alpha=.15)
axes[0].invert_yaxis()
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False)
fig.tight_layout(rect=(0, 0, 1, .9))
fig.savefig("outputs/figures/likelihood_selection_report_gain.svg", bbox_inches="tight")
plt.close(fig)

changes = pl.read_csv("outputs/tables/likelihood_selection_changes.csv").filter(pl.col("group") == "all", pl.col("vs") == "baseline").sort("speed", "fold")
fig, ax = plt.subplots(figsize=(10, 3.6))
for column, label, marker in [("mean_abs_win_pp", "Mean", "o"), ("p90_abs_win_pp", "90th percentile", "x")]:
    ax.scatter(changes[column], np.arange(changes.height), label=label, marker=marker, s=55)
ax.set(yticks=np.arange(changes.height), yticklabels=[f"{r['speed']} / {r['fold']}" for r in changes.iter_rows(named=True)], xlabel="Absolute win-probability change (percentage points)")
ax.invert_yaxis()
ax.grid(alpha=.15)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("outputs/figures/likelihood_selection_report_changes.svg", bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for ax, target in zip(axes, ["loss", "draw", "win"]):
    for j, (model, label, color) in enumerate([("baseline", "Baseline", "#888888"), ("recent", "Simple recent", "#b05a16"), ("selected", "Selected model (BIC)", "#247a70")]):
        d = c.filter(pl.col("group") == "all", pl.col("target") == target, pl.col("binning") == "own_prediction", pl.col("model") == model).sort("speed", "fold")
        ax.scatter(d["weighted_abs_gap_pp"], np.arange(d.height)+(j-1)*.18, label=label, color=color)
    ax.set(yticks=np.arange(d.height), yticklabels=[f"{r['speed']} / {r['fold']}" for r in d.iter_rows(named=True)], title=target.title(), xlabel="Weighted absolute calibration gap (pp)")
    ax.grid(axis="x", alpha=.15)
axes[0].invert_yaxis()
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False)
fig.tight_layout(rect=(0, 0, 1, .9))
fig.savefig("outputs/figures/likelihood_selection_report_classes.svg", bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, fold in zip(axes, ["Jul-Sep", "Oct-Dec"]):
    for model, label, color in [("baseline", "Baseline", "#888888"), ("recent", "Simple recent", "#b05a16"), ("selected", "Selected model (BIC)", "#247a70")]:
        d = bins.filter(pl.col("speed") == "Blitz", pl.col("fold") == fold, pl.col("group") == "all", pl.col("target") == "score", pl.col("binning") == "baseline_prediction", pl.col("model") == model).sort("bin")
        reference = bins.filter(pl.col("speed") == "Blitz", pl.col("fold") == fold, pl.col("group") == "all", pl.col("target") == "score", pl.col("binning") == "baseline_prediction", pl.col("model") == "baseline").sort("bin")
        ax.plot(reference["predicted"], d["residual"]*100, marker="o", label=label, color=color)
    ax.axhline(0, color="grey", linestyle="--")
    ax.set(title=f"Blitz / {fold}", xlabel="Baseline mean expected score", ylabel="Observed - predicted score (pp)")
    ax.grid(alpha=.15)
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False)
fig.tight_layout(rect=(0, 0, 1, .9))
fig.savefig("outputs/figures/likelihood_selection_report_score.svg", bbox_inches="tight")
plt.close(fig)

entry = pl.read_csv("outputs/tables/entry_context_scores.csv").filter(pl.col("group") == "all", pl.col("metric") == "log_loss", pl.col("model") == "first", pl.col("reference") == "recalibrated")
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharex=True, sharey=True)
for ax, weighting, title in zip(axes, ["games", "players"], ["Game weights", "Equal player weights"]):
    d = entry.filter(pl.col("weighting") == weighting).sort("speed", "fold")
    x = 1000*d["diff"].to_numpy()
    ax.errorbar(x, np.arange(d.height), xerr=np.array([x-1000*d["lo"].to_numpy(), 1000*d["hi"].to_numpy()-x]), fmt="o", color="#247a70", capsize=3)
    ax.set(yticks=np.arange(d.height), yticklabels=[f"{r['speed']} / {r['fold']}" for r in d.iter_rows(named=True)], title=title, xlabel="Entry-label log-loss difference (x 0.001)")
    ax.axvline(0, color="grey", linestyle="--")
    ax.grid(axis="x", alpha=.15)
axes[0].invert_yaxis()
fig.tight_layout()
fig.savefig("outputs/figures/likelihood_selection_report_entry.svg", bbox_inches="tight")
plt.close(fig)
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, metric, title in zip(axes, ["log_loss", "score_mse"], ["Log loss", "Score MSE"]):
    for j, (group, label, color) in enumerate([("all", "All", "#555555"), ("novice", "Either RD high", "#b05a16"), ("established", "Both RD low", "#247a70")]):
        d = p.filter(pl.col("model") == "recent", pl.col("vs") == "baseline", pl.col("metric") == metric, pl.col("group") == group).sort("speed", "fold")
        pos = np.arange(d.height)+(j-1)*.2
        ax.hlines(pos, d["lo"], d["hi"], color=color)
        ax.scatter(d["diff"], pos, color=color, label=label, s=25)
    ax.axvline(0, color="grey", linestyle="--")
    ax.set(title=title, xlabel="Simple recent minus rating baseline", yticks=np.arange(d.height), yticklabels=[f"{r['speed']} / {r['fold']}" for r in d.iter_rows(named=True)])
    ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))
axes[0].invert_yaxis()
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False)
fig.tight_layout(rect=(0, 0, 1, .92))
fig.savefig("outputs/figures/likelihood_selection_report_metrics.svg", bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for ax, group, title in zip(axes, ["all", "novice", "established"], ["All games", "At least one high-RD player", "Both players low-RD"]):
    d = p.filter(pl.col("model") == "aic", pl.col("vs") == "selected", pl.col("metric") == "log_loss", pl.col("group") == group).sort("speed", "fold")
    ax.errorbar(d["diff"].to_numpy()*1000, np.arange(d.height), xerr=d["se"].to_numpy()*1960, fmt="o", color="#247a70", capsize=3)
    ax.axvline(0, color="grey", linestyle="--")
    ax.set(title=title, xlabel="AIC minus BIC test log loss (x 0.001)", yticks=np.arange(d.height), yticklabels=[f"{r['speed']} / {r['fold']}" for r in d.iter_rows(named=True)])
    ax.grid(axis="x", alpha=.15)
axes[0].invert_yaxis()
fig.tight_layout()
fig.savefig("outputs/figures/likelihood_selection_report_criteria_test.svg", bbox_inches="tight")
plt.close(fig)
print(pl.DataFrame(model_rows).select("speed", "fold", "label", "aic_label", "cl_bic", "aic_choice_delta_bic", "player_scale_label", "row_scale_label"))
