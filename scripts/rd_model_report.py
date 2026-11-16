# 報告只整理已固定的共同流程輸出；模型名稱、例子、損失與校準使用相同的訓練內選擇。
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

p = pl.read_csv("outputs/tables/rd_model_comparison_prediction.csv")
c = pl.read_csv("outputs/tables/rd_model_comparison_calibration.csv")
bins = pl.read_csv("outputs/tables/rd_model_comparison_calibration_bins.csv")
old = pl.read_csv("outputs/tables/temporal_prediction_prediction.csv")
export = json.loads(Path("outputs/tables/rd_model_comparison_models.json").read_text())
coef = pl.read_csv("outputs/tables/temporal_prediction_coef.csv")
model_rows = []
labels = {"spline": "RD 樣條", "interaction": "RD 交互作用", "recent": "近況", "baseline": "評分基準"}
for spec in export:
    choice = spec["models"][spec["selected"]]
    model_rows.append({"speed": spec["speed"], "fold": spec["fold"], "selected": spec["selected"], "label": labels[spec["selected"]], "columns": ",".join(choice["columns"]), "spline_df": spec["spline_df"] if choice["spline_coef"] else 0, "spline_parameters": len(choice["spline_coef"]), "train_end_exclusive": spec["train_end_exclusive"]})
# Classical 沿用已完成的共同流程：兩段皆未選入近況，直接輸出各段的評分基準。
for fold, end in [("Jul-Sep", "2014-07-01"), ("Oct-Dec", "2014-10-01")]:
    beta = coef.filter(pl.col("game") == "Lichess", pl.col("speed") == "Classical", pl.col("fold") == fold, pl.col("fit") == "base_refit").sort("param_index")["est"].to_list()
    export.append({"speed": "Classical", "fold": fold, "selected": "baseline", "train_end_exclusive": end, "test_end_exclusive": "2014-10-01" if fold == "Jul-Sep" else "2015-01-01", "baseline_coef": beta, "models": {"baseline": {"columns": [], "coef": [], "knots": [], "center": [], "spline_coef": []}}})
    model_rows.append({"speed": "Classical", "fold": fold, "selected": "baseline", "label": labels["baseline"], "columns": "", "spline_df": 0, "spline_parameters": 0, "train_end_exclusive": end})
pl.DataFrame(model_rows).sort("speed", "fold").write_csv("outputs/tables/rd_model_report_models.csv")
Path("outputs/tables/prediction_models.json").write_text(json.dumps(export, ensure_ascii=False, indent=2))

plt.rcParams.update({"svg.fonttype": "path", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
for filename, pairs in [("branches", [("interaction", "recent", "Interaction", "#247a70"), ("spline", "recent", "Spline", "#b05a16")]), ("selected", [("selected", "baseline", "Training-selected model", "#33414a")])]:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, group, title in zip(axes, ["all", "novice", "established"], ["All games", "At least one high-RD player", "Both players low-RD"]):
        for j, (model, reference, label, color) in enumerate(pairs):
            d = p.filter(pl.col("group") == group, pl.col("metric") == "log_loss", pl.col("model") == model, pl.col("vs") == reference).sort("speed", "fold")
            y = np.arange(d.height)+(j-(len(pairs)-1)/2)*.18
            ax.errorbar(d["diff"].to_numpy()*1000, y, xerr=d["se"].to_numpy()*1960, fmt="o", color=color, capsize=3, label=label)
        ax.set(yticks=np.arange(d.height), yticklabels=[f"{r['speed']} / {r['fold']}" for r in d.iter_rows(named=True)], title=title, xlabel="Log-loss difference (x 0.001)")
        ax.axvline(0, color="grey", linestyle="--")
        ax.grid(axis="x", alpha=.15)
    axes[0].invert_yaxis()
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=len(pairs), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .9))
    fig.savefig(f"outputs/figures/rd_model_report_{filename}.svg", bbox_inches="tight")
    plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for ax, group, title in zip(axes, ["all", "novice", "established"], ["All games", "At least one high-RD player", "Both players low-RD"]):
    for j, (source, model, label, color) in enumerate([(old.filter(pl.col("game") == "Lichess", pl.col("speed").is_in(["Bullet", "Blitz"])), "recent", "Recent", "#b05a16"), (p, "selected", "Training-selected", "#33414a")]):
        d = source.filter(pl.col("group") == group, pl.col("metric") == "log_loss", pl.col("model") == model, pl.col("vs") == "baseline").sort("speed", "fold")
        reference = d["mean"].to_numpy()-d["diff"].to_numpy()
        ax.errorbar(-100*d["diff"].to_numpy()/reference, np.arange(d.height)+(j-.5)*.18, xerr=196*d["se"].to_numpy()/reference, fmt="o", color=color, capsize=3, label=label)
    ax.set(yticks=np.arange(d.height), yticklabels=[f"{r['speed']} / {r['fold']}" for r in d.iter_rows(named=True)], title=title, xlabel="Relative log-loss reduction (%)")
    ax.axvline(0, color="grey", linestyle="--")
    ax.grid(axis="x", alpha=.15)
axes[0].invert_yaxis()
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False)
fig.tight_layout(rect=(0, 0, 1, .9))
fig.savefig("outputs/figures/rd_model_report_gain.svg", bbox_inches="tight")
plt.close(fig)

changes = pl.read_csv("outputs/tables/rd_model_comparison_changes.csv").filter(pl.col("group") == "all", pl.col("vs") == "baseline").sort("speed", "fold")
fig, ax = plt.subplots(figsize=(10, 3.6))
for column, label, marker in [("mean_abs_win_pp", "Mean", "o"), ("p90_abs_win_pp", "90th percentile", "x")]:
    ax.scatter(changes[column], np.arange(changes.height), label=label, marker=marker, s=55)
ax.set(yticks=np.arange(changes.height), yticklabels=[f"{r['speed']} / {r['fold']}" for r in changes.iter_rows(named=True)], xlabel="Absolute win-probability change (percentage points)")
ax.invert_yaxis()
ax.grid(alpha=.15)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("outputs/figures/rd_model_report_changes.svg", bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for ax, target in zip(axes, ["loss", "draw", "win"]):
    for j, (model, label, color) in enumerate([("baseline", "Baseline", "#888888"), ("recent", "Recent", "#b05a16"), ("selected", "Training-selected", "#247a70")]):
        d = c.filter(pl.col("group") == "all", pl.col("target") == target, pl.col("binning") == "own_prediction", pl.col("model") == model).sort("speed", "fold")
        ax.scatter(d["weighted_abs_gap_pp"], np.arange(d.height)+(j-1)*.18, label=label, color=color)
    ax.set(yticks=np.arange(d.height), yticklabels=[f"{r['speed']} / {r['fold']}" for r in d.iter_rows(named=True)], title=target.title(), xlabel="Weighted absolute calibration gap (pp)")
    ax.grid(axis="x", alpha=.15)
axes[0].invert_yaxis()
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False)
fig.tight_layout(rect=(0, 0, 1, .9))
fig.savefig("outputs/figures/rd_model_report_classes.svg", bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, fold in zip(axes, ["Jul-Sep", "Oct-Dec"]):
    for model, label, color in [("baseline", "Baseline", "#888888"), ("recent", "Recent", "#b05a16"), ("selected", "Training-selected", "#247a70")]:
        d = bins.filter(pl.col("speed") == "Blitz", pl.col("fold") == fold, pl.col("group") == "all", pl.col("target") == "score", pl.col("binning") == "baseline_prediction", pl.col("model") == model).sort("bin")
        reference = bins.filter(pl.col("speed") == "Blitz", pl.col("fold") == fold, pl.col("group") == "all", pl.col("target") == "score", pl.col("binning") == "baseline_prediction", pl.col("model") == "baseline").sort("bin")
        ax.plot(reference["predicted"], d["residual"]*100, marker="o", label=label, color=color)
    ax.axhline(0, color="grey", linestyle="--")
    ax.set(title=f"Blitz / {fold}", xlabel="Baseline mean expected score", ylabel="Observed - predicted score (pp)")
    ax.grid(alpha=.15)
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False)
fig.tight_layout(rect=(0, 0, 1, .9))
fig.savefig("outputs/figures/rd_model_report_score.svg", bbox_inches="tight")
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
fig.savefig("outputs/figures/rd_model_report_entry.svg", bbox_inches="tight")
plt.close(fig)
print(pl.DataFrame(model_rows))
