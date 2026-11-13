# 報告用的精簡序列圖直接讀既有摘要，沒有另估模型或選門檻。
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import polars as pl

s = pl.read_csv("outputs/tables/sequence_diagnostics_summary.csv")
plt.rcParams.update({"svg.fonttype": "path", "font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True, sharey=True)
for ax, weighting in zip(axes, ["window", "player"]):
    for j, (model, label, color) in enumerate([("baseline", "Baseline", "#555555"), ("recent", "Recent", "#b05a16"), ("novice_extension", "RD extension", "#247a70")]):
        d = s.filter(pl.col("metric") == "loss_run", pl.col("length") == 5, pl.col("group") == "all", pl.col("weighting") == weighting, pl.col("model") == model).sort("speed", "fold")
        y = np.arange(d.height) + (j-1)*.18
        den = d["analytic_reference"].to_numpy()
        ax.hlines(y, d["reference_lo"].to_numpy()/den, d["reference_hi"].to_numpy()/den, color=color, linewidth=6, alpha=.18)
        ax.scatter(d["observed"].to_numpy()/den, y, color=color, s=30, label=label)
    ax.axvline(1, color="grey", linestyle="--")
    ax.set(yticks=np.arange(d.height), yticklabels=[f"{r['speed']} / {r['fold']}" for r in d.iter_rows(named=True)], xlabel="Observed loss-run rate / reference", title=f"Equal {weighting} weight")
    ax.invert_yaxis()
axes[1].tick_params(labelleft=True)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
fig.text(.5, .01, "Five-game windows. Faint bars: central 95% artificial reference range, not confidence intervals.", ha="center", fontsize=10)
fig.tight_layout(rect=(0, .05, 1, .9))
fig.savefig("outputs/figures/prediction_report_sequences.svg", bbox_inches="tight")
plt.close(fig)

# 同頁只呈現全體對局的幅度；分組改善已在前頁以正式區間呈現。
changes = pl.read_csv("outputs/tables/prediction_value_changes.csv").filter(pl.col("game") == "Lichess", pl.col("speed").is_in(["Bullet", "Blitz"]), pl.col("group") == "all", pl.col("model") == "recent", pl.col("vs") == "baseline").sort("speed", "fold")
calibration = pl.read_csv("outputs/tables/prediction_value_calibration.csv").filter(pl.col("game") == "Lichess", pl.col("speed").is_in(["Bullet", "Blitz"]), pl.col("group") == "all", pl.col("target") == "score", pl.col("binning") == "baseline_prediction")
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
y = np.arange(changes.height)
for column, label, marker in [("mean_abs_win_pp", "Mean", "o"), ("p90_abs_win_pp", "90th percentile", "x")]:
    axes[0].scatter(changes[column], y, label=label, marker=marker)
axes[0].set(xlabel="Absolute win-probability change (pp)", title="Recent vs baseline")
for model, label in [("baseline", "Baseline"), ("recent", "Recent"), ("novice_extension", "RD extension")]:
    d = calibration.filter(pl.col("model") == model).sort("speed", "fold")
    axes[1].scatter(d["weighted_abs_gap_pp"], y, label=label)
axes[1].set(xlabel="Weighted absolute score gap (pp)", title="Same baseline-defined bins")
for ax in axes:
    ax.set_yticks(y, [f"{r['speed']} / {r['fold']}" for r in changes.iter_rows(named=True)])
    ax.legend(loc="lower right", fontsize=10)
    ax.set_ylim(changes.height-.4, -.6)
    ax.grid(alpha=.15)
fig.text(.5, .01, "All player-game rows. pp: percentage points. Bin summaries are descriptive, not full calibration tests.", ha="center", fontsize=10)
fig.tight_layout(rect=(0, .06, 1, 1))
fig.savefig("outputs/figures/prediction_report_magnitude.svg", bbox_inches="tight")
plt.close(fig)

# 分頁呈現幅度與校準，避免兩種不同問題共用同一張摘要圖。
fig, ax = plt.subplots(figsize=(9, 4))
for column, label, marker in [("mean_abs_win_pp", "Mean", "o"), ("p90_abs_win_pp", "90th percentile", "x")]:
    ax.scatter(changes[column], y, label=label, marker=marker, s=65)
ax.set(yticks=y, yticklabels=[f"{r['speed']} / {r['fold']}" for r in changes.iter_rows(named=True)], xlabel="Absolute win-probability change (percentage points)")
ax.invert_yaxis()
ax.legend(frameon=False)
ax.grid(alpha=.15)
fig.tight_layout()
fig.savefig("outputs/figures/prediction_report_changes.svg", bbox_inches="tight")
plt.close(fig)

bins = pl.read_csv("outputs/tables/prediction_value_calibration_bins.csv").filter(pl.col("game") == "Lichess", pl.col("speed") == "Blitz", pl.col("group") == "all", pl.col("target") == "score", pl.col("binning") == "baseline_prediction")
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, fold in zip(axes, ["Jul-Sep", "Oct-Dec"]):
    ref = bins.filter(pl.col("fold") == fold, pl.col("model") == "baseline").sort("bin")
    for model, label, marker in [("baseline", "Baseline", "o"), ("recent", "Recent", "s"), ("novice_extension", "RD extension", "^")]:
        d = bins.filter(pl.col("fold") == fold, pl.col("model") == model).sort("bin")
        ax.plot(ref["predicted"], 100*d["residual"], marker=marker, label=label)
    ax.axhline(0, color="grey", linestyle="--")
    ax.set(title=f"Blitz / {fold}", xlabel="Mean baseline prediction in bin")
    ax.grid(alpha=.15)
axes[0].set_ylabel("Observed minus predicted score (pp)")
axes[1].legend(frameon=False)
fig.tight_layout()
fig.savefig("outputs/figures/prediction_report_calibration.svg", bbox_inches="tight")
plt.close(fig)

scores = pl.read_csv("outputs/tables/prediction_value_scores.csv").filter(pl.col("game") == "Lichess", pl.col("model") == "recent", pl.col("vs") == "baseline")
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, metric, title in zip(axes, ["log_loss", "score_mse"], ["Log loss", "Score MSE"]):
    for j, (group, label, color) in enumerate([("all", "All", "#555555"), ("novice", "Either RD high", "#b05a16"), ("established", "Both RD low", "#247a70")]):
        d = scores.filter(pl.col("metric") == metric, pl.col("group") == group).sort("speed", "fold")
        pos = np.arange(d.height)+(j-1)*.2
        ax.hlines(pos, d["lo"], d["hi"], color=color)
        ax.scatter(d["diff"], pos, color=color, label=label, s=25)
    ax.axvline(0, color="grey", linestyle="--")
    ax.set(title=title, xlabel="Recent minus baseline", yticks=np.arange(d.height), yticklabels=[f"{r['speed']} / {r['fold']}" for r in d.iter_rows(named=True)])
    ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))
    ax.xaxis.set_major_locator(MaxNLocator(5))
axes[0].invert_yaxis()
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
fig.tight_layout(rect=(0, 0, 1, .92))
fig.savefig("outputs/figures/prediction_report_metrics.svg", bbox_inches="tight")
fig.savefig("outputs/figures/prediction_report_metrics.png", bbox_inches="tight", dpi=130)
plt.close(fig)

seq = s.filter(pl.col("metric") == "cumulative_square", pl.col("group") == "all", pl.col("model") == "recent")
fig, axes = plt.subplots(2, 2, figsize=(11, 5.5), sharex=True, sharey="row", gridspec_kw={"height_ratios": [1.4, 1]})
for col, speed in enumerate(["Bullet", "Blitz"]):
    for fold, color in [("Jul-Sep", "#b05a16"), ("Oct-Dec", "#247a70")]:
        for weighting, style, marker in [("window", "-", "o"), ("player", "--", "s")]:
            d = seq.filter(pl.col("speed") == speed, pl.col("fold") == fold, pl.col("weighting") == weighting).sort("length")
            axes[0, col].plot(d["length"], d["observed"], linestyle=style, marker=marker, color=color, label=f"{fold}, {weighting}")
        axes[1, col].plot(d["length"], d["n_players"], marker="o", color=color)
        for r in d.iter_rows(named=True):
            if r["n_players"] > 0:
                axes[1, col].annotate(str(r["n_players"]), (r["length"], r["n_players"]), xytext=(0, -14 if fold == "Jul-Sep" else 7), textcoords="offset points", ha="center", fontsize=8, color=color)
    axes[0, col].axhline(1, color="grey", linestyle=":")
    axes[0, col].set_title(speed)
    axes[1, col].set(xlabel="Window length (games)", yscale="log", xticks=sorted(seq["length"].unique().to_list()))
axes[0, 0].set_ylabel("Cumulative deviation")
axes[1, 0].set_ylabel("Players (log scale)")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=9)
fig.tight_layout(rect=(0, 0, 1, .92))
fig.savefig("outputs/figures/prediction_report_support.svg", bbox_inches="tight")
fig.savefig("outputs/figures/prediction_report_support.png", bbox_inches="tight", dpi=130)
plt.close(fig)
