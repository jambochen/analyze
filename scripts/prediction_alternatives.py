# 只重畫已完成的比較；不同基準與選模流程各自成圖，不合併成模型排行榜。
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

speed_labels = {"Bullet": "Short", "Blitz": "Medium", "Classical": "Long"}
speed_type = pl.Enum(list(speed_labels))

plt.rcParams.update({"svg.fonttype": "path", "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
colors = {"Bullet": "#b05a16", "Blitz": "#247a70", "Classical": "#555555"}


def differences(ax, d, labels, se="diff_se_dyadic", color="#247a70"):
    y = np.arange(d.height)
    ax.errorbar(d["diff"].to_numpy(), y, xerr=1.96*d[se].to_numpy(), fmt="o", color=color, capsize=3)
    ax.set(yticks=y, yticklabels=labels, xlabel="Log-loss difference (candidate - reference)")
    ax.invert_yaxis()
    ax.axvline(0, color="grey", linestyle="--", linewidth=1)
    ax.grid(axis="x", alpha=.15)


markov = pl.read_csv("outputs/tables/markov_order_prediction.csv").with_columns(pl.col("speed").cast(speed_type)).filter(pl.col("metric") == "log_loss", pl.col("k") > 1).sort("speed", "k")
hmm = pl.read_csv("outputs/tables/hmm_tilt_prediction.csv").with_columns(pl.col("speed").cast(speed_type)).filter(pl.col("metric") == "log_loss", pl.col("vs") == "prev_result", pl.col("model").is_in(["hmm2", "hmm3"])).sort("speed", "model")
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
differences(axes[0], markov, [f"{speed_labels[r['speed']]}: order {r['k']} vs {r['vs']}" for r in markov.iter_rows(named=True)])
differences(axes[1], hmm, [f"{speed_labels[r['speed']]}: {r['model'].upper()}" for r in hmm.iter_rows(named=True)])
axes[0].set_title("Longer result history")
axes[1].set_title("Hidden states vs previous result")
fig.tight_layout()
fig.savefig("outputs/figures/prediction_alternatives_history.svg", bbox_inches="tight")
plt.close(fig)

ml = pl.read_csv("outputs/tables/state_adjusted_probs_prediction.csv").with_columns(pl.col("speed").cast(speed_type)).filter(pl.col("metric") == "log_loss", pl.col("vs") == "opp_sym", pl.col("model").is_in(["lightgbm", "lightgbm_offset", "xgboost", "mlp"]), pl.col("speed").is_in(["Bullet", "Blitz"]))
names = {"lightgbm": "LightGBM", "lightgbm_offset": "LightGBM + baseline", "xgboost": "XGBoost", "mlp": "MLP"}
fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
for ax, group, title in zip(axes, ["all", "novice", "established"], ["All games", "At least one high-RD player", "Both players low-RD"]):
    d = ml.filter(pl.col("group") == group).sort("speed", "model")
    differences(ax, d, [f"{speed_labels[r['speed']]}: {names[r['model']]}" for r in d.iter_rows(named=True)])
    ax.set_title(title)
fig.tight_layout()
fig.savefig("outputs/figures/prediction_alternatives_flexible.svg", bbox_inches="tight")
plt.close(fig)

# 專案有序模型內的樣條比較避免混入 mgcv ocat 的不同和局切點設定。
gam = pl.read_csv("outputs/tables/explore_gam_in_project_family.csv").filter(pl.col("metric") == "log_loss", pl.col("group") == "all")
splines = {"P2": "RD splines", "P3": "Recent-change splines", "P4": "Both spline sets"}
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharex=True)
for ax, speed in zip(axes, ["bullet", "blitz"]):
    d = gam.filter(pl.col("speed") == speed).sort("model")
    differences(ax, d, [splines[r["model"]] for r in d.iter_rows(named=True)], se="se", color=colors[speed.title()])
    ax.set_title(f"{speed_labels[speed.title()]}: all player-game rows")
fig.tight_layout()
fig.savefig("outputs/figures/prediction_alternatives_splines.svg", bbox_inches="tight")
plt.close(fig)

prospective = pl.read_csv("outputs/tables/prospective_validation_log_score.csv").with_columns(pl.col("speed").cast(speed_type)).filter(pl.col("metric") == "log_loss").sort("speed")
fig, ax = plt.subplots(figsize=(10, 3.5))
differences(ax, prospective, [f"{speed_labels[r['speed']]} ({'registered' if r['registered'] else 'descriptive'})" for r in prospective.iter_rows(named=True)])
ax.set_title("Fixed recent-history model vs fixed RD baseline")
fig.tight_layout()
fig.savefig("outputs/figures/prediction_alternatives_prospective.svg", bbox_inches="tight")
plt.close(fig)

lobby = pl.read_csv("outputs/tables/state_adjusted_lobby_prediction.csv").with_columns(pl.col("speed").cast(speed_type)).filter(pl.col("metric") == "log_loss", pl.col("group") == "all", pl.col("gap_min") == 60)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharex=True)
for ax, model, title in zip(axes, ["opp_sym", "arena_model"], ["Fit recent terms on lobby", "Apply arena coefficients"]):
    d = lobby.filter(pl.col("model") == model, pl.col("vs") == "baseline").sort("speed")
    differences(ax, d, [speed_labels[value] for value in d["speed"]])
    ax.set_title(title)
fig.tight_layout()
fig.savefig("outputs/figures/prediction_alternatives_lobby.svg", bbox_inches="tight")
plt.close(fig)
