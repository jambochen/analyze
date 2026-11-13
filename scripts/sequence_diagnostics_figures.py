# 全樣本分別展示視窗等權及玩家等權；自身 RD 分層保留於摘要表。
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

s = pl.read_csv("outputs/tables/sequence_diagnostics_summary.csv")
for weighting in ["window", "player"]:
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    colors = ["#4c78a8", "#f58518", "#54a24b"]
    models = ["baseline", "recent", "novice_extension"]
    labels = ["Baseline", "Recent history", "RD extension"]
    for col, (speed, fold) in enumerate([(s, f) for s in ["Bullet", "Blitz"] for f in ["Jul-Sep", "Oct-Dec"]]):
        for row, metric in enumerate(["loss_run", "cumulative_square"]):
            ax = axes[row, col]
            for model, color, label in zip(models, colors, labels):
                d = s.filter(pl.col("speed") == speed, pl.col("fold") == fold, pl.col("metric") == metric, pl.col("model") == model, pl.col("group") == "all", pl.col("weighting") == weighting).sort("length")
                x = d["length"].to_numpy()
                denom = d["analytic_reference"].to_numpy()
                ax.plot(x, d["observed"].to_numpy()/denom, "o-", color=color, label=label, markersize=4)
                ax.fill_between(x, d["reference_lo"].to_numpy()/denom, d["reference_hi"].to_numpy()/denom, color=color, alpha=.12)
            if row == 1:
                last = d.row(-1, named=True)
                ax.text(.03, .97, f"20 games: {last['n_windows']:,} windows / {last['n_players']:,} players", transform=ax.transAxes, va="top", fontsize=8)
            ax.axhline(1, color="black", linewidth=.8, linestyle="--")
            ax.set_xticks(x)
            ax.set_xlabel("Window length (games)")
            ax.grid(alpha=.15)
            if row == 0:
                ax.set_title(f"{speed} / {fold}")
            if col == 0:
                ax.set_ylabel("Loss-run rate / reference" if row == 0 else "Mean squared cumulative residual\n/ independent variance")
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.5, .96))
    fig.suptitle(f"Observed sequences versus frozen-probability reference ({weighting} weighted)", y=.99, fontsize=14)
    fig.text(.5, .015, f"All eligible windows, {weighting} weighted; panel scales differ. Shading: central 95% simulation range; not a confidence interval.\nObserved histories and continuation are fixed; this is not a generative null test.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .08, 1, .92))
    fig.savefig("outputs/figures/sequence_diagnostics" + ("" if weighting == "window" else "_player") + ".png", dpi=170)
    plt.close(fig)
