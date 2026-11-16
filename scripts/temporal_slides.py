# 第二次報告的時間外驗證圖；直接取正式輸出，不重估模型。
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import polars as pl

plt.rcParams.update({"svg.fonttype": "path", "font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
p = pl.read_csv("outputs/tables/temporal_prediction_prediction.csv").filter(pl.col("metric") == "log_loss")
for game, model in [("Lichess", "recent"), ("Lichess", "novice_extension"), ("OGS", "recent"), ("OGS", "novice_extension"), ("AoE2", "recent")]:
    g = p.filter(pl.col("game") == game, pl.col("model") == model)
    panels = g.select("speed", "fold").unique().sort("speed", "fold").rows()
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    for j, (group, color) in enumerate([("all", "#555555"), ("novice", "#eb6834"), ("established", "#1baf7a")]):
        d = g.filter(pl.col("group") == group).sort("speed", "fold")
        y = np.arange(len(panels)) + (j - 1) * 0.18
        ax.errorbar(d["diff"], y, xerr=1.96 * d["se"].to_numpy(), fmt="o", markersize=5, color=color, label=group, capsize=3)
    ax.axvline(0, color="grey", linestyle="--")
    ax.set(yticks=np.arange(len(panels)), yticklabels=[f"{a}: {b}" for a, b in panels], xlabel="Log-loss difference (lower is better)", ylim=(len(panels)-0.6, -0.6))
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(f"outputs/figures/temporal_slides_{game.lower()}_{model}.svg", bbox_inches="tight")
    plt.close(fig)

# 同時呈現校準與近況的增量，避免只見最後一步而漏掉校準造成的惡化。
for game in ["OGS", "AoE2"]:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    group_labels = (["All", "Either RD high", "Both RD low"] if game == "OGS"
                    else ["All", "Low cumulative games", "High cumulative games"])
    reference = "rating reference" if game == "OGS" else "Elo formula"
    for ax, model, vs, title in zip(axes, ["baseline", "recent"], ["system", "baseline"],
                                    [f"Calibrated baseline - {reference}", "Simple recent - calibrated baseline"]):
        g = p.filter(pl.col("game") == game, pl.col("model") == model, pl.col("vs") == vs)
        panels = g.select("speed", "fold").unique().sort("speed", "fold").rows()
        for j, (group, label, color) in enumerate(zip(["all", "novice", "established"], group_labels,
                                                    ["#555555", "#eb6834", "#1baf7a"])):
            d = g.filter(pl.col("group") == group).sort("speed", "fold")
            y = np.arange(len(panels)) + (j - 1) * .18
            ax.errorbar(d["diff"], y, xerr=1.96*d["se"].to_numpy(), fmt="o", color=color, label=label, capsize=3)
        ax.axvline(0, color="grey", linestyle="--")
        ax.set(title=title, yticks=np.arange(len(panels)), yticklabels=[f"{a}: {b}" for a, b in panels],
               xlabel="Log-loss difference (lower is better)", ylim=(len(panels)-.6, -.6))
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))
        ax.grid(axis="x", alpha=.15)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .87))
    fig.savefig(f"outputs/figures/temporal_slides_{game.lower()}_comparisons.svg", bbox_inches="tight")
    plt.close(fig)

pl.DataFrame([
    {"game":"Lichess", "fold":"Jul-Sep", "fit":"2014/1–5", "select":"2014/6", "refit":"2014/1–6", "test":"2014/7–9"},
    {"game":"Lichess", "fold":"Oct-Dec", "fit":"2014/1–8", "select":"2014/9", "refit":"2014/1–9", "test":"2014/10–12"},
    {"game":"OGS", "fold":"2023", "fit":"2021", "select":"2022", "refit":"2021–2022", "test":"2023"},
    {"game":"AoE2", "fold":"Jun24-Jul03", "fit":"2026/6/2–16", "select":"2026/6/17–23", "refit":"2026/6/2–23", "test":"2026/6/24–7/3"},
]).write_csv("outputs/tables/temporal_slides_periods.csv")
