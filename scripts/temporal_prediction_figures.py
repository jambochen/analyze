# 時間外驗證圖與輸出一致性檢查；遊戲分開作圖，避免把原始損失跨遊戲排名。
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import polars as pl

p = pl.read_csv("outputs/tables/temporal_prediction_prediction.csv")
s = pl.read_csv("outputs/tables/temporal_prediction_sample.csv")
f = pl.read_csv("outputs/tables/temporal_prediction_fits.csv")
c = pl.read_csv("outputs/tables/temporal_prediction_calibration.csv")
assert s.filter(pl.col("segment") == "test").height == 9
assert p.select(pl.struct("game", "speed", "fold", "group", "model", "metric").n_unique()).item() == p.height
assert p.select(pl.all_horizontal(pl.col("mean", "diff", "se", "lo", "hi").is_finite()).all()).item()
assert f["score_scaled_max"].max() < 0.01
for game, speed, fold in s.select("game", "speed", "fold").unique().rows():
    rows = s.filter(pl.col("game") == game, pl.col("speed") == speed, pl.col("fold") == fold)
    n = rows.filter(pl.col("segment") == "test")["n"][0]
    for model in ["system", "baseline", "recent", "novice_extension"]:
        assert c.filter(pl.col("game") == game, pl.col("speed") == speed, pl.col("fold") == fold, pl.col("group") == "all", pl.col("model") == model)["n"].sum() == n

for game in ["Lichess", "OGS", "AoE2"]:
    g = p.filter(pl.col("game") == game, pl.col("metric") == "log_loss")
    panels = g.select("speed", "fold").unique().sort("speed", "fold").rows()
    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, len(panels) * 0.8)), squeeze=False)
    for ax, model in zip(axes[0], ["recent", "novice_extension"]):
        if game == "AoE2" and model == "novice_extension":
            ax.axis("off")
            ax.text(0.5, 0.5, "RD extension not evaluated\nRD unavailable in this dataset", ha="center", va="center", transform=ax.transAxes)
            continue
        for j, (group, color) in enumerate([("all", "#555555"), ("novice", "#d95f02"), ("established", "#1b9e77")]):
            d = g.filter(pl.col("model") == model, pl.col("group") == group).sort("speed", "fold")
            y = np.arange(len(panels)) + (j - 1) * 0.19
            ax.errorbar(d["diff"], y, xerr=1.96 * d["se"].to_numpy(), fmt="o", markersize=4, color=color, label=group, capsize=3)
        ax.axvline(0, color="grey", linestyle="--")
        ax.set(yticks=np.arange(len(panels)), yticklabels=[f"{a}: {b}" for a, b in panels], xlabel="Log-loss difference (lower is better)", title="Recent vs baseline" if model == "recent" else "Novice extension vs recent")
        ax.set_ylim(len(panels) - 0.6, -0.6)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))
        ax.legend(fontsize=8)
    title = "OGS (end-time order only)" if game == "OGS" else game
    fig.suptitle(f"{title}: temporal evaluation, conditional 95% intervals")
    fig.text(0.5, 0.01, "Zero differences can mean that no additional terms were selected.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(f"outputs/figures/temporal_prediction_{game.lower()}_comparison.png", dpi=150)
    plt.close(fig)

pl.DataFrame([
    {"game": "Lichess", "baseline": "RD ordinal", "outcome": "loss/draw/win", "history": "same arena, previous game", "novice": "either RD >= 110", "evaluation": "2014 Jul-Sep; Oct-Dec", "limitation": "exploratory reused data; conditional intervals"},
    {"game": "OGS", "baseline": "RD binary logistic", "outcome": "loss/win", "history": "end-time order, same day", "novice": "either RD >= 160", "evaluation": "2023", "limitation": "no start time; pregame availability not verified"},
    {"game": "AoE2", "baseline": "binary logistic without RD", "outcome": "loss/win", "history": "rating chain, previous completion <= start", "novice": "either prior games < 30", "evaluation": "2026 Jun24-Jul03", "limitation": "single month; Elo proxy, not verified system forecast"},
]).write_csv("outputs/tables/temporal_prediction_design.csv")
print("Temporal output checks passed")
print(f.group_by("game").agg(fits=pl.len(), fit_seconds=pl.col("seconds").sum(), score_scaled_max=pl.col("score_scaled_max").max()))

# 保留各段實際選中的欄位，係數表可直接對應回模型。
selection = pl.read_csv("outputs/tables/temporal_prediction_selection.csv")
coef = pl.read_csv("outputs/tables/temporal_prediction_coef.csv")
models, labeled = [], []
for game, speed, fold in selection.select("game", "speed", "fold").unique().sort("game", "speed", "fold").rows():
    ss = selection.filter(pl.col("game") == game, pl.col("speed") == speed, pl.col("fold") == fold)
    chosen, recent = [], []
    for row in ss.iter_rows(named=True):
        if row["kept"]:
            chosen = row["columns"].split(",")
        if row["step"] == "rating_change":
            recent = chosen.copy()
    for name, cols in [("recent", recent), ("novice_extension", chosen)]:
        models.append({"game": game, "speed": speed, "fold": fold, "model": name, "columns": ",".join(cols)})
        fit = "recent_refit" if name == "recent" or chosen == recent else "novice_refit"
        cc = coef.filter(pl.col("game") == game, pl.col("speed") == speed, pl.col("fold") == fold, pl.col("fit") == fit)
        for row in cc.iter_rows(named=True):
            labeled.append({"game": game, "speed": speed, "fold": fold, "model": name, "param": (["baseline_offset"] + cols)[row["param_index"]], "est": row["est"]})
pl.DataFrame(models).write_csv("outputs/tables/temporal_prediction_models.csv")
pl.DataFrame(labeled).write_csv("outputs/tables/temporal_prediction_state_coef.csv")
