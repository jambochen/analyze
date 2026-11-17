# 教學模擬使用固定亂數與相同散點，讓差別只來自曲線形式；不作本題的實證證據。
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from patsy import cr

rng = np.random.default_rng(2026)
x = np.linspace(-3, 3, 120)
mean = 1.2 * np.tanh(2 * x) + 0.3 * x
y = mean + rng.normal(0, 0.22, len(x))
grid = np.linspace(-3, 3, 400)
knots = [-1, 0, 1]
designs = [
    ("Linear", np.vander(x, 2, increasing=True), np.vander(grid, 2, increasing=True)),
    ("Cubic polynomial", np.vander(x, 4, increasing=True), np.vander(grid, 4, increasing=True)),
    ("Natural cubic spline", np.asarray(cr(x, knots=knots, lower_bound=-3, upper_bound=3)),
     np.asarray(cr(grid, knots=knots, lower_bound=-3, upper_bound=3))),
]

pl.DataFrame({"x": x, "y": y, "generating_mean": mean}).write_csv("outputs/tables/spline_illustration_points.csv")
plt.rcParams.update({"svg.fonttype": "path", "font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharex=True, sharey=True)
curves = []
for ax, (label, design, prediction_design) in zip(axes, designs):
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    predicted = prediction_design @ coef
    curves.append(pl.DataFrame({"model": label, "x": grid, "predicted": predicted}))
    ax.scatter(x, y, color="#9a9fa4", s=13, alpha=0.65, edgecolors="none")
    ax.plot(grid, predicted, color="#33414a", linewidth=2)
    if label == "Natural cubic spline":
        for i, knot in enumerate(knots):
            ax.axvline(knot, color="#9c7a5b", linestyle="--", linewidth=0.9, label="Interior knots" if i == 0 else None)
        ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none", framealpha=0.95, fontsize=10)
    ax.set(title=label, xlabel="x", xticks=[-3, -1, 0, 1, 3])
    ax.grid(axis="y", alpha=0.12)
axes[0].set_ylabel("y")
fig.tight_layout()
fig.savefig("outputs/figures/spline_illustration.svg", bbox_inches="tight")
plt.close(fig)
pl.concat(curves).write_csv("outputs/tables/spline_illustration_curves.csv")
