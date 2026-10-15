"""方法報告的模擬流程示意圖。"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


labels = [
    ("Keep observed\nflights", "Carrier, weather,\nroute, delay"),
    ("Generate latent\ncauses", "Four nonnegative\nminute components"),
    ("Inject reporting\nshifts", "Selected carriers;\nfixed minute total"),
    ("Refit both\nmodels", "Same analysis\nrules"),
    ("Check the\naudit list", "Compare flagged and\nplanted carriers"),
]
fills = ["#e9f2fd", "#f1f2f3", "#fff0e6", "#e9f2fd", "#e9f5ec"]
edges = ["#2a78d6", "#6b7280", "#e87232", "#2a78d6", "#2f8a57"]

fig, ax = plt.subplots(figsize=(11.2, 2.8))
fig.patch.set_facecolor("white")
ax.set_xlim(0, 11.2)
ax.set_ylim(0, 2.8)
ax.axis("off")

width = 1.95
gap = 0.29
left = 0.12
bottom = 0.65
height = 1.55

for i, ((title, detail), fill, edge) in enumerate(zip(labels, fills, edges)):
    x = left + i * (width + gap)
    ax.add_patch(FancyBboxPatch(
        (x, bottom), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.11",
        facecolor=fill, edgecolor=edge, linewidth=1.4,
    ))
    ax.text(x + width / 2, bottom + 1.15, title, ha="center", va="center",
            fontsize=10.2, weight="bold", linespacing=1.15, color="#1e293b")
    ax.text(x + width / 2, bottom + 0.49, detail, ha="center", va="center",
            fontsize=9.0, linespacing=1.25, color="#344054")
    if i < len(labels) - 1:
        ax.add_patch(FancyArrowPatch(
            (x + width + 0.025, bottom + height / 2),
            (x + width + gap - 0.025, bottom + height / 2),
            arrowstyle="-|>", mutation_scale=13, linewidth=1.2, color="#64748b",
        ))

ax.text(0.15, 0.30, "Fixed flight structure", fontsize=9.5, color="#2a78d6")
ax.text(4.58, 0.30, "Known causes and shifts", fontsize=9.5, color="#e87232")
ax.text(8.98, 0.30, "Carrier-level evaluation", fontsize=9.5, color="#2f8a57")

Path("outputs/figures").mkdir(parents=True, exist_ok=True)
fig.savefig("outputs/figures/methods_simulation_flow.svg", bbox_inches="tight")
plt.close(fig)
