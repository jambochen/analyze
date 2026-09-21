# 投影片專用圖：字級與長寬比為 16:9 簡報設計，與報告用圖分開
import matplotlib.pyplot as plt
import matplotlib as mpl
import polars as pl

mpl.rcParams["font.sans-serif"] = ["Noto Sans CJK TC"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["font.size"] = 15

BLUE = "#2b6cb0"
RED = "#c53030"
GREY = "#a0aec0"

cs = pl.read_csv("outputs/tables/cause_share.csv").sort("weather")
fig, ax = plt.subplots(figsize=(11, 5.4))
bottom = [0.0] * cs.height
for col, color, label in [
    ("weather", BLUE, "天氣"),
    ("nas", "#63b3ed", "NAS"),
    ("carrier", "#feb2b2", "carrier"),
    ("late_aircraft", GREY, "late-aircraft"),
]:
    v = cs[col].to_list()
    ax.bar(cs["operating_carrier"], v, bottom=bottom, color=color, label=label)
    bottom = [b + x for b, x in zip(bottom, v)]
ax.set_ylabel("延誤分鐘數佔比")
ax.set_xlabel("執飛航空公司")
ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
ax.set_ylim(0, 1)
fig.tight_layout()
fig.savefig("outputs/figures/slides_cause_share.png", dpi=200)

pl_df = pl.read_csv("outputs/tables/priority_list.csv").sort("y1")
fig, ax = plt.subplots(figsize=(11, 5.4))
y = range(pl_df.height)
colors = [RED if v > 0 else BLUE for v in pl_df["y1"]]
ax.barh(list(y), pl_df["y1"].to_list(), color=colors)
ax.errorbar(
    pl_df["y1"].to_list(), list(y),
    xerr=[
        (pl_df["y1"] - pl_df["y1_lo"]).to_list(),
        (pl_df["y1_hi"] - pl_df["y1"]).to_list(),
    ],
    fmt="none", ecolor="#2d3748", capsize=3, lw=1.2,
)
ax.set_yticks(list(y))
ax.set_yticklabels(pl_df["carrier"].to_list())
ax.axvline(0, color="#2d3748", lw=1)
ax.set_xlabel("Y1 公司效應（控制機場 × 年月與實測氣象後的機率差）")
fig.tight_layout()
fig.savefig("outputs/figures/slides_priority_y1.png", dpi=200)

fig, ax = plt.subplots(figsize=(11, 5.4))
ax.axhline(0, color="#2d3748", lw=1)
ax.axvline(0, color="#2d3748", lw=1)
ax.scatter(pl_df["y1"], pl_df["y2"], s=90, color=BLUE, zorder=3)
for c, x, yy in zip(pl_df["carrier"], pl_df["y1"], pl_df["y2"]):
    ax.annotate(c, (x, yy), xytext=(6, 5), textcoords="offset points")
ax.set_xlabel("Y1：是否歸因天氣")
ax.set_ylabel("Y2：歸因時掛多少比例")
fig.tight_layout()
fig.savefig("outputs/figures/slides_y1_vs_y2.png", dpi=200)
