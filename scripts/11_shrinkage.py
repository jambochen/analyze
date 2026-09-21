import matplotlib.pyplot as plt
import numpy as np
import polars as pl

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK TC", "Noto Sans CJK SC", "Noto Sans CJK JP"]
plt.rcParams["axes.unicode_minus"] = False

d = pl.read_csv("outputs/tables/13_priority_list.csv")

# 樣本量在公司間差距達一個數量級以上，估計雜訊大的公司不應因雜訊排前面。
# 經驗貝氏：把各公司的係數向整體均值收縮，收縮量由該公司的標準誤決定。
# 不用 MixedLM——近百萬列的隨機效應模型過慢，而此處收縮量手算即可。
out = d.select("carrier", "n", "regional")
for tag in ["y1", "y2", "y3"]:
    b = d[tag].to_numpy()
    se = d[f"{tag}_se"].to_numpy()
    mu = float(np.average(b, weights=1.0 / se**2))
    # 矩估計：觀測變異扣掉抽樣變異即為真實的公司間變異。負值代表公司間
    # 沒有超出雜訊的差異，截為 0（此時全部收縮到均值）。
    tau2 = max(float(np.mean((b - mu) ** 2 - se**2)), 0.0)
    shrink = se**2 / (se**2 + tau2)
    post = mu + (1 - shrink) * (b - mu)
    post_se = np.sqrt(tau2 * se**2 / (tau2 + se**2))
    print(f"{tag}: 整體均值 {mu:.4f} tau {np.sqrt(tau2):.4f} 收縮量 {shrink.min():.3f}-{shrink.max():.3f}")
    out = out.with_columns(
        pl.Series(f"{tag}_raw", b),
        pl.Series(f"{tag}_eb", post),
        pl.Series(f"{tag}_eb_lo", post - 1.96 * post_se),
        pl.Series(f"{tag}_eb_hi", post + 1.96 * post_se),
        pl.Series(f"{tag}_shrink", shrink),
    )

out = out.with_columns(
    sum((pl.col(f"{t}_eb_lo") > 0).cast(pl.Int8) for t in ["y1", "y2", "y3"]).alias("n_pos_eb"),
).sort("n_pos_eb", "y1_eb", descending=[True, True])
print(out.select("carrier", "n", "n_pos_eb", "y1_raw", "y1_eb", "y1_shrink", "y3_raw", "y3_eb", "y3_shrink"))
out.write_csv("outputs/tables/14_shrunk_list.csv")

raw_rank = d.sort("y1", descending=True)["carrier"].to_list()
eb_rank = out.sort("y1_eb", descending=True)["carrier"].to_list()
print("\nY1 排序 收縮前", raw_rank[:5])
print("Y1 排序 收縮後", eb_rank[:5])
moved = [(c, raw_rank.index(c) - eb_rank.index(c)) for c in raw_rank if raw_rank.index(c) != eb_rank.index(c)]
print("名次改變者", moved)

prev = pl.read_csv("outputs/tables/13_priority_list.csv").select("carrier", "n_pos")
chk = out.join(prev, on="carrier").filter(pl.col("n_pos") != pl.col("n_pos_eb"))
print("\n收縮後顯著項數改變者")
print(chk.select("carrier", "n", "n_pos", "n_pos_eb"))

fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), sharey=True)
order = out["carrier"].to_list()[::-1]
pos = {k: i for i, k in enumerate(order)}
for ax, tag, title in zip(axes, ["y1", "y2", "y3"],
                          ["Y1 是否歸因天氣", "Y2 歸因時掛多少比例", "Y3 天氣 vs NAS 邊界"]):
    t = out.with_columns(pl.col("carrier").replace_strict(pos).alias("pos"))
    ax.scatter(t[f"{tag}_raw"], t["pos"], s=18, facecolors="none", edgecolors="gray", label="收縮前")
    ax.errorbar(t[f"{tag}_eb"], t["pos"],
                xerr=[t[f"{tag}_eb"] - t[f"{tag}_eb_lo"], t[f"{tag}_eb_hi"] - t[f"{tag}_eb"]],
                fmt="o", ms=4, lw=1, color="C0", label="收縮後")
    ax.axvline(0, ls="--", lw=0.8, color="gray")
    ax.set_title(title, fontsize=10)
axes[0].set_yticks(range(len(order)))
axes[0].set_yticklabels(order, fontsize=8)
axes[0].legend(fontsize=8, loc="lower right")
fig.supxlabel("相對 DL 的效應（2022-2024，經驗貝氏收縮）")
fig.tight_layout()
fig.savefig("outputs/figures/14_shrunk_list.png", dpi=150)
