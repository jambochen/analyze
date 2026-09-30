import matplotlib.pyplot as plt
import polars as pl

import stratum

b = stratum.load()

# statsmodels 的 GLM 不吃 sparse，而 ConditionalLogit 以遞迴計算條件似然的分母，
# 層上千列即遞迴溢位——它是為配對研究的小 strata 設計的。三個結果變數統一以
# 層內去均值吸收固定效應。Y1 為二元，係數直接是機率差。
for col, label in [("y1_attributed", "y1"), ("y2_share", "y2"), ("y3_wx_vs_nas", "y3")]:
    out = stratum.fit(b, col)
    print(out.filter(pl.col("term").str.starts_with("carrier_")).sort("coef"))
    out.write_csv(f"outputs/tables/carrier_effects_{label}.csv")

# 雙向 cluster 的前提：機尾號與層若高度巢狀，層的 cluster 即已吸收機隊層相關，
# 無須雙向。跨度隨樣本界定而變，故兩個口徑並列。
rows = []
for tag, d in [("bad_weather", b), ("本腳本樣本", stratum.informative(b.drop_nulls(stratum.WX), "y1_attributed"))]:
    t = d.group_by("tail_number").agg(pl.struct("origin_iata", "ym").n_unique().alias("c"))
    c = d.group_by("origin_iata", "ym").agg(pl.len().alias("n"))
    rows.append({"sample": tag, "tail_cells_median": int(t["c"].median()),
                 "cell_flights_median": int(c["n"].median())})
cc = pl.DataFrame(rows)
cc.write_csv("outputs/tables/cluster_crossing.csv")
print(cc)

# 三圖共用 y 軸，須固定為同一順序——各自 sort 會讓標籤對不上公司。
order = pl.read_csv("outputs/tables/carrier_effects_y1.csv").filter(pl.col("term").str.starts_with("carrier_")).sort("coef")["term"].to_list()

fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=True)
for ax, label, title in zip(axes, ["y1", "y2", "y3"], ["Y1 weather attributed", "Y2 weather share when attributed", "Y3 weather vs NAS margin"]):
    t = pl.read_csv(f"outputs/tables/carrier_effects_{label}.csv").filter(pl.col("term").str.starts_with("carrier_"))
    t = t.with_columns(pl.col("term").replace_strict({k: i for i, k in enumerate(order)}).alias("pos")).sort("pos")
    ax.errorbar(t["coef"], t["pos"], xerr=1.96 * t["se"], fmt="o", ms=3, lw=1)
    ax.axvline(0, ls="--", lw=0.8, color="gray")
    ax.set_title(title, fontsize=10)
axes[0].set_yticks(range(len(order)))
axes[0].set_yticklabels([x.replace("carrier_", "") for x in order], fontsize=8)
fig.supxlabel("Effect vs DL (within airport x month, ordered by Y1)")
fig.tight_layout()
fig.savefig("outputs/figures/carrier_effects.png", dpi=150)
