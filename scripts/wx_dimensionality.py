# 氣象欄位的統計自由度：106 個 wx_ 欄實際承載幾個獨立方向。
# 用未篩天氣的 audit_clean，不用 bad_weather——後者的氣象變異被 IFR/LIFR
# 截斷，相關結構不是這份資料集本身的結構。
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

mpl.rcParams["font.sans-serif"] = ["Noto Sans CJK TC"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["font.size"] = 15
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.right"] = False

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
GREY = "#8a8a86"
FAINT = "#d6d5d1"
INK = "#0b0b0b"
SURFACE = "#fcfcfb"

sc = pl.scan_parquet("data/processed/audit_clean.parquet").collect_schema()
# _absent 與 obs_missing 是缺失指示，不是氣象測量，排除。
wx = [
    c for c in sc.names()
    if c.startswith("wx_") and sc[c] in (pl.Float64, pl.Int64, pl.Int32)
    and not c.endswith("_absent") and "obs_missing" not in c
]
print(len(wx), "numeric wx columns")

d = pl.read_parquet("data/processed/audit_clean.parquet", columns=wx).drop_nulls()
print("complete cases", d.height, "/", pl.scan_parquet("data/processed/audit_clean.parquet").select(pl.len()).collect().item())

# ISD 以 22000 表示無雲冪（見 data_dictionary.md）。該值是真實雲冪 p99 的
# 兩倍餘且佔四成五，留在連續欄裡會使該欄實質退化為二元指示——原樣時
# corr(ceiling_m, is_unlimited) 達 0.98。故與雪深同法拆為指示加數值：
# 無限者取各欄自身的非哨兵 p99 截頂，高度的變異才不被哨兵主導。
ceiling = [c for c in wx if c.endswith("ceiling_m")]
for c in ceiling:
    unlimited = pl.col(c) == 22000
    cap = d.get_column(c).filter(~d.get_column(c).eq(22000)).quantile(0.99)
    d = d.with_columns(
        unlimited.cast(pl.Float64).alias(c + "_unlimited"),
        pl.when(unlimited).then(cap).otherwise(pl.col(c)).alias(c),
    )
wx = wx + [c + "_unlimited" for c in ceiling]
print(len(ceiling), "ceiling columns split into indicator + capped height")

X = d.select(wx).to_numpy().astype(float)
X = (X - X.mean(0)) / X.std(0)
R = np.corrcoef(X, rowvar=False)
ev = np.linalg.eigvalsh(R)[::-1]
ratio = ev / ev.sum()
cum = np.cumsum(ratio)

k90 = int(np.searchsorted(cum, 0.90) + 1)
k95 = int(np.searchsorted(cum, 0.95) + 1)
# Kaiser 準則：相關矩陣的特徵值 > 1 表該成分解釋量多於單一原始欄位。
kaiser = int((ev > 1).sum())
print(f"n={len(wx)} k90={k90} k95={k95} kaiser={kaiser}")
print("cond", round(float(ev[0] / ev[-1]), 1))

pl.DataFrame({
    "pc": range(1, len(ev) + 1), "eigenvalue": ev, "ratio": ratio, "cumulative": cum,
}).write_csv("outputs/tables/wx_eigenvalues.csv")

# 欄位的階層聚類：距離取 1 - |corr|，故強負相關（能見度 vs 雲冪）視為同一方向。
dist = 1 - np.abs(R)
np.fill_diagonal(dist, 0)
Z = linkage(squareform(dist, checks=False), method="average")

# 群數取 k90，檢驗統計自由度與命名模板是否給出一致的分群數。
lab = fcluster(Z, k90, criterion="maxclust")
groups = pl.DataFrame({"col": wx, "cluster": lab}).sort("cluster")
print(groups.group_by("cluster").agg(pl.col("col").str.replace("wx_", "").alias("cols")).sort("cluster"))
groups.write_csv("outputs/tables/wx_column_clusters.csv")


def tag(c):
    end = "dest" if "_dest_" in c else "origin"
    lagm = [m for m in ("m1", "m3", "m6") if f"_{m}_" in c]
    return end, lagm[0] if lagm else "t0"


# 交叉表：若群主要沿「氣象量」切開而非沿「端」或「落後時點」，
# 表示冗餘來自空間與時間相關，與命名模板的展開方向一致。
ct = pl.DataFrame({
    "cluster": lab,
    "end": [tag(c)[0] for c in wx],
    "lag": [tag(c)[1] for c in wx],
})
print(ct.group_by("cluster").agg(
    pl.col("end").eq("origin").sum().alias("origin"), pl.col("end").eq("dest").sum().alias("dest"),
    pl.col("lag").n_unique().alias("n_lags"), pl.len().alias("n"),
).sort("cluster"))

fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1.35, 1]})

ax = axes[0]
# 著色門檻取 0.2：相關係數約 0.8 以上才視為同一方向。k90 切點會產生
# 三十餘群，顏色反而讀不出結構。
dendrogram(Z, ax=ax, no_labels=True, color_threshold=0.2, above_threshold_color=FAINT)
ax.axhline(0.2, color=INK, lw=1, ls="--")
ax.set_ylabel("1 − |相關係數|")
ax.set_xlabel(f"{len(wx)} 個數值型氣象欄位")
ax.annotate("|r| > 0.8 才併為同一方向", xy=(0.98, 0.2), xycoords=("axes fraction", "data"),
            xytext=(0, 6), textcoords="offset points", ha="right", color=INK, fontsize=13)

ax = axes[1]
ax.plot(range(1, len(cum) + 1), cum, color=BLUE, lw=2)
for k, c, lab_, dy in [(k90, ORANGE, "90%", 0.34), (k95, GREY, "95%", 0.20)]:
    ax.axvline(k, color=c, lw=1, ls="--")
    ax.annotate(f"{lab_} 變異：{k} 維", xy=(k, dy), xytext=(-8, 0), textcoords="offset points",
                color=c, fontsize=14, va="center", ha="right")
ax.annotate(f"原始 {len(wx)} 欄", xy=(len(wx), 1.0), xytext=(-6, -10),
            textcoords="offset points", color=GREY, fontsize=13, ha="right", va="top")
ax.set_xlabel("主成分數")
ax.set_ylabel("累積解釋變異")
ax.set_ylim(0, 1.02)

for a in axes:
    a.set_facecolor(SURFACE)
    for s in a.spines.values():
        s.set_color(FAINT)
        s.set_linewidth(1)
    a.tick_params(colors=INK, length=3, width=1, color=FAINT)
fig.patch.set_facecolor(SURFACE)
fig.tight_layout()
fig.savefig("outputs/figures/slides_wx_dimensionality.png", dpi=200, facecolor=SURFACE)

# 前幾個主成分由哪些欄主導，判斷是否對應可解釋的物理維度。
w, V = np.linalg.eigh(R)
V = V[:, ::-1]
for k in range(5):
    load = pl.DataFrame({"col": wx, "load": V[:, k]}).with_columns(pl.col("load").abs().alias("a")).sort("a", descending=True)
    print(f"PC{k+1} ({ratio[k]:.1%})", [(r["col"].replace("wx_", ""), round(r["load"], 2)) for r in load.head(6).iter_rows(named=True)])

# flight_category_ordinal 由能見度與雲冪按 AWC 門檻導出，非獨立觀測
# （corr 與自身輸入達 -0.78），與輸入並列會把同一資訊算兩次而推高 PC1。
# 排除後重算，確認「冗餘低」的結論不是這層重複造成的。
keep = [c for c in wx if "flight_category_ordinal" not in c]
Xk = d.select(keep).to_numpy().astype(float)
Xk = (Xk - Xk.mean(0)) / Xk.std(0)
evk = np.linalg.eigvalsh(np.corrcoef(Xk, rowvar=False))[::-1]
cumk = np.cumsum(evk / evk.sum())
print(f"excl. flight_category_ordinal: n={len(keep)} "
      f"k90={int(np.searchsorted(cumk, 0.90) + 1)} k95={int(np.searchsorted(cumk, 0.95) + 1)} "
      f"kaiser={int((evk > 1).sum())} pc1={evk[0] / evk.sum():.1%}")
