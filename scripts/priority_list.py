import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import stats
from statsmodels.stats.multitest import multipletests

import stratum

b = stratum.load().drop_nulls(stratum.WX)


def effects(d, outcome):
    return (
        stratum.fit(d, outcome)
        .filter(pl.col("term").str.starts_with("carrier_"))
        .with_columns(pl.col("term").str.replace("carrier_", "").alias("carrier"))
        .select("carrier", "coef", "se")
    )


# hold-out 顯示係數水準會隨時期漂移，故主清單只用近期資料；前期僅用於算變化。
recent, prior = b.filter(pl.col("yr") >= 2022), b.filter(pl.col("yr") <= 2021)

parts = []
for outcome, tag in [("y1_attributed", "y1"), ("y2_share", "y2"), ("y3_wx_vs_nas", "y3")]:
    a = effects(recent, outcome).rename({"coef": f"{tag}", "se": f"{tag}_se"})
    p = effects(prior, outcome).select("carrier", pl.col("coef").alias(f"{tag}_prior"))
    parts.append(a.join(p, on="carrier", how="left").with_columns(
        (pl.col(f"{tag}") - pl.col(f"{tag}_prior")).alias(f"{tag}_chg"),
        (pl.col(f"{tag}") - 1.96 * pl.col(f"{tag}_se")).alias(f"{tag}_lo"),
        (pl.col(f"{tag}") + 1.96 * pl.col(f"{tag}_se")).alias(f"{tag}_hi"),
    ))

lst = parts[0]
for q in parts[1:]:
    lst = lst.join(q, on="carrier", how="outer_coalesce")

# 反向檢核：營運表現差與回報不誠實是兩個量。表現差但誠實者不該進清單。
perf = (
    recent.group_by("operating_carrier")
    .agg(
        pl.len().alias("n"),
        pl.col("arr_delay_min").median().alias("delay_median"),
        pl.col("y1_attributed").mean().alias("raw_y1"),
        pl.col("is_regional").mean().alias("regional"),
    )
    .rename({"operating_carrier": "carrier"})
)
lst = lst.join(perf, on="carrier", how="left")

# 第二個反向檢核：取消班的 arr_delay_min 為 null，被申報門檻排除，故稽核樣本
# 看不到它。但取消也是一次天氣歸因決策，一家公司可以在延誤欄少報天氣卻以天氣
# 為由大量取消。歸因偏低者須並列此欄，否則「以取消消化天氣」會被誤讀為誠實申報。
# 須回讀原始檔，惡劣天氣定義與 clean.py 一致。
cxl = (
    pl.scan_parquet([f"data/raw/flights-weather/flights_weather_{y}.parquet" for y in range(2022, 2025)])
    .select("operating_carrier", "cancelled", "cancellation_reason",
            "wx_origin_flight_category", "wx_dest_flight_category")
    .filter(
        pl.col("wx_origin_flight_category").is_in(["IFR", "LIFR"])
        | pl.col("wx_dest_flight_category").is_in(["IFR", "LIFR"])
    )
    .group_by("operating_carrier")
    .agg(
        pl.col("cancelled").mean().alias("cxl_rate"),
        # 分母為取消班：取消時有多少比例掛天氣。與 Y1「延誤時是否掛天氣」同形態，
        # 兩者才可並排讀。
        ((pl.col("cancellation_reason") == "Weather").sum() / pl.col("cancelled").sum()).alias("cxl_wx"),
    )
    .rename({"operating_carrier": "carrier"})
    .collect(engine="streaming")
)
lst = lst.join(cxl, on="carrier", how="left")

# 三個指標乘各家公司構成數十個同時檢定，未校正的區間會高估異常家數。
# 清單本身是排序不是檢定，但「幾項顯著」這個粗排序既然用了門檻，就須校正。
for t in ["y1", "y2", "y3"]:
    z = (lst[t] / lst[f"{t}_se"]).to_numpy()
    pv = 2 * (1 - stats.norm.cdf(np.abs(z)))
    lst = lst.with_columns(pl.Series(f"{t}_p", pv))

allp = np.concatenate([lst[f"{t}_p"].to_numpy() for t in ["y1", "y2", "y3"]])
rej = multipletests(allp, alpha=0.05, method="fdr_bh")[0].reshape(3, -1)
for i, t in enumerate(["y1", "y2", "y3"]):
    lst = lst.with_columns(pl.Series(f"{t}_bh", rej[i]))

lst = lst.with_columns(
    sum(((pl.col(f"{t}_lo") > 0) & pl.col(f"{t}_bh")).cast(pl.Int8) for t in ["y1", "y2", "y3"]).alias("n_pos"),
    sum(((pl.col(f"{t}_hi") < 0) & pl.col(f"{t}_bh")).cast(pl.Int8) for t in ["y1", "y2", "y3"]).alias("n_neg"),
    sum((pl.col(f"{t}_lo") > 0).cast(pl.Int8) for t in ["y1", "y2", "y3"]).alias("n_pos_raw"),
).sort("n_pos", "y1", descending=[True, True])

cols = ["carrier", "n", "regional", "n_pos", "n_neg", "n_pos_raw",
        "y1", "y1_se", "y1_lo", "y1_hi", "y1_chg", "y2", "y2_se", "y2_lo", "y2_hi", "y2_chg",
        "y3", "y3_se", "y3_lo", "y3_hi", "y3_chg", "raw_y1", "delay_median", "cxl_rate", "cxl_wx"]
lst = lst.select(cols)
print(lst)
lst.write_csv("outputs/tables/priority_list.csv")

print("\n三項皆顯著為正")
print(lst.filter(pl.col("n_pos") == 3).select("carrier", "y1", "y2", "y3", "y1_chg", "y3_chg", "delay_median"))
print("\n表現差但未進清單者（延誤中位高於整體，卻無指標顯著為正）")
med = float(recent["arr_delay_min"].median())
print(lst.filter((pl.col("delay_median") > med) & (pl.col("n_pos") == 0)).select("carrier", "delay_median", "y1", "n_pos"))

# 歸因顯著偏低者：取消天氣歸因率高者，其偏低可能只是改用取消欄消化天氣。
print("\n歸因顯著偏低者的取消檢核")
print(lst.filter(pl.col("n_neg") > 0).select("carrier", "n_neg", "y1", "raw_y1", "cxl_rate", "cxl_wx").sort("cxl_wx", descending=True))

fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), sharey=True)
order = lst["carrier"].to_list()[::-1]
for ax, tag, title in zip(axes, ["y1", "y2", "y3"],
                          ["Y1 weather attributed", "Y2 weather share when attributed", "Y3 weather vs NAS margin"]):
    t = lst.with_columns(pl.col("carrier").replace_strict({k: i for i, k in enumerate(order)}).alias("pos"))
    ax.errorbar(t[tag], t["pos"], xerr=1.96 * t[f"{tag}_se"], fmt="o", ms=4, lw=1)
    ax.axvline(0, ls="--", lw=0.8, color="gray")
    ax.set_title(title, fontsize=10)
axes[0].set_yticks(range(len(order)))
axes[0].set_yticklabels(order, fontsize=8)
fig.supxlabel("Effect vs DL (2022-2024, within airport x month)")
fig.tight_layout()
fig.savefig("outputs/figures/priority_list.png", dpi=150)
