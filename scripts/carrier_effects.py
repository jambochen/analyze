import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import statsmodels.api as sm


b = pl.read_parquet("data/processed/bad_weather.parquet")

# 時段設三段而非二分：航班量從未回到 2019 水準，2022 下半年起穩定在 0.89-0.92，
# 新常態與疫情前是不同的制度。切點判準見 docs/decisions.md。
b = b.with_columns(
    pl.when(pl.col("sched_dep_utc") < pl.lit("2020-04-01").str.to_datetime().dt.replace_time_zone("UTC"))
    .then(pl.lit("pre"))
    .when(pl.col("sched_dep_utc") < pl.lit("2021-06-01").str.to_datetime().dt.replace_time_zone("UTC"))
    .then(pl.lit("break"))
    .otherwise(pl.lit("new"))
    .alias("era"),
    pl.col("sched_dep_utc").dt.truncate("1mo").alias("ym"),
)

# 條件化於機場 × 年月：同一機場同一個月內比較不同公司，機場的物理條件與
# 當地航管慣例自動吸收。只保留格子內有多家公司且結果有變異者——單一公司或
# 結果全同的格子對公司效應沒有資訊。
cell = b.group_by(["origin_iata", "ym"]).agg(
    pl.len().alias("cn"), pl.col("operating_carrier").n_unique().alias("nc"), pl.col("y1_attributed").mean().alias("cr")
)
keep = cell.filter((pl.col("nc") >= 2) & (pl.col("cr") > 0) & (pl.col("cr") < 1)).select("origin_iata", "ym")
b = b.join(keep, on=["origin_iata", "ym"])
print("rows", b.height, "cells", keep.height)

wx = [
    "wx_origin_visibility_m", "wx_origin_ceiling_m", "wx_origin_wind_speed_ms", "wx_origin_wind_gust_ms",
    "wx_origin_precip_1h_mm", "wx_origin_snow_depth_cm", "wx_origin_temp_c", "wx_origin_sky_cover_oktas",
    "wx_origin_flight_category_ordinal", "wx_dest_flight_category_ordinal", "wx_dest_visibility_m",
    "wx_dest_ceiling_m", "wx_dest_wind_speed_ms", "wx_dest_precip_1h_mm",
    "wx_origin_thunderstorm", "wx_origin_snow", "wx_origin_freezing_precip", "wx_origin_fog",
    "wx_origin_m3_flight_category_ordinal", "wx_origin_m6_flight_category_ordinal",
    "arr_delay_min", "distance_miles", "origin_sched_departures_this_hour", "prior_leg_arr_delay_min",
    # 當日首航段的 prior_leg 填零是「不適用」而非「上段準點」，兩者形態相同，
    # 須留指示變數供模型分辨。與雪深、陣風的 _absent 同一原則。
    "prior_leg_available",
]
b = b.drop_nulls(wx)
print("after drop_nulls", b.height)

# 變化率不另建欄位：ordinal 與其 m3、m6 回看值三者皆已在 wx 中，相減得到的
# 差分與原欄位完全共線，會使設計矩陣降秩而標準誤變成 NaN。天氣惡化速度的
# 資訊由三個 ordinal 各自的係數承載。
num = wx

X_num = b.select(num).to_numpy().astype(float)
X_num = (X_num - X_num.mean(0)) / (X_num.std(0) + 1e-9)

carriers = sorted(b["operating_carrier"].unique().to_list())
ref = "DL"  # 樣本量大且歸因率居中，作為對照組
others = [c for c in carriers if c != ref]
X_car = np.column_stack([(b["operating_carrier"] == c).to_numpy().astype(float) for c in others])

cell_id = b.select(pl.struct("origin_iata", "ym").rank("dense").alias("i"))["i"].to_numpy() - 1
X = np.column_stack([X_num, X_car])
names = num + [f"carrier_{c}" for c in others]
print("design", X.shape, "cells", cell_id.max() + 1)


def report(coef, se, label):
    out = pl.DataFrame({"term": names, "coef": coef, "se": se}).with_columns(
        (pl.col("coef") - 1.96 * pl.col("se")).alias("lo"),
        (pl.col("coef") + 1.96 * pl.col("se")).alias("hi"),
    )
    print(out.filter(pl.col("term").str.starts_with("carrier_")).sort("coef"))
    out.write_csv(f"outputs/tables/carrier_effects_{label}.csv")


# 9,888 個格子的 dummy 展開為 dense 要 65 GB，statsmodels 的 GLM 不吃 sparse，
# 而 ConditionalLogit 以遞迴計算條件似然的分母，格子上千列即遞迴溢位——它是為
# 配對研究的小 strata 設計的。三個結果變數統一以格子內去均值吸收固定效應，
# 標準誤按格子叢集。Y1 為二元，去均值後即線性機率模型，係數直接是機率差。
for col, label in [("y1_attributed", "y1"), ("y2_share", "y2"), ("y3_wx_vs_nas", "y3")]:
    mask = b[col].is_not_null().to_numpy()
    y = b[col].fill_null(0).to_numpy().astype(float)[mask]
    g = cell_id[mask]
    dm = pl.DataFrame(X[mask], schema=names).with_columns(pl.Series("g", g), pl.Series("y", y))
    dd = dm.select([(pl.col(c) - pl.col(c).mean().over("g")).alias(c) for c in names + ["y"]])
    r = sm.OLS(dd["y"].to_numpy(), dd.select(names).to_numpy()).fit(cov_type="cluster", cov_kwds={"groups": g})
    print(label, "n", len(y), "cells", len(set(g.tolist())))
    report(r.params, r.bse, label)

# 雙向聚類的前提：機尾號與格子若高度巢狀，格子聚類即已吸收機隊層相關，
# 無須雙向。跨度隨樣本界定而變，故三個口徑並列——結論（兩者不巢狀）在三者皆同。
rows = []
for tag, d in [("bad_weather", pl.read_parquet("data/processed/bad_weather.parquet").with_columns(
                    pl.col("sched_dep_utc").dt.truncate("1mo").alias("ym"))),
               ("本腳本樣本", b)]:
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
