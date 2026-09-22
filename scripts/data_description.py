# 報告與簡報的資料描述：樣本漏斗、結果變數分布、惡劣天氣觸發來源。
# 與 eda.py 分開：eda.py 服務四個方法決策，本檔服務報告的資料節。
# 配色與 figures_slides.py 同一組 palette。
import matplotlib.pyplot as plt
import matplotlib as mpl
import polars as pl

mpl.rcParams["font.sans-serif"] = ["Noto Sans CJK TC"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["font.size"] = 15
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["axes.spines.right"] = False

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREY = "#8a8a86"
FAINT = "#d6d5d1"
INK = "#0b0b0b"
SURFACE = "#fcfcfb"


def finish(fig, ax, name):
    for a in (ax if isinstance(ax, (list, tuple)) else [ax]):
        a.set_facecolor(SURFACE)
        for s in a.spines.values():
            s.set_color(FAINT)
            s.set_linewidth(1)
        a.tick_params(colors=INK, length=3, width=1, color=FAINT)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(f"outputs/figures/{name}.png", dpi=200, facecolor=SURFACE)


raw = [f"data/raw/flights-weather/flights_weather_{y}.parquet" for y in range(2019, 2025)]
n_raw = pl.scan_parquet(raw).select(pl.len()).collect(engine="streaming").item()
audit = pl.read_parquet("data/processed/audit_sample.parquet")
bad = pl.read_parquet("data/processed/bad_weather.parquet")

# 2025 檔未納入下載，年度檔本身即 2019–2024，故漏斗無「限年份」一步。
funnel = pl.DataFrame({
    "step": ["2019–2024 年度檔", "限申報門檻 arr_delay_min >= 15", "限惡劣天氣 IFR/LIFR"],
    "reason": ["—", "BTS 申報義務的定義域，非清理選擇", "晴天談天氣歸因無意義"],
    "n": [n_raw, audit.height, bad.height],
})
funnel = funnel.with_columns(
    (pl.col("n") / n_raw).alias("frac_of_raw"),
    (pl.col("n") / pl.col("n").shift(1)).alias("frac_kept"),
)
print(funnel)
funnel.write_csv("outputs/tables/sample_funnel.csv")

fig, ax = plt.subplots(figsize=(11, 4.2))
y = list(range(funnel.height))[::-1]
ax.barh(y, funnel["n"].to_list(), color=[FAINT, BLUE, ORANGE], height=0.66)
for yy, n, lab in zip(y, funnel["n"], funnel["step"]):
    ax.text(n * 1.02, yy, f"{n/1e6:.2f}M", va="center", color=INK, fontsize=13)
ax.set_yticks(y)
ax.set_yticklabels(funnel["step"].to_list(), fontsize=13)
ax.set_xscale("log")
ax.set_xlabel("航班數（對數刻度）")
ax.margins(x=0.18)
finish(fig, ax, "slides_sample_funnel")

# 結構性缺失：三類成因不同，分開列。y2/y3 的 null 是定義域排除，不是缺失。
miss = pl.DataFrame({
    "col": [c for c in bad.columns if bad[c].null_count() > 0],
}).with_columns(
    pl.col("col").map_elements(lambda c: bad[c].null_count() / bad.height, return_dtype=pl.Float64).alias("null_rate")
).sort("null_rate", descending=True)
print(miss.head(20))
miss.write_csv("outputs/tables/missingness.csv")

# 三個結果變數並列。Y1 為二元故畫公司間比例，Y2 與 Y3 畫分布。
# Y3 在 0 與 1 皆有質量點，這決定了它不能用單一 beta。
y1 = (
    bad.group_by("operating_carrier")
    .agg(pl.len().alias("n"), pl.col("y1_attributed").mean().alias("rate"))
    .filter(pl.col("n") > 5000).sort("rate")
)
print(y1)
y1.write_csv("outputs/tables/y1_rate_by_carrier.csv")

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
axes[0].barh(range(y1.height), y1["rate"].to_list(), color=BLUE, height=0.7)
axes[0].set_yticks(range(y1.height))
axes[0].set_yticklabels(y1["operating_carrier"].to_list(), fontsize=11)
axes[0].set_xlabel("Y1：歸因天氣的比例")

y2 = bad["y2_share"].drop_nulls()
axes[1].hist(y2, bins=50, color=BLUE)
axes[1].set_xlabel("Y2：天氣分鐘 / 總延誤")
axes[1].set_ylabel("航班數")

y3 = bad["y3_wx_vs_nas"].drop_nulls()
axes[2].hist(y3, bins=50, color=ORANGE, log=True)
axes[2].set_xlabel("Y3：天氣 / (天氣 + NAS)")
axes[2].set_ylabel("航班數（對數）")
axes[2].annotate("0 與 1 皆為質量點", xy=(0.5, 0.96), xycoords="axes fraction",
                 ha="center", va="top", color=GREY, fontsize=13)
finish(fig, list(axes), "slides_outcome_dist")

q = pl.DataFrame({
    "var": ["y2_share", "y3_wx_vs_nas"],
    "n": [y2.len(), y3.len()],
    "frac_0": [(y2 == 0).mean(), (y3 == 0).mean()],
    "frac_1": [(y2 == 1).mean(), (y3 == 1).mean()],
    "p25": [y2.quantile(0.25), y3.quantile(0.25)],
    "median": [y2.median(), y3.median()],
    "p75": [y2.quantile(0.75), y3.quantile(0.75)],
})
print(q)
q.write_csv("outputs/tables/outcome_quantiles.csv")

# 樣本界定靠起點與目的地兩欄，須交代惡劣天氣由哪一端觸發。
# 大宗是單端惡劣，故「兩端同時惡劣」不是本樣本的典型情境。
cat = ["VFR", "MVFR", "IFR", "LIFR"]
ct = (
    bad.group_by(["wx_origin_flight_category", "wx_dest_flight_category"]).len()
    .pivot("wx_dest_flight_category", index="wx_origin_flight_category", values="len")
    .sort("wx_origin_flight_category")
)
print(ct)
ct.write_csv("outputs/tables/flight_category_crosstab.csv")

mat = [[(bad.filter((pl.col("wx_origin_flight_category") == o) & (pl.col("wx_dest_flight_category") == d)).height) for d in cat] for o in cat]
fig, ax = plt.subplots(figsize=(7.5, 5.2))
im = ax.imshow(mat, cmap="Blues")
for i in range(4):
    for j in range(4):
        v = mat[i][j]
        ax.text(j, i, f"{v/1000:.0f}k" if v else "—", ha="center", va="center",
                color="white" if v > max(max(r) for r in mat) * 0.5 else INK, fontsize=13)
ax.set_xticks(range(4), cat)
ax.set_yticks(range(4), cat)
ax.set_xlabel("目的地天氣分類")
ax.set_ylabel("起點天氣分類")
fig.colorbar(im, ax=ax, fraction=0.045).outline.set_visible(False)
finish(fig, ax, "slides_flight_category")


# 欄位漏斗：與列數漏斗分開。oracle_ 的 54 欄是機械式整組移除，
# 這個數字本身就是洩漏防護的規模證據。
def grp(c):
    for p in ("oracle_", "reported_", "wx_", "prior_leg_"):
        if c.startswith(p):
            return p
    return "base"


raw_cols = pl.scan_parquet(raw[0]).collect_schema().names()
audit_cols = pl.scan_parquet("data/processed/audit_sample.parquet").collect_schema().names()
bad_cols = bad.columns
groups = ["base", "prior_leg_", "wx_", "oracle_", "reported_"]
col_funnel = pl.DataFrame({
    "group": groups,
    "raw": [sum(grp(c) == g for c in raw_cols) for g in groups],
    "audit": [sum(grp(c) == g for c in audit_cols) for g in groups],
    "bad": [sum(grp(c) == g for c in bad_cols) for g in groups],
})
print(col_funnel)
print(col_funnel.sum())
col_funnel.write_csv("outputs/tables/column_funnel.csv")

label = {"base": "班表 / 機場 / 公司 / 壅塞 / 日曆", "prior_leg_": "prior_leg_ 前一航段",
         "wx_": "wx_ 截止前實測氣象", "oracle_": "oracle_ 事件時刻氣象", "reported_": "reported_ 自報成因"}
color = {"base": FAINT, "prior_leg_": FAINT, "wx_": BLUE, "oracle_": GREY, "reported_": ORANGE}
fig, ax = plt.subplots(figsize=(11, 4.6))
left_raw, left_bad = 0, 0
for g in groups:
    r, bd = col_funnel.filter(pl.col("group") == g)["raw"][0], col_funnel.filter(pl.col("group") == g)["bad"][0]
    ax.barh(1, r, left=left_raw, color=color[g], height=0.5, edgecolor=SURFACE, linewidth=1.5)
    ax.barh(0, bd, left=left_bad, color=color[g], height=0.5, edgecolor=SURFACE, linewidth=1.5,
            label=label[g])
    if r >= 8:
        ax.text(left_raw + r / 2, 1, str(r), ha="center", va="center", color="white", fontsize=13)
    if bd >= 8:
        ax.text(left_bad + bd / 2, 0, str(bd), ha="center", va="center", color="white", fontsize=13)
    left_raw, left_bad = left_raw + r, left_bad + bd
ax.annotate("oracle_ 整組移除", xy=(len(bad_cols) + 27, 0.5), ha="center", va="center",
            color=GREY, fontsize=13)
ax.set_yticks([1, 0], [f"原始 {len(raw_cols)} 欄", f"分析用 {len(bad_cols)} 欄"])
ax.set_xlabel("欄位數")
ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize=12)
ax.set_ylim(-0.6, 1.5)
finish(fig, ax, "slides_column_funnel")

# 稽核樣本的規模：投影片用，避免口頭念一串數字。
dims = pl.DataFrame({
    "item": ["航班數", "執飛航空公司", "起點機場", "機尾號", "機場 × 年月格子", "欄位數"],
    "n": [bad.height, bad["operating_carrier"].n_unique(), bad["origin_iata"].n_unique(),
          bad["tail_number"].n_unique(),
          bad.group_by(["origin_iata", pl.col("sched_dep_utc").dt.truncate("1mo")]).len().height,
          len(bad_cols)],
})
print(dims)
dims.write_csv("outputs/tables/sample_dims.csv")

by_year = bad.group_by(pl.col("sched_dep_utc").dt.year().alias("year")).len().sort("year")
print(by_year)
by_year.write_csv("outputs/tables/sample_by_year.csv")


# --- 以下為資料集介紹用圖 ---

# 資料描述一律讀全量六年，與漏斗的口徑一致。oracle 落差為逐班比對，
# 單年已足以判斷，故另取一年以節省時間。
ALL = raw
ONE = "data/raw/flights-weather/flights_weather_2023.parquet"

# 缺失率：reported_ 與代表性 wx_ 欄並列，凸顯兩者缺失成因不同。
miss_cols = [
    "reported_carrier_delay_min", "reported_weather_delay_min",
    "reported_nas_delay_min", "reported_security_delay_min",
    "reported_late_aircraft_delay_min",
    "wx_origin_snow_depth_cm", "wx_origin_wind_gust_ms",
    "wx_origin_ceiling_m", "wx_origin_visibility_m",
    "wx_origin_temp_c", "wx_origin_precip_1h_mm",
]
miss = (
    pl.scan_parquet(ALL)
    .select([pl.col(c).is_null().mean().alias(c) for c in miss_cols])
    .collect(engine="streaming")
)
miss = pl.DataFrame({"col": miss_cols, "null_frac": [miss[c][0] for c in miss_cols]})
print(miss)
miss.write_csv("outputs/tables/missingness.csv")

fig, ax = plt.subplots(figsize=(11, 4.6))
y = list(range(miss.height))[::-1]
colors = [ORANGE if c.startswith("reported_") else BLUE for c in miss["col"]]
ax.barh(y, miss["null_frac"].to_list(), color=colors, height=0.66)
for yy, v in zip(y, miss["null_frac"]):
    ax.text(v + 0.012, yy, f"{v:.0%}", va="center", color=INK, fontsize=12)
ax.set_yticks(y)
ax.set_yticklabels([c.replace("wx_origin_", "wx_") for c in miss["col"]], fontsize=11)
ax.set_xlim(0, 1.08)
ax.set_xlabel("缺失比例")
finish(fig, ax, "slides_missingness")

# arr_delay_min 的邊際分布：強右偏且含負值，決定不可直接用最小平方。
delay = (
    pl.scan_parquet(ALL)
    .select("arr_delay_min")
    .filter(pl.col("arr_delay_min").is_not_null())
    .collect(engine="streaming")["arr_delay_min"]
)
q = delay.quantile
print("arr_delay_min 分位數",
      [(p, q(p)) for p in (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)],
      "負值比例", (delay < 0).mean())

fig, ax = plt.subplots(figsize=(10, 4.2))
clip = delay.filter((delay > -60) & (delay < 240))
ax.hist(clip.to_list(), bins=120, color=BLUE, edgecolor="none")
ax.axvline(0, color=GREY, lw=1.2, ls="--")
ax.axvline(15, color=ORANGE, lw=1.6)
ax.text(17, ax.get_ylim()[1] * 0.85, "申報門檻 15 分", color=ORANGE, fontsize=12)
ax.set_xlabel("arr_delay_min（截於 -60 至 240 分）")
ax.set_ylabel("航班數")
finish(fig, ax, "slides_delay_dist")

# 班次集中度：機場與公司皆高度集中，直接影響固定效應格子的樣本量。
conc = pl.scan_parquet(ALL).select("origin_iata", "operating_carrier").collect(engine="streaming")
ap = conc["origin_iata"].value_counts(sort=True)["count"].to_numpy()
ca = conc["operating_carrier"].value_counts(sort=True)["count"].to_numpy()
print("前20大機場佔比", ap[:20].sum() / ap.sum(), "機場數", len(ap))
print("前5大公司佔比", ca[:5].sum() / ca.sum(), "公司數", len(ca))

fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
for a, v, lab, col in ((axes[0], ap, "機場", BLUE), (axes[1], ca, "執飛公司", ORANGE)):
    cum = v.cumsum() / v.sum()
    a.plot(range(1, len(cum) + 1), cum, color=col, lw=2)
    a.set_xlabel(f"{lab}數（依班次由多至少）")
    a.set_ylim(0, 1.02)
axes[0].set_ylabel("累積班次比例")
finish(fig, axes.tolist(), "slides_concentration")

# oracle_ 與 wx_ 的實際落差：wx_ 作為基準的品質，直接關係識別策略。
gap = (
    pl.scan_parquet(ONE)
    .select(
        (pl.col("oracle_origin_dep_visibility_m") - pl.col("wx_origin_visibility_m")).alias("vis"),
        (pl.col("oracle_origin_dep_wind_speed_ms") - pl.col("wx_origin_wind_speed_ms")).alias("wind"),
    )
    .collect(engine="streaming")
)
for c in ("vis", "wind"):
    v = gap[c].drop_nulls()
    print(c, "中位差", v.median(), "四分位距", v.quantile(0.75) - v.quantile(0.25),
          "完全相同比例", (v == 0).mean())

# 能見度只有 45 個相異值，且上限 16093 公尺（十法定英里）獨佔多數，
# 全樣本的「完全相同」多為兩小時同時觸頂，非天氣真的沒變。
vis_pair = (
    pl.scan_parquet(ONE)
    .select("wx_origin_visibility_m", "oracle_origin_dep_visibility_m")
    .drop_nulls()
    .collect(engine="streaming")
)
uncapped = vis_pair.filter(pl.col("wx_origin_visibility_m") < 16093)
print("能見度相異值數", vis_pair["wx_origin_visibility_m"].n_unique())
print("非上限樣本佔比", uncapped.height / vis_pair.height,
      "其中完全相同比例",
      (uncapped["wx_origin_visibility_m"] == uncapped["oracle_origin_dep_visibility_m"]).mean())

fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
for a, c, lab, lo, hi in ((axes[0], "vis", "能見度差（公尺）", -4000, 4000),
                          (axes[1], "wind", "風速差（公尺每秒）", -8, 8)):
    v = gap[c].drop_nulls()
    a.hist(v.filter((v > lo) & (v < hi)).to_list(), bins=90, color=BLUE, edgecolor="none")
    a.axvline(0, color=ORANGE, lw=1.4)
    a.set_xlabel(lab)
axes[0].set_ylabel("航班數")
finish(fig, axes.tolist(), "slides_oracle_gap")

# 測站對照品質：距離分布與比對方法構成，支撐後續的敏感度分析。
ref = pl.read_parquet("data/raw/flights-weather/airports_reference.parquet")
print(ref["station_match_method"].value_counts(sort=True))
print("距離分位數", [(p, ref["station_distance_km"].quantile(p)) for p in (0.5, 0.9, 0.99)])
print("逾 10 公里比例", ref["flag_distance_gt_10km"].mean())

fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
d = ref["station_distance_km"].drop_nulls()
axes[0].hist(d.filter(d < 20).to_list(), bins=60, color=BLUE, edgecolor="none")
axes[0].axvline(10, color=ORANGE, lw=1.4)
axes[0].set_xlabel("機場至測站距離（公里，截於 20）")
axes[0].set_ylabel("機場數")
mm = ref["station_match_method"].value_counts(sort=True)
yy = list(range(mm.height))[::-1]
axes[1].barh(yy, mm["count"].to_list(), color=BLUE, height=0.62)
axes[1].set_yticks(yy)
axes[1].set_yticklabels(mm["station_match_method"].to_list(), fontsize=11)
axes[1].set_xlabel("機場數")
finish(fig, axes.tolist(), "slides_station_match")
