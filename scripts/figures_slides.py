# 投影片專用圖：字級與長寬比為 16:9 簡報設計，與報告用圖分開。
# 配色取 dataviz 參考 palette 並經 validate_palette.js 驗過：
# 四段堆疊用 categorical slot 1-4（adjacent pairlist 通過），
# 強調式圖只用 slot 1（藍）與 slot 2（橘）＋灰，all-pairs 亦通過。
import matplotlib.pyplot as plt
import matplotlib as mpl
import polars as pl

mpl.rcParams["font.sans-serif"] = ["Noto Sans CJK TC"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["font.size"] = 18
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


def finish(fig, ax, name):
    ax.set_facecolor(SURFACE)
    fig.patch.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_color(FAINT)
        s.set_linewidth(1)
    ax.tick_params(colors=INK, length=3, width=1, color=FAINT)
    fig.tight_layout()
    fig.savefig(f"outputs/figures/{name}.png", dpi=200, facecolor=SURFACE)


# 成因分配：part-to-whole，故堆疊。段間留 2px 空隙取代描邊。
# 論點只在天氣那一段的公司間落差，其餘三段灰階退為背景——四色等亮度會把注意力
# 分散到 NAS 與 carrier 上，而它們在此不承擔論點。
cs = pl.read_csv("outputs/tables/cause_share.csv").sort("weather")
fig, ax = plt.subplots(figsize=(11, 5.0))
bottom = [0.0] * cs.height
for col, color, label in [
    ("weather", BLUE, "天氣"),
    ("nas", "#b8b7b2", "NAS"),
    ("carrier", "#cfcec9", "carrier"),
    ("late_aircraft", "#e4e3df", "late-aircraft"),
]:
    v = cs[col].to_list()
    ax.bar(cs["operating_carrier"], v, bottom=bottom, color=color, label=label,
           width=0.72, linewidth=1.6, edgecolor=SURFACE)
    # 段內直接標值：四段已達 categorical 上限，direct label 為必要條件。
    for i, (b, x) in enumerate(zip(bottom, v)):
        if x >= 0.07:
            ax.text(i, b + x / 2, f"{x*100:.0f}", ha="center", va="center",
                    color="white" if col == "weather" else INK, fontsize=13)
    bottom = [b + x for b, x in zip(bottom, v)]
ax.set_ylabel("延誤分鐘數佔比")
ax.set_xlabel("執飛航空公司")
ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False)
ax.set_ylim(0, 1)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
finish(fig, ax, "slides_cause_share")

# 優先清單：顏色編碼「三項中幾項的信賴區間下界為正」——清單的實際判準，
# 不是編碼正負號（長度已表達正負，再上色即雙重編碼）。
pr = pl.read_csv("outputs/tables/priority_list.csv").sort("y1")
fig, ax = plt.subplots(figsize=(11, 5.4))
y = list(range(pr.height))
tone = {3: ORANGE, 2: BLUE}
colors = [tone.get(k, FAINT) for k in pr["n_pos"]]
ax.barh(y, pr["y1"].to_list(), color=colors, height=0.7)
ax.errorbar(
    pr["y1"].to_list(), y,
    xerr=[(pr["y1"] - pr["y1_lo"]).to_list(), (pr["y1_hi"] - pr["y1"]).to_list()],
    fmt="none", ecolor=INK, capsize=0, lw=1.4,
)
ax.set_yticks(y)
ax.set_yticklabels(pr["carrier"].to_list())
ax.axvline(0, color=INK, lw=1)
ax.set_xlabel("Y1 公司效應（控制機場 × 年月與實測氣象後的機率差）")
handles = [mpl.patches.Patch(color=h, label=lab) for h, lab in
           [(ORANGE, "三項下界皆為正"), (BLUE, "兩項"), (FAINT, "零或一項")]]
ax.legend(handles=handles, ncol=3, loc="lower center",
          bbox_to_anchor=(0.5, 1.0), frameon=False)
finish(fig, ax, "slides_priority_y1")

# Y1 對 Y2：scatter 屬 all-pairs，categorical 上限三色，故用強調式——
# 講稿的對比只在 OO 與 MQ，其餘為背景。
fig, ax = plt.subplots(figsize=(11, 5.4))
ax.axhline(0, color=FAINT, lw=1)
ax.axvline(0, color=FAINT, lw=1)
focus = {"OO": ORANGE, "MQ": BLUE}
for c, x, yy in zip(pr["carrier"], pr["y1"], pr["y2"]):
    hit = c in focus
    ax.scatter(x, yy, s=190 if hit else 80, zorder=3 if hit else 2,
               color=focus.get(c, FAINT),
               edgecolor=SURFACE, linewidth=1.5)
    # 只標強調者與必要的參照，不每點都標。
    if hit:
        ax.annotate(c, (x, yy), xytext=(-14, 0), textcoords="offset points",
                    ha="right", va="center", color=INK, fontsize=17,
                    fontweight="bold")
ax.annotate("兩者 Y1 都高，但 Y2 方向相反", xy=(0.03, 0.96), xycoords="axes fraction",
            ha="left", va="top", color=GREY, fontsize=14)
ax.margins(x=0.08)
ax.set_xlabel("Y1：是否歸因天氣")
ax.set_ylabel("Y2：歸因時掛多少比例")
finish(fig, ax, "slides_y1_vs_y2")

# 航班量趨勢：單一序列，1 hue，不需圖例。
# 畫絕對月量而非相對 2019 同月的比值：比值在 2019 恆等於 1，那段水平線是除法
# 的恆等式而非觀測，會被誤讀成「2019 航班量穩定」。絕對量讓每一點都帶資訊，
# 季節性起伏也看得見。切點仍取自 eda.py 的比值，比值抵銷季節性才定得準。
cv = pl.read_csv("outputs/tables/covid_volume.csv", try_parse_dates=True)
fig, ax = plt.subplots(figsize=(11, 4.4))
ax.plot(cv["mo"], cv["n"] / 1e3, color=BLUE, lw=2)
# 兩條切點與 eda.py 一致：崩跌與回復的實測時點，非行政上的疫情起訖日期。
for cut, lab in [("2020-04-01", "崩跌"), ("2021-06-01", "回復")]:
    t = pl.Series([cut]).str.to_datetime()[0]
    ax.axvline(t, ls="-", lw=1, color=GREY)
    ax.annotate(lab, xy=(t, 690), xytext=(6, 0), textcoords="offset points",
                color=GREY, fontsize=13, va="top")
# 疫情前的夏季高點當參考線：回復後的高點始終低於它，三段時段設定的依據。
peak19 = cv.filter(pl.col("mo").dt.year() == 2019)["n"].max() / 1e3
ax.axhline(peak19, lw=1, ls=(0, (4, 4)), color=FAINT)
ax.annotate("2019 夏季高點", xy=(cv["mo"][-1], peak19), xytext=(-4, 9),
            textcoords="offset points", color=GREY, fontsize=13, ha="right")
ax.set_ylabel("每月航班數（千班）")
ax.set_ylim(120, 720)
finish(fig, ax, "slides_covid_volume")

# 五個成因欄位的佔比：magnitude 比較，故水平長條而非堆疊——security 在堆疊裡
# 會細到不可見，而它的極小正是單體模型只取四欄的理由，必須看得見。
# 顏色只編碼「可歸責 vs 對外歸因」這一個二分，不另外替 security 上色。
cm = pl.read_csv("outputs/tables/cause_mix.csv")
label = {"carrier": "carrier\n自身責任", "weather": "weather\n天氣",
         "nas": "NAS\n航管與機場", "security": "security\n安檢",
         "late_aircraft": "late-aircraft\n前段晚到"}
cm = cm.sort("min_share")
fig, ax = plt.subplots(figsize=(11, 4.6))
y = list(range(cm.height))
colors = [ORANGE if c == "carrier" else BLUE for c in cm["cause"]]
ax.barh(y, cm["min_share"].to_list(), color=colors, height=0.66)
for i, v in enumerate(cm["min_share"]):
    ax.text(v + 0.006, i, f"{v*100:.1f}%", va="center", color=INK, fontsize=15)
ax.set_yticks(y)
ax.set_yticklabels([label[c] for c in cm["cause"]], fontsize=14)
ax.set_xlabel("延誤分鐘數佔比")
ax.set_xlim(0, 0.40)
ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4])
ax.set_xticklabels(["0", "10%", "20%", "30%", "40%"])
handles = [mpl.patches.Patch(color=ORANGE, label="可歸責"),
           mpl.patches.Patch(color=BLUE, label="對外歸因")]
ax.legend(handles=handles, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0),
          frameon=False)
finish(fig, ax, "slides_cause_mix")
