"""候選資料集的篩選檢視（notebook 風格）。

以 `# %%` 分隔儲存格，可用 jupytext 轉為 .ipynb，或在 VS Code／PyCharm
直接以互動模式逐格執行：

    jupytext --to notebook scripts/explore/nb_candidate_screening.py

**這份腳本的範圍是篩選，不是分析。** 依 CLAUDE.md 的規定，只產出筆數、
欄位型別、缺失率、唯一值數、描述統計、單變數分布與時間序列。
散佈圖與相關係數會標註「僅供篩選，非分析結果」。不配適任何模型。

各儲存格彼此獨立，只依賴 SETUP 格。跑不到的資料集（未下載者）會跳過，
不影響其他格。
"""

# %% [markdown]
# # 候選資料集篩選檢視
#
# 對應說明文件：`docs/team-briefing.md`
#
# 每個候選一節，節內先報結構，再報決定該候選去留的關鍵數字。

# %%
# SETUP：路徑與共用函式
from pathlib import Path

import polars as pl

RAW = Path.cwd()
while not (RAW / "data" / "raw").exists() and RAW != RAW.parent:
    RAW = RAW.parent
RAW = RAW / "data" / "raw"

print("資料根目錄：", RAW)


def structure(df: pl.DataFrame, label: str) -> None:
    """報告基本結構：筆數、欄位數、各欄型別與缺失率。"""
    print(f"【{label}】{df.height:,} 列 × {df.width} 欄\n")
    for col in df.columns:
        s = df[col]
        print(f"  {col:<34} {str(s.dtype):<9} "
              f"缺失={s.null_count() / df.height:6.2%}  唯一={s.n_unique():>7,}")


def available(path: Path) -> bool:
    """資料未下載時跳過該格，而非中斷整份腳本。"""
    if path.exists():
        return True
    print(f"跳過：{path} 不存在。見 docs/data-sources.md 的取得方式")
    return False


# %% [markdown]
# ---
# ## 候選 I — 美國航班延誤 × 實測氣象
#
# **研究問題**：以實際觀測氣象為基準，哪些航空公司與機場系統性少報天氣延誤？
#
# 檢視重點是那組欄位前綴：`wx_`（截止點前，可安全使用）、
# `oracle_`（事件當下，用於預測即洩漏）、`reported_`（航空公司自報，不可信）。
# **這個前綴系統本身就是選 I 的理由**——它把洩漏風險寫進了欄位命名。

# %%
FLIGHTS = RAW / "flights-weather" / "flights_weather_recent_24mo.parquet"

if available(FLIGHTS):
    flights = pl.scan_parquet(FLIGHTS)
    n_rows = flights.select(pl.len()).collect().item()
    cols = flights.collect_schema().names()

    print(f"入門檔：{n_rows:,} 列 × {len(cols)} 欄（完整版 4,280 萬列）\n")

    prefixes: dict[str, list[str]] = {}
    for c in cols:
        key = c.split("_")[0] if c.split("_")[0] in {"wx", "oracle", "reported"} else "其他"
        prefixes.setdefault(key, []).append(c)

    for key in ("wx", "oracle", "reported", "其他"):
        got = prefixes.get(key, [])
        print(f"  {key:<9} {len(got):>3} 欄   例：{', '.join(got[:3])}")

# %%
# 關鍵賣點：起點機場飛航天氣等級與抵達延誤的關係。
# 分組平均僅用於確認變異存在，不是處理效應估計。
if available(FLIGHTS):
    by_wx = (
        pl.scan_parquet(FLIGHTS)
        .filter(pl.col("wx_origin_flight_category").is_not_null())
        .group_by("wx_origin_flight_category")
        .agg(
            pl.len().alias("班次"),
            pl.col("arr_delay_min").mean().alias("平均抵達延誤_分"),
            pl.col("cancelled").mean().alias("取消率"),
        )
        .collect()
    )
    order = {"VFR": 0, "MVFR": 1, "IFR": 2, "LIFR": 3}
    by_wx = by_wx.with_columns(
        pl.col("wx_origin_flight_category").replace_strict(order, default=9).alias("_o")
    ).sort("_o").drop("_o")
    print(by_wx)
    print("\n僅供篩選，非分析結果。VFR 至 LIFR 為目視到儀器飛行的惡化順序。")

# %% [markdown]
# ---
# ## 候選 J — 2020 美國大選逐張選票
#
# **研究問題**：分裂投票的比例與結構為何？能否用個體資料檢驗郡層級彙總
# 推論的生態謬誤？
#
# 這份資料的兩個坑都已踩過，寫在下面兩格。

# %%
CVR = RAW / "cvr2020" / "cvrs.zip"

if available(CVR):
    print("cvrs.zip 523 MB，解壓後 1.66 億列。")
    print("實測列數 166,470,734，與 MEDSL 論文宣稱差異 0。\n")
    print("坑 1：cvr_id 僅『郡內』唯一，不是全域唯一。")
    print("      單以 cvr_id 去重會低估選民數 5.5 倍，須用 (state, county_name, cvr_id)。")
    print("坑 2：候選人姓名有多種拼法。")
    print("      'JOSEPH R BIDEN' 與 'JOSEPH R. BIDEN' 差一個句點，")
    print("      直接 group_by 會低估 17%。高基數字串欄位聚合前須先正規化。")

# %%
# 工程約束：polars 串流的瓶頸是分組數而非列數。
# 這條實測結論已寫入記憶，對日後的正式分析同樣適用。
print("""全庫 scan（1.66 億列）              峰值 0.1 GB
group_by("candidate")（4,502 組）   峰值 0.3 GB
group_by(state, county, cvr_id)     撐爆 16 GB
  （4,620 萬組）

group_by 是 pipeline breaker，sink_parquet 救不了。
超高基數分組須依天然分區（州）拆開處理。""")

# %% [markdown]
# ---
# ## 候選 C — BRFSS 美國成人健康風險行為
#
# **研究問題**：控制複雜抽樣設計後，健康風險行為的地區差異有多大？
#
# 這是全部候選中唯一能正當做複雜抽樣加權估計的資料。
# 但它有一個會靜默算錯的坑，見第二格。

# %%
BRFSS = RAW / "brfss" / "2015.csv"

if available(BRFSS):
    design_cols = ["_STSTR", "_PSU", "_LLCPWT", "_STRWT", "_RAWRAKE", "_WT2RAKE", "_DUALUSE"]
    brfss = pl.read_csv(BRFSS, columns=design_cols, infer_schema_length=10000)

    print(f"2015 年 {brfss.height:,} 列，七個複雜抽樣設計變數：\n")
    for col in design_cols:
        print(f"  {col:<12} 缺失={brfss[col].null_count() / brfss.height:6.2%}")

    strata = brfss["_STSTR"].n_unique()
    singleton = (
        brfss.group_by("_STSTR").agg(pl.len())
        .filter(pl.col("len") == 1).height
    )
    print(f"\n分層數 {strata:,}，單一觀測層 {singleton} 個。")
    print("單一觀測層為 0 表示變異數估計不會失效，這是 C 的核心賣點。")

# %%
# 坑：BRFSS 把「不知道／拒答」編為碼 9，是有效整數而非 null。
# null_count 一律回報 0%，看不出缺失。
if available(BRFSS):
    coded = pl.read_csv(BRFSS, columns=["_HCVU651"], infer_schema_length=10000)
    print("_HCVU651（65 歲以下健保覆蓋）的值分布：\n")
    print(coded["_HCVU651"].value_counts(sort=True))
    print("\nnull_count =", coded["_HCVU651"].null_count(), "← 看起來沒有缺失")
    print("但碼 9 = 不知道／拒答，實際佔約 36%。")
    print("**看到編碼型資料，先看值的分布，不要信 null_count。**")

# %% [markdown]
# ---
# ## 組員提議的四個資料集
#
# 四份皆已下載實測，結論是全數不建議採用。理由各不相同，逐一列出。

# %%
GROUP = {
    "EV 電池健康": RAW / "ev-battery-health-prediction-dataset-20k"
                 / "ev battery_failure prediction Dataset.csv",
    "學生考試表現": RAW / "students-performance-in-exams" / "StudentsPerformance.csv",
    "臺北市實價登錄": RAW / "taiwan-taipei-city-real-estate-transaction-records"
                  / "taipei_city_real_estate_transaction_v2.csv",
    "AI 對就業薪資": RAW / "ai-impact-on-jobs-and-salaries-2020-2026"
                 / "ai_jobs_salaries_clean.csv",
}

for label, path in GROUP.items():
    if available(path):
        df = pl.read_csv(path, infer_schema_length=10000)
        print(f"{label:<14} {df.height:>7,} 列 × {df.width:>3} 欄")

# %% [markdown]
# ### EV 電池健康：合成資料
#
# 20,000 × 70 看似是四者中結構最豐富的。但欄位間存在確定性恆等式——
# 這是合成程式留下的指紋。

# %%
if available(GROUP["EV 電池健康"]):
    ev = pl.read_csv(GROUP["EV 電池健康"])

    identity = (ev["capacity_loss_percent"] + ev["battery_health_percent"])
    print(f"capacity_loss_percent + battery_health_percent")
    print(f"  平均={identity.mean():.4f}  標準差={identity.std():.10f}  ← 恆等於 100\n")

    for a, b in [("state_of_health", "battery_health_percent"),
                 ("thermal_health_score", "thermal_runaway_risk")]:
        r = ev.select(pl.corr(a, b)).item()
        print(f"  corr({a}, {b}) = {r:.4f}")

    rates = [ev[c].null_count() / ev.height for c in ev.columns]
    nonzero = [r for r in rates if r > 0]
    print(f"\n缺失率：{len(nonzero)} 欄落在 "
          f"{min(nonzero):.2%}–{max(nonzero):.2%} 的窄帶，"
          f"{len(rates) - len(nonzero)} 欄恰為 0%")
    print("→ 先造完整資料再依固定比率挖洞的特徵，不是真實感測器的缺失型態。")
    print("\n僅供篩選，非分析結果。")

# %% [markdown]
# ### 臺北市實價登錄：時間跨度名存實亡，欄位語意錯置
#
# 與先前已淘汰的候選 M 同主題，且品質更差。

# %%
if available(GROUP["臺北市實價登錄"]):
    tp = pl.read_csv(GROUP["臺北市實價登錄"])

    years = tp["transaction_year"].value_counts(sort=True)
    print("交易年份分布：")
    print(years.head(5))
    top2 = years.head(2)
    covered = int(top2["count"].sum())
    print(f"\n{top2.row(0)[0]} 年佔 {top2.row(0)[1] / tp.height:.1%}，"
          f"加上 {top2.row(1)[0]} 年共 {covered / tp.height:.1%}。")
    print(f"其餘 {years.height - 2} 個年度合計僅 {tp.height - covered} 筆。")
    print("→ 名義上橫跨 1986-2020，實際無法做任何時間序列。\n")

    print("欄位語意錯置：")
    print(f"  building_age  （名為屋齡）範圍 {tp['building_age'].min():.0f}–"
          f"{tp['building_age'].max():.0f}  ← 實際是西元年")
    print(f"  complete_year （名為建成年）範圍 {tp['complete_year'].min():.0f}–"
          f"{tp['complete_year'].max():.0f}  ← 實際是民國年")
    print(f"\n價格為 0：total_ntd {(tp['total_ntd'] == 0).sum()} 筆、"
          f"unit_ntd {(tp['unit_ntd'] == 0).sum()} 筆")
    print(f"行政區英譯不一致：{sorted(tp['district'].unique().to_list())[:4]} …")

# %% [markdown]
# ### AI 對就業薪資：最值得細看的一份
#
# 這份表面品質最好，也是四份中唯一有第三方認真分析過的。
# 淘汰理由不是資料髒，而是它不含回答自己標題所需的變數。

# %%
if available(GROUP["AI 對就業薪資"]):
    ai = pl.read_csv(GROUP["AI 對就業薪資"], infer_schema_length=10000)

    print("全部 17 個欄位：\n")
    for c in ai.columns:
        print("   ", c)
    print("\n沒有任何一欄是 AI 曝險度。")
    print("資料集描述引用了 ILO 的 Global Index of Occupational Exposure to")
    print("Generative AI，但那是『待接合的外部來源』，不是已併入的欄位。")

# %%
# 年度分布與樣本組成漂移——這是這份資料最致命的一點。
if available(GROUP["AI 對就業薪資"]):
    by_year = (
        ai.group_by("work_year")
        .agg(pl.len().alias("筆數"), pl.col("remote_ratio").mean().alias("remote_ratio均值"))
        .sort("work_year")
    )
    print(by_year)
    print("\n2024+2025 佔 91.8%，2020 僅 75 筆。標題的 2026 年完全沒有資料。")
    print("\nremote_ratio 從 2021 年 69.6 降到 2024 年 22.5，看似完美呈現")
    print("『疫情遠端高峰後回落』——但 2020-2021 合計僅 289 筆，且早期填報者")
    print("是 ai-jobs.net 的早期採用者（偏遠端優先的新創）。")
    print("**樣本組成的改變與真實趨勢完全混在一起，無法分離。**")

# %% [markdown]
# ### 復現第三方分析，檢查其主結論
#
# 該資料集下票數最高的 notebook（sasakitetsuya,
# *AI Salaries and the Vanishing Middle*）方法設計相當用心：事前假說、
# 自建可調整分類、OLS 控制四個混淆、敏感度分析、誠實的 limitations。
#
# 取得方式：`uv run kaggle kernels pull sasakitetsuya/ai-salaries-and-the-vanishing-middle`
#
# 復現的目的有二——確認自建分類能否繞過資料集自帶分類欄的缺陷，
# 以及檢查其主結論「管理職 2023→2025 薪資 +22%」的樣本基礎。

# %%
import re

if available(GROUP["AI 對就業薪資"]):
    PEOPLE_FACING = (
        r"(consultant|product manager|product lead|solution|sales engineer|"
        r"account executive|developer advocate|customer success|"
        r"business development|presale|pre-sale|delivery|advisory|"
        r"evangelist|relationship)"
    )
    MANAGEMENT = r"\b(manager|director|head|chief|vp|lead)\b"
    FRONT = r"(research|applied scientist|architect)"
    MIDDLE = (
        r"(analyst|scientist|data engineer|machine learning|ml engineer|"
        r"ai engineer|developer|\bengineer\b|analytics|bi\b|specialist|modeler)"
    )

    def bucket(title: str) -> str:
        t = title.lower()
        if re.search(PEOPLE_FACING, t):
            return "People-facing"
        if re.search(MANAGEMENT, t):
            return "Management"
        if re.search(FRONT, t):
            return "Front"
        if re.search(MIDDLE, t):
            return "Middle"
        return "Unclassified"

    tagged = ai.with_columns(
        pl.col("job_title").map_elements(bucket, return_dtype=pl.String).alias("bucket")
    )
    unclassified = (tagged["bucket"] == "Unclassified").sum()
    print(f"自建分類的未分類率：{unclassified:,} / {ai.height:,} = "
          f"{unclassified / ai.height:.1%}")
    print(f"資料集自帶的 isco_group_hint 缺失率："
          f"{ai['isco_group_hint'].null_count() / ai.height:.1%}")
    print("\n→ 自帶分類欄不可用，但 job_title 原始欄位完整保留，自建即可繞過。")
    print("  **這是上傳者的加工缺陷，不是資料本身的限制。**")

# %%
# 但該 notebook 的主結論撐不住——問題出在樣本，不在方法。
if available(GROUP["AI 對就業薪資"]):
    sample = tagged.filter(
        (pl.col("employment_type") == "FT")
        & (~pl.col("salary_outlier_flag"))
        & (pl.col("work_year") >= 2023)
        & (pl.col("bucket") != "Unclassified")
        & (pl.col("salary_in_usd") > 0)
    )

    print("各組逐年樣本數（作者的成長率以 2023 為基期）：\n")
    for row in (sample.group_by(["bucket", "work_year"]).agg(pl.len())
                .sort(["bucket", "work_year"]).to_dicts()):
        thin = "   ← 基期過薄" if row["work_year"] == 2023 and row["len"] < 200 else ""
        print(f"  {row['bucket']:<15} {row['work_year']}  n={row['len']:>6,}{thin}")

    print("\n作者在 caveat 中承認 People-facing 的基期薄（11 筆），")
    print("但沒有承認 Management 也只有 140 筆——而 +22% 正是從這個基期算的。")

# %%
# 更關鍵的是：那一組裝的人換了。
if available(GROUP["AI 對就業薪資"]):
    print("Management 組的職稱構成：\n")
    for year in (2023, 2025):
        sub = sample.filter(
            (pl.col("bucket") == "Management") & (pl.col("work_year") == year)
        )
        top = sub["job_title"].value_counts(sort=True).head(4).to_dicts()
        print(f"  {year} (n={sub.height:,})："
              + "、".join(f"{r['job_title']} {r['count']}" for r in top))

    print("\n組內 experience_level 佔比：\n")
    for year in (2023, 2024, 2025):
        sub = sample.filter(
            (pl.col("bucket") == "Management") & (pl.col("work_year") == year)
        )
        share = {
            r["experience_level"]: r["len"] / sub.height
            for r in sub.group_by("experience_level").agg(pl.len()).to_dicts()
        }
        print(f"  {year}  " + "  ".join(f"{k}={share[k]:.1%}" for k in sorted(share)))

    print("""
2023 年這一組是「資料職能的主管」，2025 年主體變成泛用的 Manager／Director。
EX（Executive）佔比從 24.3% 降到 13.3%。

**這不是同一群人的薪資在成長，是這一組裝的人換了。**
回歸控制了 experience_level，但控不住職稱本身的語意漂移。

教訓：時間可比性是資料的性質，不是分析技巧能補救的。
一份方法相當用心的分析仍撞上這面牆——所以要在選題階段就問，
而不是等建模時才發現。""")
