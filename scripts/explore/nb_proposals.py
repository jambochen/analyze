"""選題建議的篩選檢視（notebook 風格）。

對應文件：`docs/proposal-share.md`

以 `# %%` 分隔儲存格，可用 jupytext 轉為 .ipynb，或在編輯器裡逐格執行：

    jupytext --to notebook scripts/explore/nb_proposals.py

**範圍是篩選，不是分析。** 只產出筆數、欄位型別、缺失率、唯一值數、
描述統計與分組計數。分組平均只用於確認資料結構存在可用的變異，
不構成處理效應估計，相關輸出都標註「僅供篩選，非分析結果」。
不配適任何模型。

五個候選中三個已實測（航班延誤、逐張選票、BRFSS），
兩個僅確認欄位清單與取得路徑（台灣交通事故、PJM 電網），見最後一節。
"""

# %% [markdown]
# # 選題建議的篩選檢視
#
# 對應 `docs/proposal-share.md`。順序與文件一致。

# %%
# SETUP
from pathlib import Path

import polars as pl

RAW = Path.cwd()
while not (RAW / "data" / "raw").exists() and RAW != RAW.parent:
    RAW = RAW.parent
RAW = RAW / "data" / "raw"

print("資料根目錄：", RAW)


def available(path: Path) -> bool:
    """資料未下載時跳過該格，不中斷整份腳本。"""
    if path.exists():
        return True
    print(f"跳過：{path} 不存在")
    return False


# %% [markdown]
# ---
# ## 一、美國航班延誤的低報稽核
#
# 推薦這個。檢視兩件事：欄位前綴系統是否如作者所說，
# 以及實際氣象與延誤的關係是否真實存在。

# %%
FLIGHTS = RAW / "flights-weather" / "flights_weather_recent_24mo.parquet"

if available(FLIGHTS):
    flights = pl.scan_parquet(FLIGHTS)
    n_rows = flights.select(pl.len()).collect().item()
    cols = flights.collect_schema().names()

    print(f"入門檔：{n_rows:,} 列 × {len(cols)} 欄（完整版 4,280 萬列）\n")

    groups: dict[str, list[str]] = {}
    for c in cols:
        head = c.split("_")[0]
        key = head if head in {"wx", "oracle", "reported"} else "其他"
        groups.setdefault(key, []).append(c)

    for key in ("wx", "oracle", "reported", "其他"):
        got = groups.get(key, [])
        print(f"  {key:<9} {len(got):>3} 欄   例：{', '.join(got[:3])}")

    print("\noracle_ 前綴是航班自身事件時刻的氣象，用於預測即構成洩漏。")
    print("這組命名把洩漏風險直接寫進欄位名稱。")

# %%
# 起點機場飛航天氣等級與延誤的關係。
# 分組平均僅用於確認變異存在，不是處理效應估計。
if available(FLIGHTS):
    by_weather = (
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
    severity = {"VFR": 0, "MVFR": 1, "IFR": 2, "LIFR": 3}
    by_weather = (
        by_weather
        .with_columns(
            pl.col("wx_origin_flight_category")
            .replace_strict(severity, default=9)
            .alias("_order")
        )
        .sort("_order")
        .drop("_order")
    )
    print(by_weather)
    print("\nVFR→LIFR 是天氣從好到壞。延誤與取消率單調上升。")
    print("僅供篩選，非分析結果。")

# %% [markdown]
# ---
# ## 二、2020 美國大選的逐張選票
#
# 1.66 億列，壓縮檔 523 MB。完整檢視在 `cvr2020.py`（分州串流處理），
# 這裡記錄結論與兩個會靜默算錯的坑。

# %%
CVR = RAW / "cvr2020" / "cvrs.zip"

if available(CVR):
    print("""實測列數 166,470,734，與 MEDSL 論文宣稱差異 0。

分裂投票結構（來自 cvr2020.py 的完整計算）：
  搖擺州可比對選票   8,570,000 張
  明確反向分裂投票     139,000 張（1.62%）

「明確反向」= 總統投 DEM 而下游全投 REP，或反之。
確認分裂投票在個體層級真實可觀測，不是彙總資料的假象。""")

# %%
# 兩個坑。都會算出看似合理的錯數字。
print("""坑 1：cvr_id 只在「郡內」唯一，不是全域唯一。
     單以 cvr_id 去重會低估選民數 5.5 倍。
     正確做法：用 (state, county_name, cvr_id) 三欄。

坑 2：候選人姓名有多種拼法。
     'JOSEPH R BIDEN' 與 'JOSEPH R. BIDEN' 差一個句點，
     直接 group_by 低估 17%。高基數字串欄位聚合前須先正規化。

工程約束：polars 串流的瓶頸是分組數，不是列數。
     全庫 scan（1.66 億列）              峰值 0.1 GB
     group_by("candidate")（4,502 組）   峰值 0.3 GB
     group_by(state, county, cvr_id)     撐爆 16 GB
       （4,620 萬組）
     group_by 是 pipeline breaker，sink_parquet 救不了。
     超高基數分組須依天然分區（州）拆開處理。""")

# %% [markdown]
# ---
# ## 四、美國成人健康風險行為（BRFSS）
#
# 核心賣點是複雜抽樣設計變數完整。檢視兩件事：
# 設計變數的缺失狀況與分層結構，以及編碼型缺失的陷阱。

# %%
BRFSS = RAW / "brfss" / "2015.csv"
DESIGN_COLS = ["_STSTR", "_PSU", "_LLCPWT", "_STRWT", "_RAWRAKE", "_WT2RAKE", "_DUALUSE"]

if available(BRFSS):
    design = pl.read_csv(BRFSS, columns=DESIGN_COLS, infer_schema_length=10000)

    print(f"2015 年 {design.height:,} 列，七個複雜抽樣設計變數：\n")
    for col in DESIGN_COLS:
        print(f"  {col:<12} 缺失={design[col].null_count() / design.height:6.2%}")

    n_strata = design["_STSTR"].n_unique()
    singletons = (
        design.group_by("_STSTR").agg(pl.len())
        .filter(pl.col("len") == 1).height
    )
    print(f"\n分層數 {n_strata:,}，單一觀測層 {singletons} 個。")
    print("只要有任何分層只含一個觀測值，變異數估計就會失效。")

# %%
# 坑：BRFSS 把「不知道／拒答」編為碼 9，是有效整數而非 null。
if available(BRFSS):
    coded = pl.read_csv(BRFSS, columns=["_HCVU651"], infer_schema_length=10000)
    print("_HCVU651（65 歲以下健保覆蓋）的值分布：\n")
    print(coded["_HCVU651"].value_counts(sort=True))

    code9 = (coded["_HCVU651"] == 9).sum()
    print(f"\nnull_count = {coded['_HCVU651'].null_count()}  ← 看起來沒有缺失")
    print(f"但碼 9（不知道／拒答）佔 {code9 / coded.height:.1%}")
    print("直接算平均會得到一個看起來很合理的錯數字。")

# %% [markdown]
# ---
# ## 尚未實測的兩個候選
#
# 資料源、欄位清單與取得路徑都已確認，但還沒下載跑檢視。
# 要採用的話我會先補做。

# %%
print("""三、台灣道路交通事故
   來源  data.gov.tw → 內政部警政署，無需申請
   規模  A1 類（死亡）2,587 列；A2 類（受傷）單月 80,620 列
         推估年約 100 萬列
   欄位  51 欄，已完全解碼（直接是「雨」「市區道路」這類中文值）
   結構  當事者巢狀於事故，是天然的兩層結構；另有經緯度可做空間統計

   讀檔注意
     - CSV 末尾有兩列中繼資訊（資料提供日期、事故類別），須排除
     - 編碼是 UTF-8 with BOM
     - A2 類分散在 9 個 ZIP，要寫腳本串接
     - 一筆事故對應多列（每位當事者一列，有「當事者順位」欄）

五、美國 PJM 電網負載與氣象
   來源  arimancy/pjm-hourly-demand-weather（Kaggle）
   規模  97,850 小時（2015-07 至 2026-08，11 年），54 MB
         UTC 時間軸上無缺漏小時
   品質  所有插補與初步值皆有旗標：
           demand_source、weather_is_preliminary
           temp_is_missing、is_dst_transition_day
         連夏令時轉換日都標出來

   建模注意
     - 所有需求特徵至少落後 24 小時。違反會做出看似優異實則無效的結果
     - 回測用的是實際觀測氣象而非氣象預報，不可與實際日前預測誤差直接比較""")
