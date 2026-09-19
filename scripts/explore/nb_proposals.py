"""選題建議的篩選檢視（notebook 風格）。

對應文件：`docs/proposal-share.md`

以 `# %%` 分隔儲存格，可用 jupytext 轉為 .ipynb，或在編輯器裡逐格執行：

    jupytext --to notebook scripts/explore/nb_proposals.py

**範圍是篩選，不是分析。** 只產出筆數、欄位型別、缺失率、唯一值數、
描述統計與分組計數。分組平均只用於確認資料結構存在可用的變異，
不構成處理效應估計，相關輸出都標註「僅供篩選，非分析結果」。
不配適任何模型。

四個候選中三個已實測（航班延誤、BRFSS、台灣電網），
PJM 電網僅確認欄位清單與取得路徑，見最後一節。
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
# ## 二、美國 PJM 電網
#
# 尚未實測——資料源、欄位清單與取得路徑已確認，但還沒下載跑檢視。
# 要採用的話我會先補做。說明見本檔最後一節。

# %% [markdown]
# ---
# ## 三、美國成人健康風險行為（BRFSS）
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
# ## 四、台灣電網的機組發電行為
#
# 三份資料。機組發電量那份只有 4 欄，但結構撐得起泛函資料分析、
# 平衡面板與組成資料——「窄表但結構可衍生」的案例。
#
# 檢視重點是那些會影響能不能做的內容空白。

# %%
UNIT_NETP = RAW / "taipower-unit-netp" / "001.json"

if available(UNIT_NETP):
    # JSON 有 UTF-8 BOM，polars 的 read_json 不處理，先用 Python 讀
    import json

    # 實際資料在 records.NET_P 底下，records 的其他鍵是中繼資訊
    records = json.loads(UNIT_NETP.read_text(encoding="utf-8-sig"))["records"]
    units = pl.DataFrame(records["NET_P"])

    print(f"機組發電量：{units.height:,} 列 × {units.width} 欄")
    print(f"  欄位：{units.columns}")

    deduped = units.unique()
    print(f"  去重後：{deduped.height:,} 列"
          f"（重複 {units.height - deduped.height:,} 筆）")

    n_ts = deduped["DATETIME"].n_unique()
    n_unit = deduped["UNIT_NAME"].n_unique()
    print(f"  時間戳 {n_ts:,} 個、機組 {n_unit} 個")
    print(f"  時間範圍 {deduped['DATETIME'].min()} 至 {deduped['DATETIME'].max()}")

# %%
# 內容空白：哪些燃料別實際上沒東西可研究。
if available(UNIT_NETP):
    netp = deduped.with_columns(pl.col("NET_P").cast(pl.Float64))
    by_fuel = (
        netp.group_by("FUEL_TYPE")
        .agg(
            pl.len().alias("筆數"),
            (pl.col("NET_P") == 0).mean().alias("零值率"),
            pl.col("NET_P").max().alias("最大MW"),
        )
        .sort("零值率", descending=True)
    )
    print(by_fuel)
    print("\n核能全期為零（核三已除役）、輕油 99.9% 為零。")
    print("抽蓄發電沒有負值——只記錄放電不記錄抽水，無法完整分析儲能。")
    print("僅供篩選，非分析結果。")

# %%
# 唯一鍵的陷阱：151 個機組名，但 153 個 (機組, 燃料別) 組合。
if available(UNIT_NETP):
    multi_fuel = (
        deduped.group_by("UNIT_NAME")
        .agg(pl.col("FUEL_TYPE").n_unique().alias("燃料別數"))
        .filter(pl.col("燃料別數") > 1)
    )
    print("同時擁有多種燃料別設施的機組：")
    print(multi_fuel)
    print("\n唯一鍵須用 (UNIT_NAME, FUEL_TYPE)，只用機組名會混在一起。")

# %%
# 區域潮流：我原本的賣點被實測推翻。
FLOW = RAW / "taipower-region-flow" / "001_20260919.json"

if available(FLOW):
    import json

    flow_records = json.loads(FLOW.read_text(encoding="utf-8-sig"))["records"]
    flow = (
        pl.DataFrame(flow_records["FLOW_P"])
        .unique(subset=["UNIT_NAME", "DATETIME"], keep="first")
        .with_columns(pl.col("P").cast(pl.Float64, strict=False))
    )

    by_link = (
        flow.group_by("UNIT_NAME")
        .agg(
            pl.len().alias("筆數"),
            pl.col("P").mean().round(0).alias("平均MW"),
            (pl.col("P") < 0).mean().alias("負值率"),
            pl.col("P").std().round(0).alias("標準差"),
        )
        .sort("負值率", descending=True)
    )
    print(by_link)
    print("""
我原本以為負值代表罕見的反向潮流，是研究對象。實測推翻：
負值是常態（兩條主幹線逾 80%），欄位名稱描述的是量測方向的慣例
而非實際流向——「中送北」絕大多數時間實際上是北送中。
真正罕見的是正值。

另外「中送東潮流」近乎常數（標準差遠低於其他），四條鏈路實際可用只有三條。
僅供篩選，非分析結果。""")

# %% [markdown]
# ---
# ## 尚未實測：美國 PJM 電網
#
# 資料源、欄位清單與取得路徑都已確認，但還沒下載跑檢視。
# 要採用的話我會先補做。

# %%
print("""美國 PJM 電網負載與氣象
   來源  arimancy/pjm-hourly-demand-weather（Kaggle）
   規模  97,850 小時（2015-07 至 2026-08，11 年），54 MB
         UTC 時間軸上無缺漏小時
   品質  所有插補與初步值皆有旗標：
           demand_source、weather_is_preliminary
           temp_is_missing、is_dst_transition_day
         連夏令時轉換日都標出來

   建模注意
     - 所有需求特徵至少落後 24 小時。違反會做出看似優異實則無效的結果
     - 回測用的是實際觀測氣象而非氣象預報，
       不可與實際日前預測誤差直接比較""")
