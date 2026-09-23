import numpy as np
import polars as pl
from scipy import stats

# 稽核樣本只含已執飛航班：取消班的 arr_delay_min 為 null，被 ingest.py 的申報門檻
# 篩掉。但取消本身是一次天氣歸因決策（cancellation_code 同為自報欄位），且取消率
# 按公司差異達數倍。若取消與延誤歸因同向，清單上「歸因偏低」者可能只是改用取消
# 消化天氣。此腳本量化這個競爭風險對排序的影響幅度。

files = [f"data/raw/flights-weather/flights_weather_{y}.parquet" for y in range(2019, 2025)]

# 與 clean.py 的惡劣天氣定義一致，但不限申報門檻——取消班正是要看的對象。
allf = (
    pl.scan_parquet(files)
    .select(
        "operating_carrier", "cancelled", "cancellation_reason", "arr_delay_min",
        "sched_dep_utc", "origin_iata", "wx_origin_flight_category", "wx_dest_flight_category",
    )
    .filter(pl.col("sched_dep_utc").dt.year().is_between(2019, 2024))
    .filter(
        pl.col("wx_origin_flight_category").is_in(["IFR", "LIFR"])
        | pl.col("wx_dest_flight_category").is_in(["IFR", "LIFR"])
    )
    .collect(engine="streaming")
    .with_columns(pl.col("sched_dep_utc").dt.year().alias("yr"))
)

print("惡劣天氣全航班", allf.height)
print(allf.select(pl.col("cancelled").mean(), (pl.col("cancellation_reason") == "Weather").mean()))

# 清單用 2022 年起，這裡對齊，否則相關係數跨到不同制度期。
recent = allf.filter(pl.col("yr") >= 2022)

cxl = (
    recent.group_by("operating_carrier")
    .agg(
        pl.len().alias("n_all"),
        pl.col("cancelled").mean().alias("cxl_rate"),
        # 分母為取消班，讀作「取消時有多少比例掛天氣」。
        (pl.col("cancellation_reason") == "Weather").sum().alias("n_cxl_wx"),
        pl.col("cancelled").sum().alias("n_cxl"),
        # 分母為全航班，這才是與 Y1 可比的量：每飛一班，有多少機率以天氣為由取消。
        ((pl.col("cancellation_reason") == "Weather")).mean().alias("cxl_wx_per_flight"),
    )
    .with_columns((pl.col("n_cxl_wx") / pl.col("n_cxl")).alias("cxl_wx_given_cxl"))
    .rename({"operating_carrier": "carrier"})
)

lst = pl.read_csv("outputs/tables/priority_list.csv")
j = lst.join(cxl, on="carrier").filter(pl.col("n") > 5000)

for a in ["cxl_rate", "cxl_wx_per_flight", "cxl_wx_given_cxl"]:
    for bcol in ["y1", "raw_y1", "y3"]:
        x, y = j[a].to_numpy(), j[bcol].to_numpy()
        r, p = stats.pearsonr(x, y)
        rs, ps = stats.spearmanr(x, y)
        print(f"{a} vs {bcol}: pearson {r:.3f} (p={p:.3f}) spearman {rs:.3f} (p={ps:.3f})")

# 嚴重度的直接問法：若取消班全部算成「已歸因天氣」，Y1 的原始排序會怎麼動。
# 這是上界情境，不是估計值——真實情況介於「取消與延誤歸因無關」與此之間。
b = pl.read_parquet("data/processed/bad_weather.parquet").filter(
    pl.col("sched_dep_utc").dt.year() >= 2022
)
y1 = b.group_by("operating_carrier").agg(
    pl.len().alias("n_audit"), pl.col("y1_attributed").sum().alias("n_wx")
).rename({"operating_carrier": "carrier"})

bound = (
    y1.join(cxl, on="carrier")
    .filter(pl.col("n_audit") > 5000)
    .with_columns(
        (pl.col("n_wx") / pl.col("n_audit")).alias("y1_raw"),
        ((pl.col("n_wx") + pl.col("n_cxl_wx")) / (pl.col("n_audit") + pl.col("n_cxl_wx"))).alias("y1_incl_cxl"),
    )
    .with_columns(
        pl.col("y1_raw").rank(descending=True).alias("rk_raw"),
        pl.col("y1_incl_cxl").rank(descending=True).alias("rk_incl"),
    )
    .with_columns((pl.col("rk_incl") - pl.col("rk_raw")).alias("rk_shift"))
    .sort("y1_raw", descending=True)
)
print(bound.select("carrier", "n_audit", "cxl_rate", "cxl_wx_given_cxl", "y1_raw", "y1_incl_cxl", "rk_raw", "rk_incl", "rk_shift"))
print("排序位移絕對值最大", bound.select(pl.col("rk_shift").abs().max()).item())
print("Spearman 原排序 vs 納入取消", stats.spearmanr(bound["y1_raw"], bound["y1_incl_cxl"]))

bound.write_csv("outputs/tables/cancellation_selection.csv")
