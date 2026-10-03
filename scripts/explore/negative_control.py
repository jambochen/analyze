import sys

import numpy as np
import polars as pl

sys.path.insert(0, "scripts")
import stratum

# 稽核樣本以起飛前的 wx_ 判定惡劣天氣。以事件時刻的 oracle_ 再判定一次，未納入的航班
# 分成兩類：起飛後才轉壞的（補集），與兩組氣象皆良好的（negative control）。
# 後者記入天氣沒有機場觀測支持，其公司係數的分布用來估計 empirical null。
cols = list(dict.fromkeys(stratum.WX + [
    "wx_origin_flight_category", "wx_dest_flight_category",
    "oracle_origin_dep_flight_category_ordinal", "oracle_dest_scharr_flight_category_ordinal",
    "reported_weather_delay_min", "operating_carrier", "origin_iata", "dest_iata", "tail_number", "sched_dep_utc",
]))
bad = ["IFR", "LIFR"]
d = pl.read_parquet("data/processed/audit_clean.parquet", columns=cols).with_columns(
    pl.col("sched_dep_utc").dt.truncate("1mo").alias("ym"),
    (pl.col("reported_weather_delay_min") > 0).alias("y1_attributed"),
    (pl.col("wx_origin_flight_category").is_in(bad) | pl.col("wx_dest_flight_category").is_in(bad)).alias("wx_bad"),
    # ordinal 0 VFR 至 3 LIFR，IFR 以上為 2。
    ((pl.col("oracle_origin_dep_flight_category_ordinal") >= 2)
     | (pl.col("oracle_dest_scharr_flight_category_ordinal") >= 2)).alias("or_bad"),
)
print("oracle_ 類別缺失", d["or_bad"].null_count(), "/", d.height)
d = d.filter(pl.col("or_bad").is_not_null())

cells = d.group_by("wx_bad", "or_bad").agg(pl.len().alias("n"), pl.col("y1_attributed").mean().alias("y1")).sort("wx_bad", "or_bad")
print(cells)
cells.write_csv("outputs/tables/negative_control_cells.csv")

samples = {
    "main": d.filter("wx_bad"),
    "complement": d.filter(~pl.col("wx_bad") & pl.col("or_bad")),
    "nc": d.filter(~pl.col("wx_bad") & ~pl.col("or_bad")),
}

# 航路上的天氣不在測站觀測內。層取航線 × 年月時，同層航班飛同一條航路，航路天氣在
# 公司之間相同；代價是含兩家以上公司的層變少。
rows = []
for tag, x in samples.items():
    for level, keys in [("機場 × 年月", ["origin_iata", "ym"]), ("航線 × 年月", ["origin_iata", "dest_iata", "ym"])]:
        c = x.group_by(keys).agg(
            pl.col("operating_carrier").n_unique().alias("nc"), pl.col("y1_attributed").n_unique().alias("ny"), pl.len().alias("n")
        )
        ok = c.filter((pl.col("nc") >= 2) & (pl.col("ny") >= 2))
        rows.append((tag, level, x.height, c.height, ok.height / c.height, ok["n"].sum() / x.height))
strata = pl.DataFrame(rows, schema=["sample", "stratum", "flights", "strata", "informative_strata", "informative_flights"], orient="row")
print(strata)
strata.write_csv("outputs/tables/negative_control_strata.csv")

route = samples["nc"].with_columns((pl.col("origin_iata") + "|" + pl.col("dest_iata")).alias("origin_iata"))
out = None
for tag, x in [*samples.items(), ("nc_route", route)]:
    r = (
        stratum.fit(x, "y1_attributed")
        .filter(pl.col("term").str.starts_with("carrier_"))
        .select("term", pl.col("coef").alias(tag), pl.col("se").alias(f"{tag}_se"))
    )
    out = r if out is None else out.join(r, on="term", how="left")

# 矩估計同 shrinkage.py：觀測變異扣掉抽樣變異。
b0, s0 = out["nc"].to_numpy(), out["nc_se"].to_numpy()
mu0 = float(np.average(b0, weights=1 / s0**2))
tau0 = max(float(np.mean((b0 - mu0) ** 2 - s0**2)), 0.0) ** 0.5
b1, s1 = out["main"].to_numpy(), out["main_se"].to_numpy()
mu1 = float(np.average(b1, weights=1 / s1**2))
tau1 = max(float(np.mean((b1 - mu1) ** 2 - s1**2)), 0.0) ** 0.5
out = out.with_columns(
    (pl.col("main") / pl.col("main_se")).alias("z_theory"),
    ((pl.col("main") - mu0) / (tau0**2 + pl.col("main_se") ** 2).sqrt()).alias("z_empirical"),
)
print(out.sort("main", descending=True))
summ = pl.DataFrame({
    "item": ["mu0", "tau0", "主樣本 tau", "tau0 / tau", "理論參照下 |z| > 1.96 家數", "empirical null 下 |z| > 1.96 家數",
             "Spearman 主樣本–negative control", "Spearman 主樣本–補集", "Spearman 機場層–航線層"],
    "v": pl.Series([
        mu0, tau0, tau1, tau0 / tau1, (out["z_theory"].abs() > 1.96).sum(), (out["z_empirical"].abs() > 1.96).sum(),
        out.select(pl.corr("main", "nc", method="spearman")).item(),
        out.select(pl.corr("main", "complement", method="spearman")).item(),
        out.select(pl.corr("nc", "nc_route", method="spearman")).item(),
    ], dtype=pl.Float64, strict=False),
})
print(summ)
out.write_csv("outputs/tables/negative_control.csv")
summ.write_csv("outputs/tables/negative_control_summary.csv")
