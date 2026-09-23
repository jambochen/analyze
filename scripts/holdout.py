import numpy as np
import polars as pl
import statsmodels.api as sm
from scipy import stats

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

b = (
    pl.read_parquet("data/processed/bad_weather.parquet")
    .with_columns(
        pl.col("sched_dep_utc").dt.truncate("1mo").alias("ym"),
        pl.col("sched_dep_utc").dt.year().alias("yr"),
    )
    .drop_nulls(wx)
)


def estimate(d, outcome):
    cell = d.group_by("origin_iata", "ym").agg(
        pl.col("operating_carrier").n_unique().alias("nc"), pl.col(outcome).mean().alias("cr")
    )
    keep = cell.filter((pl.col("nc") >= 2) & (pl.col("cr") > 0) & (pl.col("cr") < 1)).select("origin_iata", "ym")
    d = d.join(keep, on=["origin_iata", "ym"]).filter(pl.col(outcome).is_not_null())
    carriers = sorted(c for c in d["operating_carrier"].unique().to_list() if c != "DL")
    Xn = d.select(wx).to_numpy().astype(float)
    Xn = (Xn - Xn.mean(0)) / (Xn.std(0) + 1e-9)
    X = np.column_stack([Xn] + [(d["operating_carrier"] == c).to_numpy().astype(float) for c in carriers])
    names = wx + [f"carrier_{c}" for c in carriers]
    g = d.select(pl.struct("origin_iata", "ym").rank("dense").alias("i"))["i"].to_numpy() - 1
    y = d[outcome].to_numpy().astype(float)
    dm = pl.DataFrame(X, schema=names).with_columns(pl.Series("g", g), pl.Series("y", y))
    dd = dm.select([(pl.col(c) - pl.col(c).mean().over("g")).alias(c) for c in names + ["y"]])
    M = dd.select(names).to_numpy()
    nz = M.std(0) > 1e-10
    M, nm = M[:, nz], [n for n, k in zip(names, nz) if k]
    r = sm.OLS(dd["y"].to_numpy(), M).fit(cov_type="cluster", cov_kwds={"groups": g})
    assert not np.isnan(r.bse).any()
    return pl.DataFrame({"term": nm, "coef": r.params, "se": r.bse}).filter(
        pl.col("term").str.starts_with("carrier_")
    )


# 監理清單若年度間翻盤即不可用。前段估計、後段驗證。
train, test = b.filter(pl.col("yr") <= 2021), b.filter(pl.col("yr") >= 2022)
print("train", train.height, "test", test.height)

for outcome in ["y1_attributed", "y2_share", "y3_wx_vs_nas"]:
    a = estimate(train, outcome).rename({"coef": "coef_train", "se": "se_train"})
    c = estimate(test, outcome).rename({"coef": "coef_test", "se": "se_test"})
    j = a.join(c, on="term", how="inner").with_columns(
        (pl.col("coef_train") - pl.col("coef_test")).alias("diff"),
        ((pl.col("coef_train") - pl.col("coef_test")) / (pl.col("se_train") ** 2 + pl.col("se_test") ** 2).sqrt()).alias("z"),
    ).sort("coef_train", descending=True)
    rho = stats.spearmanr(j["coef_train"], j["coef_test"])
    # 排序一致性才是清單能否採信的判準，係數本身相等與否是次要的。
    print(f"\n=== {outcome} 兩段共同公司 {j.height}")
    print(f"排序 spearman {rho.statistic:.4f} (p={rho.pvalue:.4f})")
    print(f"係數差的 |z| 中位 {j['z'].abs().median():.2f}  |z|>1.96 者 {j.filter(pl.col('z').abs() > 1.96).height}")
    print(j.select("term", "coef_train", "coef_test", "diff", "z"))
    j.write_csv(f"outputs/tables/holdout_{outcome.split('_')[0]}.csv")

# 清單的實際用途是「該優先查誰」，故檢查前段的高位者在後段是否仍在高位。
y1a = estimate(train, "y1_attributed").sort("coef", descending=True)
y1b = estimate(test, "y1_attributed").sort("coef", descending=True)
for k in [3, 5]:
    ta, tb = set(y1a.head(k)["term"]), set(y1b.head(k)["term"])
    print(f"\nY1 前 {k} 名重疊 {len(ta & tb)}/{k}: 前段 {sorted(x.replace('carrier_', '') for x in ta)}"
          f" 後段 {sorted(x.replace('carrier_', '') for x in tb)}")
