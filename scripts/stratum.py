import numpy as np
import polars as pl
import statsmodels.api as sm

WX = [
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

# 時段設三段而非二分：航班量從未回到疫情前水準，新常態與疫情前是不同的制度。
# 切點判準見 docs/decisions.md。
ERA_CUTS = ("2020-04-01", "2021-06-01")

# 樣本量大且歸因率居中。
REF = "DL"


def load():
    t = [pl.lit(c).str.to_datetime().dt.replace_time_zone("UTC") for c in ERA_CUTS]
    return pl.read_parquet("data/processed/bad_weather.parquet").with_columns(
        pl.col("sched_dep_utc").dt.truncate("1mo").alias("ym"),
        pl.col("sched_dep_utc").dt.year().alias("yr"),
        pl.when(pl.col("sched_dep_utc") < t[0]).then(pl.lit("pre"))
        .when(pl.col("sched_dep_utc") < t[1]).then(pl.lit("break"))
        .otherwise(pl.lit("new")).alias("era"),
    )


def informative(d, outcome, unit="operating_carrier"):
    # 單一公司或結果全同的層對公司效應沒有資訊。每個結果變數各自篩，否則某一
    # 結果變數的無資訊層會混進另一個的估計。
    d = d.filter(pl.col(outcome).is_not_null())
    cell = d.group_by("origin_iata", "ym").agg(
        pl.col(unit).n_unique().alias("nc"), pl.col(outcome).n_unique().alias("ny")
    )
    keep = cell.filter((pl.col("nc") >= 2) & (pl.col("ny") >= 2)).select("origin_iata", "ym")
    return d.join(keep, on=["origin_iata", "ym"])


def fit(d, outcome, controls=WX, extra=(), unit="operating_carrier", ref=REF, full=False):
    """層（機場 × 年月）固定效應下的公司主效應。

    controls 標準化後進模型；extra 為 polars 表達式，原尺度進模型，供交互作用與
    指示變數使用。回傳全部非退化項的係數表，公司項以 carrier_ 為前綴。
    full 為真時另回傳係數的共變異矩陣與扣除層固定效應後的殘差自由度。
    """
    d = informative(d.drop_nulls(list(controls)), outcome, unit)

    Xn = d.select(controls).to_numpy().astype(float)
    Xn = (Xn - Xn.mean(0)) / (Xn.std(0) + 1e-9)
    units = sorted(u for u in d[unit].unique().to_list() if u != ref)
    cols = [Xn] + [(d[unit] == u).to_numpy().astype(float)[:, None] for u in units]
    names = list(controls) + [f"carrier_{u}" for u in units]
    if extra:
        e = d.select(extra)
        cols.append(e.to_numpy().astype(float))
        names += e.columns
    X = np.column_stack(cols)

    g = d.select(pl.struct("origin_iata", "ym").rank("dense"))[:, 0].to_numpy()
    # 同一機尾號跨上百個層，層的 cluster 吸收不到機隊層的相關，故雙向。
    tail = d.select(pl.col("tail_number").rank("dense"))[:, 0].to_numpy()

    # 固定效應以層內去均值吸收：層數上萬，dummy 展開為 dense 矩陣放不進記憶體。
    dm = pl.DataFrame(X, schema=names).with_columns(pl.Series("g", g), pl.Series("y", d[outcome].cast(pl.Float64)))
    dd = dm.select([(pl.col(c) - pl.col(c).mean().over("g")).alias(c) for c in names + ["y"]])
    M = dd.select(names).to_numpy()
    # 層內恆定的變數去均值後全為零，會使矩陣降秩；此時係數與標準誤照樣輸出為零
    # 或 NaN 而不報錯，須在此剔除並於下方檢查。
    nz = M.std(0) > 1e-10
    M, names = M[:, nz], [n for n, k in zip(names, nz) if k]
    r = sm.OLS(dd["y"].to_numpy(), M).fit(cov_type="cluster", cov_kwds={"groups": np.column_stack([g, tail])})
    assert not np.isnan(r.bse).any() and (r.bse > 0).all(), f"{outcome} 標準誤退化"
    print(outcome, "n", d.height, "strata", g.max(), "R2", round(r.rsquared, 5))
    out = pl.DataFrame({"term": names, "coef": r.params, "se": r.bse}).with_columns(
        (pl.col("coef") - 1.96 * pl.col("se")).alias("lo"),
        (pl.col("coef") + 1.96 * pl.col("se")).alias("hi"),
    )
    return (out, r.cov_params(), r.df_resid - g.max()) if full else out
