import time

import numpy as np
import polars as pl
from scipy.stats import spearmanr

# 稽核樣本以到達延誤達 15 分鐘為條件，而公司本身影響航班是否達門檻。未達門檻的航班
# 不在 audit_sample.parquet 內，須自原始檔取。
REF = "DL"
bad = ["IFR", "LIFR"]
t0 = time.time()

# 逐年彙總到 層 × 公司 的計數再合併：全體航班數千萬列，逐班載入放不進記憶體，
# 而結果變數為二元、模型只有層固定效應與公司因子時，計數是充分統計量。
parts = []
for y in range(2019, 2025):
    parts.append(
        pl.scan_parquet(f"data/raw/flights-weather/flights_weather_{y}.parquet")
        # 年度檔依當地飛行日切分，年底的航班 UTC 年份可能落在次年；與 ingest.py 同一條件。
        .filter(pl.col("sched_dep_utc").dt.year().is_between(2019, 2024))
        .filter(pl.col("wx_origin_flight_category").is_in(bad) | pl.col("wx_dest_flight_category").is_in(bad))
        .filter(~pl.col("cancelled") & ~pl.col("diverted") & pl.col("arr_delay_min").is_not_null())
        .with_columns(
            pl.col("sched_dep_utc").dt.truncate("1mo").alias("ym"),
            (pl.col("arr_delay_min") >= 15).alias("over"),
        )
        .group_by("origin_iata", "ym", "operating_carrier")
        .agg(
            pl.len().alias("n"),
            pl.col("over").sum().alias("n_over"),
            (pl.col("over") & (pl.col("reported_weather_delay_min") > 0)).sum().alias("n_wx"),
        )
        .collect(engine="streaming")
    )
    print(y, parts[-1].height, parts[-1]["n"].sum(), round(time.time() - t0, 1))
c = pl.concat(parts).with_columns(pl.struct("origin_iata", "ym").rank("dense").alias("g"))
print("flights", c["n"].sum(), "over", c["n_over"].sum(), "wx", c["n_wx"].sum(), "cells", c.height, "strata", c["g"].max())


def fit(num, den):
    """層固定效應下公司主效應的加權 within 估計，權重為格內航班數，與逐班 OLS 等價。"""
    x = c.filter(pl.col(den) > 0)
    w = x[den].to_numpy().astype(float)
    ybar = x[num].to_numpy() / w
    # 分母為零的格剔除後層的編號有空號，空號的權重和為零會產生 0/0。
    g = np.unique(x["g"].to_numpy(), return_inverse=True)[1]
    units = sorted(u for u in x["operating_carrier"].unique().to_list() if u != REF)
    D = np.column_stack([(x["operating_carrier"] == u).to_numpy().astype(float) for u in units])
    sw = np.bincount(g, w)
    Dt = D - np.column_stack([np.bincount(g, w * D[:, j]) for j in range(D.shape[1])])[g] / sw[g][:, None]
    yt = ybar - (np.bincount(g, w * ybar) / sw)[g]
    A = np.linalg.inv((Dt * w[:, None]).T @ Dt)
    b = A @ (Dt * w[:, None]).T @ yt
    # 層內逐班殘差之和等於格內航班數乘以格均值的殘差，故層的 cluster 分數可由計數算出。
    s = Dt * (w * (yt - Dt @ b))[:, None]
    S = np.column_stack([np.bincount(g, s[:, j]) for j in range(s.shape[1])])
    se = np.sqrt(np.diag(A @ S.T @ S @ A))
    return pl.DataFrame({"carrier": units, "coef": b, "se": se})


over = fit("n_over", "n").rename({"coef": "over", "se": "over_se"})
cond = fit("n_wx", "n_over").rename({"coef": "cond", "se": "cond_se"})
uncond = fit("n_wx", "n").rename({"coef": "uncond", "se": "uncond_se"})

raw = c.group_by("operating_carrier").agg(pl.col("n", "n_over", "n_wx").sum()).with_columns(
    (pl.col("n_over") / pl.col("n")).alias("p_over"),
    (pl.col("n_wx") / pl.col("n_over")).alias("p_cond"),
    (pl.col("n_wx") / pl.col("n")).alias("p_uncond"),
).rename({"operating_carrier": "carrier"})
ref = raw.filter(pl.col("carrier") == REF).row(0, named=True)

out = raw.join(over, on="carrier", how="left").join(cond, on="carrier", how="left").join(uncond, on="carrier", how="left")
out = out.with_columns(pl.col("over", "cond", "uncond").fill_null(0)).with_columns(
    pl.col("cond").rank(descending=True).alias("rank_cond"),
    pl.col("uncond").rank(descending=True).alias("rank_uncond"),
)

# Lee 式的 trimming：假設公司對「是否達門檻」的效應單調，達門檻比例較高的一方多出來的
# 航班比例為 q。把這 q 全當成歸因天氣者或全當成未歸因者修剪掉，得到「不論哪家執飛都會
# 達門檻」的航班上歸因率差異的上下界。比例取層調整後的值，以 REF 的原始比例為截距。
p1, p0 = ref["p_over"] + out["over"].to_numpy(), ref["p_over"]
y1, y0 = ref["p_cond"] + out["cond"].to_numpy(), ref["p_cond"]
q = np.abs(p1 - p0) / np.maximum(p1, p0)
hi_c = p1 >= p0
lo1, up1 = np.where(hi_c, np.clip((y1 - q) / (1 - q), 0, 1), y1), np.where(hi_c, np.clip(y1 / (1 - q), 0, 1), y1)
lo0, up0 = np.where(hi_c, y0, np.clip((y0 - q) / (1 - q), 0, 1)), np.where(hi_c, y0, np.clip(y0 / (1 - q), 0, 1))
out = out.with_columns(
    pl.Series("trim_q", q), pl.Series("lee_lo", lo1 - up0), pl.Series("lee_hi", up1 - lo0)
).with_columns(((pl.col("lee_lo") > 0) | (pl.col("lee_hi") < 0)).alias("lee_excludes_zero")).sort("cond", descending=True)
with pl.Config(tbl_rows=30, tbl_cols=30, tbl_width_chars=250):
    print(out)
out.write_csv("outputs/tables/threshold_selection.csv")

x = out.filter(pl.col("carrier") != REF)
summary = pl.DataFrame(
    [
        ("flights_bad_weather_flown", c["n"].sum()),
        ("flights_over_threshold", c["n_over"].sum()),
        ("share_over_threshold", c["n_over"].sum() / c["n"].sum()),
        ("carrier_over_rate_min", raw["p_over"].min()),
        ("carrier_over_rate_max", raw["p_over"].max()),
        ("sd_over_effect", x["over"].std()),
        ("sd_cond_effect", x["cond"].std()),
        ("sd_uncond_effect", x["uncond"].std()),
        ("spearman_cond_uncond", spearmanr(out["cond"], out["uncond"])[0]),
        ("spearman_over_cond", spearmanr(out["over"], out["cond"])[0]),
        ("spearman_over_uncond", spearmanr(out["over"], out["uncond"])[0]),
        ("max_rank_shift", (out["rank_cond"] - out["rank_uncond"]).abs().max()),
        ("carriers_sig_over", (x["over"].abs() > 1.96 * x["over_se"]).sum()),
        ("carriers_sig_cond", (x["cond"].abs() > 1.96 * x["cond_se"]).sum()),
        ("carriers_sig_uncond", (x["uncond"].abs() > 1.96 * x["uncond_se"]).sum()),
        ("carriers_lee_excludes_zero", x["lee_excludes_zero"].sum()),
        ("median_trim_q", x["trim_q"].median()),
        ("seconds", time.time() - t0),
    ],
    schema=["quantity", "value"], orient="row", schema_overrides={"value": pl.Float64},
)
with pl.Config(tbl_rows=30):
    print(summary)
summary.write_csv("outputs/tables/threshold_selection_summary.csv")
