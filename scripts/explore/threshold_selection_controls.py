import resource
import time

import numpy as np
import polars as pl

import sys

sys.path.insert(0, "scripts")
import stratum

# threshold_selection.py 只放層的固定效應。此處改為逐班配適並放入氣象共變量，確認
# 「以全體已執飛航班為分母」的估計對象在與主模型相同的控制下是否給出同樣的結論。
# arr_delay_min 不放：它決定航班是否達門檻，是選樣本身。
CTRL = [c for c in stratum.WX if c != "arr_delay_min"]
bad = ["IFR", "LIFR"]
t0 = time.time()
parts = []
for y in range(2019, 2025):
    parts.append(
        pl.scan_parquet(f"data/raw/flights-weather/flights_weather_{y}.parquet")
        # 年度檔依當地飛行日切分，年底的航班 UTC 年份可能落在次年；與 ingest.py 同一條件。
        .filter(pl.col("sched_dep_utc").dt.year().is_between(2019, 2024))
        .filter(pl.col("wx_origin_flight_category").is_in(bad) | pl.col("wx_dest_flight_category").is_in(bad))
        .filter(~pl.col("cancelled") & ~pl.col("diverted") & pl.col("arr_delay_min").is_not_null())
        .with_columns(
            pl.col("wx_origin_snow_depth_cm", "wx_origin_wind_gust_ms", "prior_leg_arr_delay_min").fill_null(0),
            pl.col("sched_dep_utc").dt.truncate("1mo").alias("ym"),
            (pl.col("arr_delay_min") >= 15).alias("over"),
        )
        .with_columns((pl.col("over") & (pl.col("reported_weather_delay_min") > 0)).fill_null(False).alias("wx"))
        .select(pl.col(CTRL).cast(pl.Float32), "arr_delay_min", "operating_carrier", "origin_iata", "ym", "tail_number", "over", "wx")
        .collect()
    )
d = pl.concat(parts)
del parts
print("flights", d.height, "over", d["over"].sum(), "wx", d["wx"].sum(), "null rows", d.height - d.drop_nulls(CTRL).height)
d = d.drop_nulls(CTRL + ["tail_number"])


def fit(x, outcome, ctrl):
    x = stratum.informative(x, outcome)
    n = x.height
    units = sorted(u for u in x["operating_carrier"].unique().to_list() if u != stratum.REF)
    M = np.empty((n, len(ctrl) + len(units)), dtype=np.float32)
    for k, c in enumerate(ctrl):
        v = x[c].cast(pl.Float64).to_numpy()
        M[:, k] = (v - v.mean()) / (v.std() + 1e-9)
    for k, u in enumerate(units):
        M[:, len(ctrl) + k] = (x["operating_carrier"] == u).to_numpy()
    g = x.select(pl.struct("origin_iata", "ym").rank("dense"))[:, 0].to_numpy() - 1
    tail = x.select(pl.col("tail_number").rank("dense"))[:, 0].to_numpy() - 1
    yv = x[outcome].to_numpy().astype(np.float64)
    ng = np.bincount(g)
    for k in range(M.shape[1]):
        M[:, k] -= (np.bincount(g, weights=M[:, k]) / ng)[g].astype(np.float32)
    yv -= (np.bincount(g, weights=yv) / ng)[g]
    K = M.shape[1]
    XtX, Xty = np.zeros((K, K)), np.zeros(K)
    for i in range(0, n, 500_000):
        c = M[i:i + 500_000].astype(np.float64)
        XtX += c.T @ c
        Xty += c.T @ yv[i:i + 500_000]
    bread = np.linalg.inv(XtX)
    beta = bread @ Xty
    resid = yv - np.concatenate([M[i:i + 500_000].astype(np.float64) @ beta for i in range(0, n, 500_000)])

    def meat(ids):
        _, inv = np.unique(ids, return_inverse=True)
        G = inv.max() + 1
        S = np.column_stack([np.bincount(inv, weights=M[:, k] * resid, minlength=G) for k in range(K)])
        return G / (G - 1) * (S.T @ S)

    V = bread @ (meat(g) + meat(tail) - meat(g.astype(np.int64) * (tail.max() + 1) + tail)) @ bread
    print(outcome, "n", n, "K", K, round(time.time() - t0, 1))
    return pl.DataFrame({"carrier": units, outcome: beta[len(ctrl):], outcome + "_se": np.sqrt(np.diag(V))[len(ctrl):]})


over = fit(d, "over", CTRL)
uncond = fit(d, "wx", CTRL).rename({"wx": "uncond", "wx_se": "uncond_se"})
cond = fit(d.filter("over"), "wx", CTRL).rename({"wx": "cond", "wx_se": "cond_se"})
# 主模型的設定：達門檻者，另控制延誤長度。
main = fit(d.filter("over"), "wx", CTRL + ["arr_delay_min"]).rename({"wx": "main", "wx_se": "main_se"})
nofe = pl.read_csv("outputs/tables/threshold_selection.csv").select("carrier", pl.col("uncond").alias("uncond_nocontrol"), pl.col("cond").alias("cond_nocontrol"), pl.col("over").alias("over_nocontrol"))
out = over.join(uncond, on="carrier").join(cond, on="carrier").join(main, on="carrier").join(nofe, on="carrier")
ref = pl.DataFrame({c: [stratum.REF if c == "carrier" else 0.0] for c in out.columns})
out = pl.concat([out, ref]).with_columns(
    pl.col("main").rank(descending=True).alias("rank_main"), pl.col("uncond").rank(descending=True).alias("rank_uncond")
).sort("main", descending=True)
with pl.Config(tbl_rows=30, tbl_cols=30, tbl_width_chars=250):
    print(out)
out.write_csv("outputs/tables/threshold_selection_controls.csv")


def sp(a, b):
    return out.select(pl.corr(a, b, method="spearman")).item()


x = out.filter(pl.col("carrier") != stratum.REF)
summ = pl.DataFrame({
    "item": ["航班數", "Spearman 主模型–無條件", "Spearman 條件–無條件", "Spearman 主模型–條件（差在是否控制延誤長度）",
             "Spearman 門檻效應–主模型", "Spearman 無條件：有無氣象控制", "Spearman 門檻效應：有無氣象控制", "排序最大位移（主模型對無條件）",
             "門檻效應的標準差", "主模型效應的標準差", "無條件效應的標準差", "門檻效應顯著家數", "主模型顯著家數", "無條件顯著家數",
             "秒數", "尖峰記憶體 GB"],
    "v": pl.Series([
        d.height, sp("main", "uncond"), sp("cond", "uncond"), sp("main", "cond"), sp("over", "main"),
        sp("uncond", "uncond_nocontrol"), sp("over", "over_nocontrol"), (out["rank_main"] - out["rank_uncond"]).abs().max(),
        x["over"].std(), x["main"].std(), x["uncond"].std(),
        (x["over"].abs() > 1.96 * x["over_se"]).sum(), (x["main"].abs() > 1.96 * x["main_se"]).sum(), (x["uncond"].abs() > 1.96 * x["uncond_se"]).sum(),
        time.time() - t0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
    ], dtype=pl.Float64, strict=False),
})
with pl.Config(tbl_rows=30, fmt_str_lengths=60):
    print(summ)
summ.write_csv("outputs/tables/threshold_selection_controls_summary.csv")
