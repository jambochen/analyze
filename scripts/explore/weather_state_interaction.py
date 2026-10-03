import resource
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, "scripts")
import stratum

# negative control 的公司係數與主樣本高度相關，公司主效應大半與天氣無關。把兩批航班合併，
# 公司係數拆成天氣良好時的基礎項與公司 × 天氣狀態的交互作用，後者才是隨天氣變化的部分。
cols = list(dict.fromkeys(stratum.WX + [
    "wx_origin_flight_category", "wx_dest_flight_category",
    "oracle_origin_dep_flight_category_ordinal", "oracle_dest_scharr_flight_category_ordinal",
    "reported_weather_delay_min", "operating_carrier", "origin_iata", "tail_number", "sched_dep_utc",
]))
bad = ["IFR", "LIFR"]
d = pl.scan_parquet("data/processed/audit_clean.parquet").select(cols).with_columns(
    pl.col("sched_dep_utc").dt.truncate("1mo").alias("ym"),
    (pl.col("reported_weather_delay_min") > 0).alias("y1_attributed"),
    (pl.col("wx_origin_flight_category").is_in(bad) | pl.col("wx_dest_flight_category").is_in(bad)).alias("wx_bad"),
    ((pl.col("oracle_origin_dep_flight_category_ordinal") >= 2)
     | (pl.col("oracle_dest_scharr_flight_category_ordinal") >= 2)).alias("or_bad"),
)
# 主樣本不看 oracle_，與 bad_weather.parquet 同一批；negative control 須兩組氣象皆非惡劣。
d = d.filter(pl.col("wx_bad") | (~pl.col("wx_bad") & ~pl.col("or_bad"))).drop_nulls(stratum.WX)
d = d.select(pl.col(stratum.WX).cast(pl.Float32), "operating_carrier", "origin_iata", "ym", "tail_number", "y1_attributed", "wx_bad").collect()
print(d.group_by("wx_bad").agg(pl.len(), pl.col("y1_attributed").mean()))

# 交互作用只由兩種天氣狀態並存的層識別。
st = d.group_by("origin_iata", "ym").agg(pl.col("wx_bad").n_unique().alias("ns"), pl.len().alias("n"))
cu = d.group_by("origin_iata", "ym", "operating_carrier").agg(pl.col("wx_bad").n_unique().alias("ns"), pl.len().alias("n"))
ident = pl.DataFrame({
    "item": ["層數", "兩種狀態並存的層佔比", "其航班佔比", "公司 × 層單位數", "兩種狀態並存的單位佔比", "其航班佔比（單位）"],
    "v": [float(st.height), (st["ns"] == 2).mean(), st.filter(pl.col("ns") == 2)["n"].sum() / d.height,
          float(cu.height), (cu["ns"] == 2).mean(), cu.filter(pl.col("ns") == 2)["n"].sum() / d.height],
})
print(ident)
ident.write_csv("outputs/tables/weather_state_interaction_identification.csv")


def fit(x, interact, controls_by_state=False):
    # statsmodels 的 cluster 共變異在數百萬列下會複製多份 float64 設計矩陣，故此處直接算
    # sandwich。設計矩陣存 Float32，累加一律轉 float64。
    t0 = time.time()
    x = stratum.informative(x, "y1_attributed")
    n = x.height
    units = sorted(u for u in x["operating_carrier"].unique().to_list() if u != stratum.REF)
    s = x["wx_bad"].to_numpy().astype(np.float32)
    # 先配置整個矩陣再逐欄填入：以區塊串接會讓設計矩陣同時存在兩份。
    C, U = len(stratum.WX), len(units)
    names = list(stratum.WX) + [f"carrier_{u}" for u in units]
    if interact:
        names += ["state"] + [f"state:carrier_{u}" for u in units]
        if controls_by_state:
            names += [f"state:{c}" for c in stratum.WX]
    M = np.empty((n, len(names)), dtype=np.float32)
    for k, c in enumerate(stratum.WX):
        v = x[c].cast(pl.Float64).to_numpy()
        M[:, k] = (v - v.mean()) / (v.std() + 1e-9)
    for k, u in enumerate(units):
        M[:, C + k] = (x["operating_carrier"] == u).to_numpy()
    if interact:
        M[:, C + U] = s
        M[:, C + U + 1:C + 2 * U + 1] = M[:, C:C + U] * s[:, None]
        if controls_by_state:
            M[:, C + 2 * U + 1:] = M[:, :C] * s[:, None]
    g = x.select(pl.struct("origin_iata", "ym").rank("dense"))[:, 0].to_numpy() - 1
    tail = x.select(pl.col("tail_number").rank("dense"))[:, 0].to_numpy() - 1
    y = x["y1_attributed"].to_numpy().astype(np.float64)
    ng = np.bincount(g)
    for k in range(M.shape[1]):
        M[:, k] -= (np.bincount(g, weights=M[:, k]) / ng)[g].astype(np.float32)
    y -= (np.bincount(g, weights=y) / ng)[g]
    nz = M.std(0) > 1e-6
    M, names = M[:, nz], [a for a, k in zip(names, nz) if k]
    K = M.shape[1]
    XtX, Xty = np.zeros((K, K)), np.zeros(K)
    for i in range(0, n, 500_000):
        c = M[i:i + 500_000].astype(np.float64)
        XtX += c.T @ c
        Xty += c.T @ y[i:i + 500_000]
    bread = np.linalg.inv(XtX)
    beta = bread @ Xty
    resid = y - np.concatenate([M[i:i + 500_000].astype(np.float64) @ beta for i in range(0, n, 500_000)])
    t_fit = time.time() - t0

    def meat(ids):
        _, inv = np.unique(ids, return_inverse=True)
        G = inv.max() + 1
        S = np.column_stack([np.bincount(inv, weights=M[:, k] * resid, minlength=G) for k in range(K)])
        return G / (G - 1) * (S.T @ S)

    # Cameron–Gelbach–Miller：兩個單向的和減去交集。
    V = bread @ (meat(g) + meat(tail) - meat(g.astype(np.int64) * (tail.max() + 1) + tail)) @ bread
    se = np.sqrt(np.diag(V))
    t_all = time.time() - t0
    print("n", n, "strata", len(ng), "K", K, "fit", round(t_fit, 1), "total", round(t_all, 1))
    return pl.DataFrame({"term": names, "coef": beta, "se": se}), V, names, n, t_fit, t_all


main, *_ = fit(d.filter("wx_bad"), False)
nc, *_ = fit(d.filter(~pl.col("wx_bad")), False)
pooled, V, names, n, t_fit, t_all = fit(d, True)
full, _, _, _, t_fit_full, t_all_full = fit(d, True, controls_by_state=True)

car = pl.col("term").str.starts_with("carrier_")
out = (
    pooled.filter(car).select("term", pl.col("coef").alias("base"), pl.col("se").alias("base_se"))
    .join(pooled.filter(pl.col("term").str.starts_with("state:carrier_")).select(
        pl.col("term").str.replace("state:", ""), pl.col("coef").alias("inter"), pl.col("se").alias("inter_se")), on="term")
    .join(full.filter(car).select("term", pl.col("coef").alias("base_full")), on="term")
    .join(full.filter(pl.col("term").str.starts_with("state:carrier_")).select(
        pl.col("term").str.replace("state:", ""), pl.col("coef").alias("inter_full"), pl.col("se").alias("inter_full_se")), on="term")
    .join(main.filter(car).select("term", pl.col("coef").alias("main"), pl.col("se").alias("main_se")), on="term")
    .join(nc.filter(car).select("term", pl.col("coef").alias("nc"), pl.col("se").alias("nc_se")), on="term")
)
# 惡劣天氣下的公司係數是基礎項與交互作用之和，其標準誤須用兩者的共變異。
ix = {a: i for i, a in enumerate(names)}
tot_se = [float(np.sqrt(V[ix[t], ix[t]] + V[ix["state:" + t], ix["state:" + t]] + 2 * V[ix[t], ix["state:" + t]])) for t in out["term"]]
out = out.with_columns((pl.col("base") + pl.col("inter")).alias("bad_total"), pl.Series("bad_total_se", tot_se))

# 天氣良好時歸因率的基準很低，機率差與比值可能給出不同的排序。
raw = (
    d.group_by("operating_carrier", "wx_bad").agg(pl.col("y1_attributed").mean().alias("p"), pl.len().alias("n"))
    .pivot("wx_bad", index="operating_carrier", values=["p", "n"]).sort("operating_carrier")
    .with_columns(
        (pl.col("p_true") - pl.col("p_false")).alias("raw_diff"),
        (pl.col("p_true") / pl.col("p_false")).alias("raw_ratio"),
        ((pl.col("p_true") / (1 - pl.col("p_true"))) / (pl.col("p_false") / (1 - pl.col("p_false")))).log().alias("raw_log_or"),
    )
)
print(raw)
raw.write_csv("outputs/tables/weather_state_interaction_raw.csv")

# 模型內的對應量：以基準水準在兩種狀態下的原始歸因率為基準，把係數換成各公司的調整後比值。
ref = raw.filter(pl.col("operating_carrier") == stratum.REF)
p0, p1 = ref["p_false"].item(), ref["p_true"].item()
out = out.with_columns(((p1 + pl.col("bad_total")) / (p0 + pl.col("base"))).alias("adj_ratio"), (p0 + pl.col("base")).alias("adj_p_good"))
print(out.sort("inter", descending=True))
out.write_csv("outputs/tables/weather_state_interaction.csv")


def sp(a, b, t=out):
    return t.select(pl.corr(a, b, method="spearman")).item()


summ = pl.DataFrame({
    "item": [
        "合併樣本航班數", "配適秒數（共同斜率）", "含 two-way cluster 秒數（共同斜率）", "含 two-way cluster 秒數（控制變數 × 狀態）",
        "尖峰記憶體 GB", "交互作用標準誤 / 主樣本主效應標準誤（中位數）", "交互作用標準誤 / 基礎項標準誤（中位數）",
        "|交互作用| / 標準誤 > 1.96 家數", "Spearman 交互作用–主樣本主效應", "Spearman 交互作用–基礎項",
        "Spearman 基礎項–negative control 單獨", "基礎項與 negative control 單獨的最大差 / 標準誤",
        "Spearman 惡劣天氣合計–主樣本主效應", "Spearman 交互作用：共同斜率–控制變數 × 狀態",
        "Spearman 基礎項：共同斜率–控制變數 × 狀態", "調整後良好天氣歸因率為負的家數",
        "Spearman 原始差–原始比", "Spearman 原始差–原始 log OR", "Spearman 交互作用–調整後比值", "Spearman 交互作用–原始差",
    ],
    "v": pl.Series([
        n, t_fit, t_all, t_all_full, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
        (out["inter_se"] / out["main_se"]).median(), (out["inter_se"] / out["base_se"]).median(),
        ((out["inter"] / out["inter_se"]).abs() > 1.96).sum(), sp("inter", "main"), sp("inter", "base"),
        sp("base", "nc"), ((out["base"] - out["nc"]).abs() / out["nc_se"]).max(),
        sp("bad_total", "main"), sp("inter", "inter_full"), sp("base", "base_full"), (out["adj_p_good"] <= 0).sum(),
        sp("raw_diff", "raw_ratio", raw), sp("raw_diff", "raw_log_or", raw),
        sp("inter", "adj_ratio", out.filter(pl.col("adj_p_good") > 0)),
        out.with_columns(pl.col("term").str.replace("carrier_", "").alias("operating_carrier"))
        .join(raw, on="operating_carrier").select(pl.corr("inter", "raw_diff", method="spearman")).item(),
    ], dtype=pl.Float64, strict=False),
})
with pl.Config(tbl_rows=30, fmt_str_lengths=60):
    print(summ)
summ.write_csv("outputs/tables/weather_state_interaction_summary.csv")
