import resource
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, "scripts")
import stratum

# identity link 下的公司 × 天氣狀態交互作用有一部分來自尺度：天氣良好時歸因率的基準低，
# 各公司按同一倍數放大也會在機率差上呈現交互作用。此處以 log link 重估同一模型，
# 交互作用即為比值之比。
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
d = d.filter(pl.col("wx_bad") | (~pl.col("wx_bad") & ~pl.col("or_bad"))).drop_nulls(stratum.WX)
d = d.select(pl.col(stratum.WX).cast(pl.Float32), "operating_carrier", "origin_iata", "ym", "tail_number", "y1_attributed", "wx_bad").collect()

t0 = time.time()
# 結果全為零的層在 Poisson 概似下固定效應趨向負無窮，informative 已排除。
x = stratum.informative(d, "y1_attributed")
del d
n = x.height
units = sorted(u for u in x["operating_carrier"].unique().to_list() if u != stratum.REF)
C, U = len(stratum.WX), len(units)
names = list(stratum.WX) + [f"carrier_{u}" for u in units] + ["state"] + [f"state:carrier_{u}" for u in units]
K = len(names)
s = x["wx_bad"].to_numpy().astype(np.float32)
M = np.empty((n, K), dtype=np.float32)
for k, c in enumerate(stratum.WX):
    v = x[c].cast(pl.Float64).to_numpy()
    M[:, k] = (v - v.mean()) / (v.std() + 1e-9)
for k, u in enumerate(units):
    M[:, C + k] = (x["operating_carrier"] == u).to_numpy()
M[:, C + U] = s
M[:, C + U + 1:] = M[:, C:C + U] * s[:, None]
g = x.select(pl.struct("origin_iata", "ym").rank("dense"))[:, 0].to_numpy() - 1
tail = x.select(pl.col("tail_number").rank("dense"))[:, 0].to_numpy() - 1
y = x["y1_attributed"].to_numpy().astype(np.float64)
cells = x.group_by("operating_carrier", "wx_bad").agg(pl.col("y1_attributed").sum().alias("events"), pl.len().alias("n"))
print(cells.sort("events").head(5))

# 以 IRLS 配適 Poisson：每一步是加權的層內轉換加 WLS，層的固定效應不必展開。
Mt = np.empty_like(M)
mu = (y + y.mean()) / 2
eta = np.log(mu)
beta = np.zeros(K)
for it in range(100):
    z = eta + (y - mu) / mu
    sw = np.bincount(g, weights=mu)
    zt = z - (np.bincount(g, weights=mu * z) / sw)[g]
    for k in range(K):
        Mt[:, k] = M[:, k] - (np.bincount(g, weights=mu * M[:, k]) / sw)[g]
    XtX, Xtz = np.zeros((K, K)), np.zeros(K)
    for i in range(0, n, 500_000):
        c = Mt[i:i + 500_000].astype(np.float64)
        cw = c * mu[i:i + 500_000, None]
        XtX += cw.T @ c
        Xtz += cw.T @ zt[i:i + 500_000]
    new = np.linalg.solve(XtX, Xtz)
    e = zt - np.concatenate([Mt[i:i + 500_000].astype(np.float64) @ new for i in range(0, n, 500_000)])
    eta = np.clip(z - e, -30, 5)
    mu = np.exp(eta)
    step = np.abs(new - beta).max()
    beta = new
    print(it, step, round(time.time() - t0, 1))
    if step < 1e-7:
        break
t_fit = time.time() - t0

bread = np.linalg.inv(XtX)
resid = y - mu


def meat(ids):
    _, inv = np.unique(ids, return_inverse=True)
    G = inv.max() + 1
    S = np.column_stack([np.bincount(inv, weights=Mt[:, k] * resid, minlength=G) for k in range(K)])
    return G / (G - 1) * (S.T @ S)


V = bread @ (meat(g) + meat(tail) - meat(g.astype(np.int64) * (tail.max() + 1) + tail)) @ bread
se = np.sqrt(np.diag(V))
t_all = time.time() - t0

ix = {a: i for i, a in enumerate(names)}
out = pl.DataFrame({
    "term": [f"carrier_{u}" for u in units],
    "log_base": [beta[ix[f"carrier_{u}"]] for u in units],
    "log_base_se": [se[ix[f"carrier_{u}"]] for u in units],
    "log_inter": [beta[ix[f"state:carrier_{u}"]] for u in units],
    "log_inter_se": [se[ix[f"state:carrier_{u}"]] for u in units],
}).join(pl.read_csv("outputs/tables/weather_state_interaction.csv").select("term", "base", "inter", "inter_se", "main"), on="term")
# 聯合檢定全部交互作用為零：比值尺度下各公司是否按同一倍數放大。
ii = [ix[f"state:carrier_{u}"] for u in units]
wald = float(beta[ii] @ np.linalg.solve(V[np.ix_(ii, ii)], beta[ii]))
with pl.Config(tbl_rows=30, tbl_cols=20, tbl_width_chars=250):
    print(out.sort("log_inter", descending=True))
out.write_csv("outputs/tables/weather_state_interaction_log.csv")


def sp(a, b):
    return out.select(pl.corr(a, b, method="spearman")).item()


summ = pl.DataFrame({
    "item": [
        "航班數", "IRLS 迭代數", "配適秒數", "含 two-way cluster 秒數", "尖峰記憶體 GB", "state 係數（基準公司的 log 比值）",
        "|log 交互作用| / 標準誤 > 1.96 家數", "|機率差交互作用| / 標準誤 > 1.96 家數", "交互作用聯合 Wald 統計量", "自由度",
        "Spearman log 交互作用–機率差交互作用", "Spearman log 交互作用–log 基礎項", "Spearman 機率差交互作用–機率差基礎項",
        "Spearman log 基礎項–機率差基礎項", "Spearman log 交互作用–主樣本主效應",
        "log 交互作用的標準差", "log 基礎項的標準差", "log 交互作用標準誤中位數",
    ],
    "v": pl.Series([
        n, it + 1, t_fit, t_all, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, beta[ix["state"]],
        ((out["log_inter"] / out["log_inter_se"]).abs() > 1.96).sum(), ((out["inter"] / out["inter_se"]).abs() > 1.96).sum(), wald, len(ii),
        sp("log_inter", "inter"), sp("log_inter", "log_base"), sp("inter", "base"), sp("log_base", "base"), sp("log_inter", "main"),
        out["log_inter"].std(), out["log_base"].std(), out["log_inter_se"].median(),
    ], dtype=pl.Float64, strict=False),
})
with pl.Config(tbl_rows=30, fmt_str_lengths=60):
    print(summ)
summ.write_csv("outputs/tables/weather_state_interaction_log_summary.csv")
