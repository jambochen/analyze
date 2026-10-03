import sys

import numpy as np
import polars as pl

sys.path.insert(0, "scripts")
import stratum

PAIRS = {
    "wx_origin_visibility_m": "oracle_origin_dep_visibility_m",
    "wx_origin_ceiling_m": "oracle_origin_dep_ceiling_m",
    "wx_origin_wind_speed_ms": "oracle_origin_dep_wind_speed_ms",
    "wx_origin_precip_1h_mm": "oracle_origin_dep_precip_1h_mm",
    "wx_dest_visibility_m": "oracle_dest_scharr_visibility_m",
    "wx_dest_ceiling_m": "oracle_dest_scharr_ceiling_m",
    "wx_dest_wind_speed_ms": "oracle_dest_scharr_wind_speed_ms",
    "wx_dest_precip_1h_mm": "oracle_dest_scharr_precip_1h_mm",
}
Y = "y1_attributed"
GAP = {"origin": "oracle_origin_dep_obs_minus_event_min", "dest": "oracle_dest_scharr_obs_minus_event_min"}

b = stratum.load().drop_nulls(stratum.WX + list(PAIRS.values()))
d = stratum.informative(b, Y)
g = d.select(pl.struct("origin_iata", "ym").rank("dense"))[:, 0].to_numpy()
tail = d.select(pl.col("tail_number").rank("dense"))[:, 0].to_numpy()
print("n", d.height, "strata", g.max())


def within(cols):
    x = d.select(cols).with_columns(pl.Series("g", g))
    return x.select([(pl.col(c) - pl.col(c).mean().over("g")).cast(pl.Float64) for c in cols]).to_numpy()


# 兩次量測若為古典誤差且互相獨立，cov(wx, oracle) 即真值的變異數，除以 wx 的變異數得
# reliability ratio。差值若與結果或延誤長度相關，誤差即非古典，校正的前提不成立。
side = within([Y, "arr_delay_min"])
rows = []
for w, o in PAIRS.items():
    m = within([w, o] + [GAP["origin" if "origin" in w else "dest"]])
    diff = m[:, 1] - m[:, 0]
    c = np.cov(m[:, 0], m[:, 1])
    raw = d.select(w, o)
    rows.append({
        "wx": w,
        "corr_within": c[0, 1] / np.sqrt(c[0, 0] * c[1, 1]),
        "reliability_wx": c[0, 1] / c[0, 0],
        "reliability_oracle": c[0, 1] / c[1, 1],
        "var_ratio_oracle_wx": c[1, 1] / c[0, 0],
        "share_identical": (raw[w] == raw[o]).mean(),
        # 能見度與雲冪有編碼上限，落在上限的觀測沒有可量測的誤差。上限取眾數而非最大值：
        # 少數測站回報超出常見上限的值。
        "share_wx_at_max": (raw[w] == raw[w].mode()[0]).mean(),
        "corr_diff_y1": np.corrcoef(diff, side[:, 0])[0, 1],
        "corr_diff_arr_delay": np.corrcoef(diff, side[:, 1])[0, 1],
        "corr_diff_obs_gap": np.corrcoef(diff, m[:, 2])[0, 1],
        "corr_diff_wx": np.corrcoef(diff, m[:, 0])[0, 1],
    })
rel = pl.DataFrame(rows)
print(rel)
rel.write_csv("outputs/tables/weather_measurement_error_reliability.csv")

endog = list(PAIRS)
exog = [c for c in stratum.WX if c not in endog]
units = sorted(u for u in d["operating_carrier"].unique().to_list() if u != stratum.REF)
d = d.with_columns([(pl.col("operating_carrier") == u).cast(pl.Float64).alias(f"carrier_{u}") for u in units])
carriers = [f"carrier_{u}" for u in units]

W, O, Z, y = within(endog), within(list(PAIRS.values())), within(exog + carriers), within([Y])[:, 0]
W, O = W / W.std(0), O / O.std(0)
keep = Z.std(0) > 1e-10
Z, znames = Z[:, keep], [n for n, k in zip(exog + carriers, keep) if k]
Z = Z / Z.std(0)
zscale = within(exog + carriers)[:, keep].std(0)
X = np.column_stack([W, Z])
names = endog + znames


def cluster_meat(S, ids):
    A = np.zeros((ids.max() + 1, S.shape[1]))
    np.add.at(A, ids, S)
    return A.T @ A


def estimate(Xh):
    bread = np.linalg.inv(Xh.T @ X)
    beta = bread @ (Xh.T @ y)
    S = Xh * (y - X @ beta)[:, None]
    both = np.unique(np.column_stack([g, tail]), axis=0, return_inverse=True)[1].ravel()
    meat = cluster_meat(S, g) + cluster_meat(S, tail) - cluster_meat(S, both)
    return beta, np.sqrt(np.diag(bread @ meat @ bread.T))


Q = np.column_stack([O, Z])
first = np.linalg.lstsq(Q, W, rcond=None)[0]
What = Q @ first
# 工具的強度以排除其他控制後的 partial R² 衡量；oracle 與 wx 高度相關時不會是弱工具。
Wres = W - Z @ np.linalg.lstsq(Z, W, rcond=None)[0]
Whres = What - Z @ np.linalg.lstsq(Z, What, rcond=None)[0]
partial = (Whres**2).sum(0) / (Wres**2).sum(0)
print("first-stage partial R2", dict(zip(endog, partial.round(3))))

b_ols, se_ols = estimate(X)
b_iv, se_iv = estimate(np.column_stack([What, Z]))

res = pl.DataFrame({"term": names, "coef_ols": b_ols, "se_ols": se_ols, "coef_iv": b_iv, "se_iv": se_iv})
# 公司欄在估計時標準化過，還原為機率差才能與主模型的係數對照。
back = pl.DataFrame({"term": znames, "scale": zscale})
res = res.join(back, on="term", how="left").with_columns(pl.col("scale").fill_null(1.0)).with_columns(
    [(pl.col(c) / pl.col("scale")).alias(c) for c in ["coef_ols", "se_ols", "coef_iv", "se_iv"]]
).drop("scale").with_columns(((pl.col("coef_iv") - pl.col("coef_ols")) / pl.col("se_ols")).alias("shift_in_se"))
res = res.join(pl.DataFrame({"term": endog, "first_stage_partial_r2": partial}), on="term", how="left")
print(res.filter(~pl.col("term").str.starts_with("carrier_")))
car = res.filter(pl.col("term").str.starts_with("carrier_")).sort("coef_ols", descending=True)
print(car)
res.write_csv("outputs/tables/weather_measurement_error.csv")

sig = lambda c, s: (pl.col(c).abs() > 1.96 * pl.col(s)).sum()
summary = car.select(
    pl.corr("coef_ols", "coef_iv", method="spearman").alias("spearman"),
    pl.col("shift_in_se").abs().max().alias("max_shift_in_se"),
    pl.col("shift_in_se").abs().median().alias("median_shift_in_se"),
    (pl.col("se_iv") / pl.col("se_ols")).median().alias("median_se_ratio"),
    sig("coef_ols", "se_ols").alias("n_sig_ols"),
    sig("coef_iv", "se_iv").alias("n_sig_iv"),
    ((pl.col("coef_ols").sign() != pl.col("coef_iv").sign()).sum()).alias("n_sign_flip"),
)
print(summary)
summary.write_csv("outputs/tables/weather_measurement_error_summary.csv")
