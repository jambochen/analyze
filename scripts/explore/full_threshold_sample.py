import resource
import sys
import time

import numpy as np
import pandas as pd
import polars as pl
from scipy.stats import chi2

sys.path.insert(0, "scripts")
import simplex
import stratum

# 稽核對象改為「延誤分鐘是否記入 carrier 欄」之後，惡劣天氣的篩選失去理由：天氣良好時
# carrier 與其餘成因的分配照樣存在。此處量全部達門檻航班能否配適，以及公司效果是否改變。
NONC = ["reported_weather_delay_min", "reported_nas_delay_min",
        "reported_security_delay_min", "reported_late_aircraft_delay_min"]
# simplex.data 以 module 層的 REPORTED 與 CAUSES 決定成因數，改為兩類即得二類模型，
# 聚合、固定效應與交互項的建構全部沿用。security 併入非 carrier 而不剔除。
simplex.REPORTED = ["reported_carrier_delay_min", "noncarrier_min"]
simplex.CAUSES = ["carrier", "noncarrier"]

bad = ["IFR", "LIFR"]
a = pl.read_parquet(
    "data/processed/audit_clean.parquet",
    columns=list(dict.fromkeys(simplex.WX_UNIT + NONC + [
        "reported_carrier_delay_min", "wx_origin_flight_category", "wx_dest_flight_category",
        "oracle_origin_dep_flight_category_ordinal", "oracle_dest_scharr_flight_category_ordinal",
        "operating_carrier", "origin_iata", "sched_dep_utc",
    ])),
).with_columns(
    pl.sum_horizontal(NONC).alias("noncarrier_min"),
    pl.col("sched_dep_utc").dt.truncate("1mo").alias("ym"),
    (pl.col("wx_origin_flight_category").is_in(bad) | pl.col("wx_dest_flight_category").is_in(bad)).alias("wx_bad"),
    ((pl.col("oracle_origin_dep_flight_category_ordinal") >= 2)
     | (pl.col("oracle_dest_scharr_flight_category_ordinal") >= 2)).alias("or_bad"),
)
print("wx_bad 缺失", a["wx_bad"].null_count(), "or_bad 缺失", a["or_bad"].null_count())

# clean.py 以 filter 取惡劣天氣，類別缺失的航班不進 bad_weather；good 取其補集中類別已知者，
# 使 bad 與 good 不重疊且合起來只少了類別缺失的航班。
samples = {
    "full": a,
    "bad": a.filter("wx_bad"),
    "good": a.filter(~pl.col("wx_bad")),
    "good_oracle": a.filter(~pl.col("wx_bad") & ~pl.col("or_bad")),
}
assert samples["bad"].height == pl.scan_parquet("data/processed/bad_weather.parquet").select(pl.len()).collect().item()

rows = []
for tag, x in samples.items():
    c = x.group_by("origin_iata", "ym").agg(pl.col("operating_carrier").n_unique().alias("nc"), pl.len().alias("n"))
    ok = c.filter(pl.col("nc") >= 2)
    rows.append((tag, x.height, x["operating_carrier"].n_unique(), c.height, ok.height / c.height, ok["n"].sum() / x.height,
                 (x["reported_carrier_delay_min"] + x["noncarrier_min"] > 0).mean()))
dims = pl.DataFrame(rows, schema=["sample", "flights", "carriers", "strata", "strata_2plus_carriers",
                                  "flights_in_2plus_strata", "flights_with_reported_minutes"], orient="row")
print(dims)
dims.write_csv("outputs/tables/full_threshold_sample.csv")

fits, car = [], None
for tag, x in samples.items():
    t0 = time.perf_counter()
    d, terms = simplex.data(x)
    t1 = time.perf_counter()
    m = simplex.fit(d, terms)
    t2 = time.perf_counter()
    fits.append((tag, d.loc[d["cause"] == "carrier", "n_flights"].sum(), d["unit"].nunique(), d["cellcause"].nunique() // 2,
                 len(d), len(terms), len(set(terms) - set(m.coef().index)), len(m.na_index), m.convergence,
                 t1 - t0, t2 - t1, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6))
    r = simplex.carrier_terms(m)[["carrier", "Estimate", "Std. Error"]].rename(
        columns={"Estimate": tag, "Std. Error": f"{tag}_se"})
    car = r if car is None else car.merge(r, on="carrier", how="outer")
fits = pd.DataFrame(fits, columns=["sample", "flights_in_model", "units", "strata", "rows", "terms", "collinear_dropped",
                                   "separated_rows", "converged", "seconds_data", "seconds_fit", "peak_rss_gb"])
print(fits.to_string())
fits.to_csv("outputs/tables/full_threshold_fit.csv", index=False)
print(car.sort_values("full", ascending=False).to_string())
car.to_csv("outputs/tables/full_threshold_carrier.csv", index=False)

tags = list(samples)
rk = []
for i, p in enumerate(tags):
    for q in tags[i + 1:]:
        s = car.dropna(subset=[p, q])
        # bad 與 good 不重疊故兩估計獨立；與 full 的對比因樣本巢狀而保守。
        z = (s[p] - s[q]) / np.sqrt(s[f"{p}_se"] ** 2 + s[f"{q}_se"] ** 2)
        top = lambda k, asc: set(s.sort_values(k, ascending=asc)["carrier"].head(3))
        rk.append((p, q, len(s), s[p].corr(s[q], method="spearman"), s[p].corr(s[q]), (s[p] - s[q]).abs().max(),
                   z.abs().max(), (z.abs() > 1.96).sum(), len(top(p, False) & top(q, False)), len(top(p, True) & top(q, True)),
                   " ".join(s.sort_values(p, ascending=False)["carrier"].head(3)),
                   " ".join(s.sort_values(q, ascending=False)["carrier"].head(3)),
                   " ".join(s.sort_values(p)["carrier"].head(3)), " ".join(s.sort_values(q)["carrier"].head(3))))
rk = pd.DataFrame(rk, columns=["a", "b", "carriers", "spearman", "pearson", "max_abs_diff", "max_abs_z", "n_abs_z_gt_1.96",
                               "top3_shared", "bottom3_shared", "top3_a", "top3_b", "bottom3_a", "bottom3_b"])
print(rk.to_string())
rk.to_csv("outputs/tables/full_threshold_rank.csv", index=False)

# 天氣狀態進模型：單元須依狀態拆開，否則單元固定效應吸收不到兩種狀態下各自的申報總量。
# 把狀態併入 origin_iata 的字串即可沿用 simplex.data，層是否依狀態拆開則在回傳後改 cellcause 決定。
t0 = time.perf_counter()
d, terms = simplex.data(a.drop_nulls("wx_bad").with_columns(
    pl.col("origin_iata") + pl.when("wx_bad").then(pl.lit("#1")).otherwise(pl.lit("#0"))))
state = d["origin_iata"].str.endswith("#1").to_numpy().astype(float)
xt = [t for t in terms if t.startswith("x_")]
ix = pd.DataFrame({f"s{t}": d[t].to_numpy() * state for t in xt}, index=d.index)
ix["s_noncarrier"] = state * (d["cause"] == "noncarrier").to_numpy()
d = pd.concat([d, ix], axis=1)
# 同一公司在同一層的兩種狀態共用申報系統與人員，cluster 取未拆狀態的單元。
d["unit0"] = d["unit"].str.replace(r"#[01]", "", regex=True)
st = [f"s{t}" for t in xt]
t_data = time.perf_counter() - t0

both = d.assign(s=state).groupby("unit0")["s"].nunique()
print("兩種狀態並存的單元佔比", (both == 2).mean())

designs = {
    # 層 × 成因固定效應兩種狀態共用：狀態主效應可估，公司間的比較混用兩種狀態。
    "common_fe": (d.assign(cellcause=d["cellcause"].str.replace(r"#[01]", "", regex=True)), terms + ["s_noncarrier"] + st),
    # 層 × 成因固定效應依狀態分開：狀態主效應被吸收，公司只在同層同狀態內比較。
    "state_fe": (d, terms + st),
}
sep = car[["carrier", "bad", "bad_se", "good", "good_se"]]
out, summ = None, []
for tag, (x, tm) in designs.items():
    t0 = time.perf_counter()
    m = simplex.fit(x, tm, vcov={"CRV1": "unit0"})
    sec = time.perf_counter() - t0
    b = m.coef()
    V = pd.DataFrame(m._vcov, index=b.index, columns=b.index)
    dropped = sorted(set(tm) - set(b.index))
    r = pd.DataFrame([{
        "carrier": t.split("_")[1], "base": b[t], "base_se": np.sqrt(V.loc[t, t]),
        "inter": b[f"s{t}"], "inter_se": np.sqrt(V.loc[f"s{t}", f"s{t}"]),
        "bad_total": b[t] + b[f"s{t}"],
        "bad_total_se": np.sqrt(V.loc[t, t] + V.loc[f"s{t}", f"s{t}"] + 2 * V.loc[t, f"s{t}"]),
    } for t in xt if t in b.index and f"s{t}" in b.index])
    ii = [f"s{t}" for t in xt if f"s{t}" in b.index]
    wald = float(b[ii] @ np.linalg.solve(V.loc[ii, ii].to_numpy(), b[ii].to_numpy()))
    j = r.merge(sep, on="carrier").merge(car[["carrier", "full"]], on="carrier")
    summ.append({
        "design": tag, "rows": len(x), "units": x["unit"].nunique(), "terms": len(tm),
        "collinear_dropped": " ".join(dropped), "separated_rows": len(m.na_index), "converged": m.convergence,
        "seconds_data": t_data, "seconds_fit": sec, "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
        "state_main": b.get("s_noncarrier", np.nan),
        "state_main_se": np.sqrt(V.loc["s_noncarrier", "s_noncarrier"]) if "s_noncarrier" in b.index else np.nan,
        "sd_base": r["base"].std(), "sd_inter": r["inter"].std(), "sd_inter_over_sd_base": r["inter"].std() / r["base"].std(),
        "median_abs_inter": r["inter"].abs().median(), "max_abs_inter": r["inter"].abs().max(),
        "median_abs_base": r["base"].abs().median(), "median_inter_se": r["inter_se"].median(),
        # 交互作用的離散扣掉抽樣變異後剩多少，判準同 shrinkage.py 的矩估計。
        "tau_inter": max(float(np.mean((r["inter"] - r["inter"].mean()) ** 2 - r["inter_se"] ** 2)), 0.0) ** 0.5,
        "n_abs_z_inter_gt_1.96": int(((r["inter"] / r["inter_se"]).abs() > 1.96).sum()),
        "wald": wald, "wald_df": len(ii), "wald_p": chi2.sf(wald, len(ii)),
        "spearman_base_badtotal": r["base"].corr(r["bad_total"], method="spearman"),
        "spearman_inter_base": r["inter"].corr(r["base"], method="spearman"),
        "spearman_base_full": j["base"].corr(j["full"], method="spearman"),
        "spearman_badtotal_full": j["bad_total"].corr(j["full"], method="spearman"),
        "spearman_inter_sepdiff": j["inter"].corr(j["bad"] - j["good"], method="spearman"),
        "spearman_base_sepgood": j["base"].corr(j["good"], method="spearman"),
        "spearman_badtotal_sepbad": j["bad_total"].corr(j["bad"], method="spearman"),
    })
    r = r.rename(columns={c: f"{c}_{tag}" for c in r.columns if c != "carrier"})
    out = r if out is None else out.merge(r, on="carrier", how="outer")
out = out.merge(sep.assign(sep_diff=sep["bad"] - sep["good"],
                           sep_diff_se=np.sqrt(sep["bad_se"] ** 2 + sep["good_se"] ** 2)), on="carrier")
print(out.sort_values("inter_state_fe", ascending=False).to_string())
out.to_csv("outputs/tables/full_threshold_interaction.csv", index=False)
summ = pd.DataFrame(summ).set_index("design").T
print(summ.to_string())
summ.to_csv("outputs/tables/full_threshold_interaction_summary.csv")
