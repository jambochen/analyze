import sys

import numpy as np
import polars as pl

sys.path.insert(0, "scripts")
import stratum

# 清單是一次性的排序。若改為逐月監測，每家每月的航班數須足以支撐一個統計量，且訊號
# 不能只來自少數幾個月。此處量這兩件事。
d = stratum.informative(stratum.load().drop_nulls(stratum.WX), "y1_attributed")

# risk adjustment 取層平均：同機場同月的航班共用天氣與航管狀況，觀測減層平均即相對於
# 同層其他公司的偏離。
d = d.with_columns((pl.col("y1_attributed") - pl.col("y1_attributed").mean().over("origin_iata", "ym")).alias("r"))
cm = d.group_by("operating_carrier", "ym").agg(pl.len().alias("n"), pl.col("r").mean().alias("rbar")).sort(
    "operating_carrier", "ym")
n_months = d["ym"].n_unique()

counts = cm.group_by("operating_carrier").agg(
    pl.len().alias("months"),
    pl.col("n").median().alias("n_median"),
    pl.col("n").min().alias("n_min"),
    (pl.col("n") < 30).mean().alias("share_lt30"),
    (pl.col("n") < 100).mean().alias("share_lt100"),
).with_columns((pl.col("months") / n_months).alias("coverage")).sort("n_median")
counts.write_csv("outputs/tables/monitoring_cusum_counts.csv")
print("公司-月總數", cm.height, "少於 30 班", (cm["n"] < 30).mean(), "少於 100 班", (cm["n"] < 100).mean())

# 管制狀態取各公司自己在疫情前的水準：以其他公司為基準的話，長期偏高的公司第一個月
# 就越界，監測退化成重述清單。
t0 = pl.lit(stratum.ERA_CUTS[0]).str.to_datetime().dt.replace_time_zone("UTC")
# 班次少的月份結果全同，當月的標準差為零。逐班的標準差改取該公司基準期全部航班的。
sd0 = dict(d.filter(pl.col("ym") < t0).group_by("operating_carrier").agg(pl.col("r").std()).iter_rows())
K_REF, H = 0.5, 5.0
rows = []
for (c,), x in cm.partition_by("operating_carrier", as_dict=True, maintain_order=True).items():
    base, mon = x.filter(pl.col("ym") < t0), x.filter(pl.col("ym") >= t0)
    if base.height < 12 or mon.height == 0:
        rows.append((c, base.height, mon.height, None, None, None, None, None, None, None))
        continue
    mu = float((base["rbar"] * base["n"]).sum() / base["n"].sum())
    z0 = ((base["rbar"] - mu) / (sd0[c] / base["n"].sqrt())).to_numpy()
    # 同層的航班相關，逐班獨立下的標準誤偏小。以基準期月統計量的實際離散重新定標，
    # 否則管制狀態下就會頻繁越界。
    phi = float(np.std(z0))
    z = ((mon["rbar"] - mu) / (sd0[c] / mon["n"].sqrt())).to_numpy() / phi
    up = dn = 0.0
    path_up, path_dn = [], []
    for v in z:
        up, dn = max(0.0, up + v - K_REF), max(0.0, dn - v - K_REF)
        path_up.append(up)
        path_dn.append(dn)
    stat = np.maximum(path_up, path_dn)
    cross = (stat > H).nonzero()[0]
    rows.append((c, base.height, mon.height, phi, float(np.max(path_up)), float(np.max(path_dn)),
                 str(mon["ym"][int(cross[0])])[:7] if len(cross) else None,
                 int(cross[0]) + 1 if len(cross) else None, float((np.abs(z) > 3).mean()), float(np.abs(z).max())))

path = pl.DataFrame(rows, orient="row", schema={
    "carrier": pl.Utf8, "base_months": pl.Int64, "monitored_months": pl.Int64, "overdispersion": pl.Float64,
    "max_up": pl.Float64, "max_down": pl.Float64, "first_signal": pl.Utf8, "months_to_signal": pl.Int64,
    "share_abs_z_gt3": pl.Float64, "max_abs_z": pl.Float64,
}).join(counts.select(pl.col("operating_carrier").alias("carrier"), "n_median"), on="carrier").sort("n_median")
path.write_csv("outputs/tables/monitoring_cusum.csv")
with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=300):
    print(counts)
    print(path)
