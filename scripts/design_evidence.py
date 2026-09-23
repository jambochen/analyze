import numpy as np
import polars as pl

b = pl.read_parquet(
    "data/processed/bad_weather.parquet",
    columns=["operating_carrier", "marketing_carrier", "origin_iata", "tail_number",
             "sched_dep_utc", "y1_attributed"],
).with_columns(
    pl.col("sched_dep_utc").dt.truncate("1mo").alias("ym"),
    pl.col("y1_attributed").cast(pl.Float64).alias("y"),
)

tot = b["y"].var()
gm = b["y"].mean()

# 變異數分解：各來源的組間變異佔總變異多少。公司佔比小是二元結果的常態
# （伯努利的個體層變異主宰總變異），重點是它與機場、格子的相對大小。
rows = []
for keys, label in [
    (["operating_carrier"], "公司"),
    (["origin_iata"], "起點機場"),
    (["ym"], "年月"),
    (["origin_iata", "ym"], "機場 × 年月格子"),
]:
    g = b.group_by(keys).agg(pl.col("y").mean().alias("m"), pl.len().alias("n"))
    v = np.average((g["m"].to_numpy() - gm) ** 2, weights=g["n"].to_numpy())
    rows.append({"item": label, "value": v / tot, "kind": "變異佔比", "k": g.height})

# 格子內去均值後，剩下的公司層變異才是識別用的那一份。
b = b.with_columns((pl.col("y") - pl.col("y").mean().over("origin_iata", "ym")).alias("dev"))
tot_dev = b["dev"].var()
g = b.group_by("operating_carrier").agg(pl.col("dev").mean().alias("m"), pl.len().alias("n"))
v_within = np.average(g["m"].to_numpy() ** 2, weights=g["n"].to_numpy())
rows.append({"item": "公司（格子內）", "value": v_within / tot_dev, "kind": "變異佔比", "k": g.height})

# 主線層 vs 執飛者層：執飛者層較大即支持清單以執飛公司為單位。
gg = b.group_by("marketing_carrier", "operating_carrier").agg(
    pl.len().alias("n"), pl.col("dev").mean().alias("cell_mean")
)
mkt = gg.group_by("marketing_carrier").agg(
    ((pl.col("cell_mean") * pl.col("n")).sum() / pl.col("n").sum()).alias("mkt_mean")
)
gg = gg.join(mkt, on="marketing_carrier").with_columns(
    (pl.col("cell_mean") - pl.col("mkt_mean")).alias("op_within_mkt")
)
w = gg["n"].to_numpy()
rows.append({"item": "主線層", "value": np.average(gg["mkt_mean"].to_numpy() ** 2, weights=w) / tot_dev,
             "kind": "變異佔比", "k": gg["marketing_carrier"].n_unique()})
rows.append({"item": "執飛者層（主線內）", "value": np.average(gg["op_within_mkt"].to_numpy() ** 2, weights=w) / tot_dev,
             "kind": "變異佔比", "k": gg["operating_carrier"].n_unique()})

# ICC：同一分組內觀測的相關程度，即標準誤須聚類的定量依據。機尾號跨越
# 數十個格子，其相關不被格子聚類吸收，故兩者皆須報。
for keys, label in [(["origin_iata", "ym"], "機場 × 年月格子"), (["tail_number"], "機尾號")]:
    g = b.group_by(keys).agg(pl.col("y").mean().alias("m"), pl.len().alias("n")).filter(pl.col("n") >= 2)
    v = np.average((g["m"].to_numpy() - gm) ** 2, weights=g["n"].to_numpy())
    rows.append({"item": label, "value": v / tot, "kind": "ICC", "k": g.height})

# tau 與 I^2：公司間的真實差異相對於抽樣誤差。I^2 接近 1 表示觀測到的
# 公司間離散幾乎全部是真的，不是雜訊。算法與 shrinkage.py 的矩估計同一套。
d = pl.read_csv("outputs/tables/priority_list.csv")
for tag in ["y1", "y2", "y3"]:
    est = d[tag].to_numpy()
    se = d[f"{tag}_se"].to_numpy()
    mu = float(np.average(est, weights=1.0 / se**2))
    tau2 = max(float(np.mean((est - mu) ** 2 - se**2)), 0.0)
    mse = float(np.mean(se**2))
    rows.append({"item": tag, "value": np.sqrt(tau2), "kind": "tau", "k": len(est)})
    rows.append({"item": tag, "value": tau2 / (tau2 + mse), "kind": "I2", "k": len(est)})
    rows.append({"item": tag, "value": tau2 / mse, "kind": "tau2/se2", "k": len(est)})

out = pl.DataFrame(rows)
print(out)
out.write_csv("outputs/tables/design_evidence.csv")
