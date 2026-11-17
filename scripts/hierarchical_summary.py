# 兩個不重疊測試段合併；固定以 slope 為新候選，intercept 只解釋改善來源。
import os
import sys
import subprocess
import polars as pl
from scipy.stats import norm
import likelihood as lk

subprocess.run([sys.executable, "scripts/hierarchical_audit.py"], check=True,
               env={k: v for k, v in os.environ.items() if k != "HIERARCHICAL_AUDIT_SPEED"})
rows = []
period_rows = []
calibration_parts = []
for speed in ["Bullet", "Blitz", "Classical"]:
    parts = []
    for fold in ["Jul-Sep", "Oct-Dec"]:
        cal = pl.read_csv(f"outputs/tables/hierarchical_{speed}_{fold}_slope_calibration.csv").filter(
            pl.col("group") == "all", pl.col("binning") == "baseline_prediction",
            pl.col("model").is_in(["slope", "selected"]))
        calibration_parts.append(cal.group_by("model", "target").agg(
            (((pl.col("predicted")-pl.col("observed")).abs()*pl.col("n")).sum()/pl.col("n").sum()).alias("weighted_gap")
        ).with_columns(speed=pl.lit(speed), fold=pl.lit(fold)))
        slope = pl.read_parquet(f"data/processed/hierarchical_{speed}_{fold}_slope_prediction.parquet")
        intercept = pl.read_parquet(f"data/processed/hierarchical_{speed}_{fold}_intercept_prediction.parquet")
        assert slope.select("game_id", "a_id").equals(intercept.select("game_id", "a_id"))
        for metric in ["log_loss", "rps", "score_mse"]:
            delta = (slope[f"slope_{metric}"]-intercept[f"intercept_{metric}"]).to_numpy()
            se = lk.dyad_se((delta-delta.mean())/len(slope), slope["a_id"].to_numpy(), slope["b_id"].to_numpy())
            period_rows.append({"speed": speed, "fold": fold, "kind": "slope", "vs": "intercept", "metric": metric,
                                "n": len(slope), "diff": delta.mean(), "se": se,
                                "lo": delta.mean()-1.96*se, "hi": delta.mean()+1.96*se})
        parts.append(slope.with_columns(fold=pl.lit(fold), **{f"intercept_{m}": intercept[f"intercept_{m}"] for m in ["log_loss", "rps", "score_mse"]}))
    d = pl.concat(parts)
    assert d.select(pl.struct("game_id", "a_id").n_unique()).item() == len(d)
    for kind in ["slope", "intercept"]:
        for reference in ["selected", "recent"]+(["intercept"] if kind == "slope" else []):
            for metric in ["log_loss", "rps", "score_mse"]:
                delta = (d[f"{kind}_{metric}"]-d[f"{reference}_{metric}"]).to_numpy()
                se = lk.dyad_se((delta-delta.mean())/len(d), d["a_id"].to_numpy(), d["b_id"].to_numpy())
                rows.append({"speed": speed, "kind": kind, "vs": reference, "metric": metric, "n": len(d),
                             "diff": delta.mean(), "se": se, "lo": delta.mean()-1.96*se, "hi": delta.mean()+1.96*se,
                             "p": 2*norm.sf(abs(delta.mean()/se))})
out = pl.DataFrame(rows)
pl.DataFrame(period_rows).write_csv("outputs/tables/hierarchical_prediction_periods.csv")
out.write_csv("outputs/tables/hierarchical_prediction_pooled.csv")
pl.concat(calibration_parts).write_csv("outputs/tables/hierarchical_calibration_gap.csv")
main = out.filter(pl.col("kind") == "slope", pl.col("vs") == "selected", pl.col("metric") == "log_loss")
main = main.with_columns(p_holm=pl.Series(lk.holm(main["p"].to_numpy())))
decisions = []
for row in main.iter_rows(named=True):
    speed = row["speed"]
    periods = pl.concat([pl.read_csv(f"outputs/tables/hierarchical_{speed}_{fold}_slope_prediction.csv")
                         .filter(pl.col("group") == "all", pl.col("vs") == "selected") for fold in ["Jul-Sep", "Oct-Dec"]])
    both_negative = periods.filter(pl.col("metric") == "log_loss")["diff"].max() < 0
    mse_worse = periods.filter(pl.col("metric") == "score_mse")["lo"].max() > 0
    mse_worse |= out.filter(pl.col("speed") == speed, pl.col("kind") == "slope", pl.col("vs") == "selected", pl.col("metric") == "score_mse")["lo"].item() > 0
    decisions.append({**row, "both_folds_improve": both_negative, "mse_deterioration": mse_worse,
                      "replace": both_negative and row["diff"] < 0 and row["p_holm"] < .05 and not mse_worse})
pl.DataFrame(decisions).write_csv("outputs/tables/hierarchical_prediction_decision.csv")
print(pl.DataFrame(decisions))
