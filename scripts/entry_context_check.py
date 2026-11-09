# 對照路徑、互斥分層及覆蓋率核對，避免用較容易的子群冒充模型改善。
import numpy as np
import polars as pl

s = pl.read_csv("outputs/tables/entry_context_scores.csv")
c = pl.read_csv("outputs/tables/entry_context_coverage.csv")
f = pl.read_csv("outputs/tables/entry_context_fits.csv")
t = pl.read_csv("outputs/tables/entry_context_sample.csv")
assert f.height == 16 and f["max_scaled_score"].max() < 1e-3
assert s.height == 4 * 9 * 2 * 7 * 2
assert np.isfinite(s.select(pl.selectors.numeric()).to_numpy()).all()
assert (t["train_max_time"] < t["test_min_time"]).all()
assert ((c["coverage"] >= 0) & (c["coverage"] <= 1)).all()
assert ((c["target_retention"] >= 0) & (c["target_retention"] <= 1)).all()
assert c.filter(pl.col("group").is_in(["low_rd", "low_rd_supported"]), pl.col("target") == "high_rd")["retained_target"].max() == 0
rows = []
for (speed, fold), d in s.partition_by("speed", "fold", as_dict=True).items():
    errors = []
    for group in d["group"].unique():
        for weighting in ["games", "players"]:
            for metric in ["log_loss", "score_mse"]:
                cell = d.filter(pl.col("group") == group, pl.col("weighting") == weighting, pl.col("metric") == metric)
                cal = cell.filter(pl.col("model") == "recalibrated")["diff"].item()
                for model in ["first", "interval", "combined"]:
                    direct = cell.filter(pl.col("model") == model, pl.col("reference") == "recent")["diff"].item()
                    extra = cell.filter(pl.col("model") == model, pl.col("reference") == "recalibrated")["diff"].item()
                    errors.append(abs(direct - extra - cal))
    game = d.filter(pl.col("weighting") == "games")
    for pair in [["any_first", "neither_first"], ["known_interval", "missing_interval"]]:
        summed = game.filter(pl.col("group").is_in(pair)).group_by("model", "reference", "metric").agg(n=pl.col("n").sum(), diff=(pl.col("diff") * pl.col("n")).sum() / pl.col("n").sum())
        full = game.filter(pl.col("group") == "all").join(summed, on=["model", "reference", "metric"], suffix="_partition")
        assert (full["n"] == full["n_partition"]).all()
        errors.append((full["diff"] - full["diff_partition"]).abs().max())
    assert max(errors) < 1e-10
    rows.append(dict(speed=speed, fold=fold, max_comparison_error=max(errors), fits_checked=True, coverage_checked=True))
pl.DataFrame(rows).write_csv("outputs/tables/entry_context_checks.csv")
print(pl.DataFrame(rows))
