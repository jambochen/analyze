# 彙整表的分母、提醒重疊、分組與時間界線互相核對。
import numpy as np
import polars as pl

s = pl.read_csv("outputs/tables/player_alerts_evaluation.csv")
o = pl.read_csv("outputs/tables/player_alerts_overlap.csv")
t = pl.read_csv("outputs/tables/player_alerts_thresholds.csv")
c = pl.read_csv("outputs/tables/player_alerts_checks.csv")
assert s.height == 4 * 3 * 4 * 2
assert np.isfinite(s.select(pl.selectors.numeric()).to_numpy()).all()
assert (t["train_max_time"] < t["test_start"]).all()
assert c["max_history_error"].max() < 1e-10
rows = []
for (speed, fold), part in s.partition_by("speed", "fold", as_dict=True).items():
    a = part.filter(pl.col("group") == "all", pl.col("model") == "recent")
    overlap = o.filter(pl.col("speed") == speed, pl.col("fold") == fold)
    assert overlap["n"].sum() == a["n"][0]
    assert abs(overlap["share"].sum() - 1) < 1e-12
    for rule, categories in [("loss3", ["both", "loss_only"]), ("adjusted_matched", ["both", "adjusted_only"])]:
        assert overlap.filter(pl.col("category").is_in(categories))["n"].sum() == a.filter(pl.col("rule") == rule)["alerted"].item()
    for rule in part["rule"].unique():
        groups = part.filter(pl.col("rule") == rule, pl.col("model") == "recent")
        for col in ["n", "alerted"]:
            assert groups.filter(pl.col("group") != "all")[col].sum() == groups.filter(pl.col("group") == "all")[col].item()
    rows.append(dict(speed=speed, fold=fold, n=a["n"][0], partition_and_overlap=True))
pl.DataFrame(rows).write_csv("outputs/tables/player_alerts_validation.csv")
print(pl.DataFrame(rows))
