# 已保存的預測重新整理校準分箱，不重抽 posterior、不重新估計。
import os
os.environ["JAX_PLATFORMS"] = "cpu"
import json
from pathlib import Path
import numpy as np
import polars as pl
from hierarchical_predictive import calibration_rows
import temporal_models as tm
import rd_spline as spline

speed, fold, kind = [os.environ[k] for k in ["HIERARCHICAL_SPEED", "HIERARCHICAL_FOLD", "HIERARCHICAL_KIND"]]
prefix = f"hierarchical_{speed}_{fold}_{kind}"
saved = pl.read_parquet(f"data/processed/{prefix}_prediction.parquet")
d = pl.read_parquet("data/processed/hierarchical_prediction.parquet").join(saved.select("game_id", "a_id"), on=["game_id", "a_id"], how="semi", maintain_order="left")
assert d.select("game_id", "a_id").equals(saved.select("game_id", "a_id"))
export = next(m for m in json.loads(Path("outputs/tables/likelihood_selection_models.json").read_text()) if m["speed"] == speed and m["fold"] == fold)
P = {kind: saved.select("p_loss", "p_draw", "p_win").to_numpy()}
for name in ["baseline", "recent", "selected"]:
    m = export["models"][name if name == "baseline" else export["aliases"][name]]
    P[name] = spline.probabilities(tm.offset(d, m["baseline_coef"], "Lichess"), m)
groups = [("all", np.ones(len(d), bool)), ("novice", d["novice"].to_numpy()), ("established", ~d["novice"].to_numpy()),
          ("both_seen", saved["both_seen"].to_numpy()), ("any_unseen", ~saved["both_seen"].to_numpy())]
rows = [{"speed": speed, "fold": fold, "kind": kind, **r} for r in calibration_rows(d["y"].to_numpy(), P, groups)]
out = pl.DataFrame(rows)
assert out.group_by("group", "model", "target", "binning").agg(pl.col("n").sum()).filter(pl.col("group") == "all")["n"].unique().to_list() == [len(d)]
# 共同分箱時各模型的觀察比例與局數必須相同。
assert out.filter(pl.col("binning") == "baseline_prediction").group_by("group", "target", "bin").agg(
    pl.col("observed").n_unique(), pl.col("n").n_unique()).select((pl.col("observed") == 1).all() & (pl.col("n") == 1).all()).item()
out.write_csv(f"outputs/tables/{prefix}_calibration.csv")
print(prefix, "calibration", len(out), flush=True)
