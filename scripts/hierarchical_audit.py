# 完整交付前核對實際產物；指定賽制可先檢查已完成部分，結案時必須跑全部。
import os
os.environ["JAX_PLATFORMS"] = "cpu"
import json
import hashlib
from pathlib import Path
import numpy as np
import polars as pl
import state_adjusted as sa

speeds = [os.environ["HIERARCHICAL_AUDIT_SPEED"]] if "HIERARCHICAL_AUDIT_SPEED" in os.environ else ["Bullet", "Blitz", "Classical"]
original = pl.read_csv("outputs/tables/likelihood_selection_sample.csv")
reference_hash = hashlib.sha256(Path("outputs/tables/likelihood_selection_models.json").read_bytes()).hexdigest()
rows = []
for speed in speeds:
    for fold in ["Jul-Sep", "Oct-Dec"]:
        for kind in ["slope", "intercept"]:
            prefix = f"hierarchical_{speed}_{fold}_{kind}"
            meta = json.loads(Path(f"outputs/tables/{prefix}_fit.json").read_text())
            diag = pl.read_csv(f"outputs/tables/{prefix}_diagnostics.csv")
            assert np.isfinite(diag.select("r_hat", "ess_bulk").to_numpy()).all()
            assert meta["divergences"] == 0 and diag["r_hat"].max() < 1.01 and diag["ess_bulk"].min() > 400
            assert meta["source_hashes"]["outputs/tables/likelihood_selection_models.json"] == reference_hash
            d = pl.read_parquet(f"data/processed/{prefix}_prediction.parquet")
            old = original.filter(pl.col("speed") == speed, pl.col("fold") == fold, pl.col("segment") == "test").row(0, named=True)
            assert len(d) == old["n"] and d["game_id"].n_unique() == old["games"] and d["novice"].sum() == old["novice"]
            assert d.select(pl.struct("game_id", "a_id").n_unique()).item() == len(d)
            p = d.select("p_loss", "p_draw", "p_win").to_numpy()
            y = d["y"].to_numpy()
            assert np.isfinite(p).all() and (p > 0).all() and np.allclose(p.sum(1), 1, atol=1e-14)
            loss = sa.losses(p, y) | {"score_mse": (y/2-p[:, 2]-p[:, 1]/2)**2}
            for metric, value in loss.items():
                assert np.allclose(value, d[f"{kind}_{metric}"].to_numpy(), rtol=1e-12, atol=1e-14)
            # 用實際雙方預測驗證整條特徵與 posterior 流程，不僅驗證單獨的 logistic 函數。
            opposite = d.select("game_id", a_id="b_id", b_id="a_id", other_loss="p_loss", other_draw="p_draw", other_win="p_win")
            pair = d.join(opposite, on=["game_id", "a_id", "b_id"], how="inner")
            assert pair.height > 0
            swap_error = pair.select(pl.max_horizontal((pl.col("p_win")-pl.col("other_loss")).abs(),
                                                       (pl.col("p_loss")-pl.col("other_win")).abs(),
                                                       (pl.col("p_draw")-pl.col("other_draw")).abs())).max().item()
            assert swap_error < 1e-12, (prefix, swap_error)
            groups = {"all": np.ones(len(d), bool), "novice": d["novice"].to_numpy(), "established": ~d["novice"].to_numpy(),
                      "both_seen": d["both_seen"].to_numpy(), "any_unseen": ~d["both_seen"].to_numpy()}
            table = pl.read_csv(f"outputs/tables/{prefix}_prediction.csv")
            assert table.height == len(groups)*3*3
            for row in table.iter_rows(named=True):
                mask = groups[row["group"]]
                delta = (loss[row["metric"]]-d[f"{row['vs']}_{row['metric']}"].to_numpy())[mask]
                assert row["n"] == int(mask.sum()) and np.isclose(row["diff"], delta.mean(), atol=1e-12, rtol=1e-9)
                assert np.isfinite(row["se"]) and row["se"] >= 0
                assert np.isclose(row["lo"], row["diff"]-1.96*row["se"], atol=1e-14)
                assert np.isclose(row["hi"], row["diff"]+1.96*row["se"], atol=1e-14)
            cal = pl.read_csv(f"outputs/tables/{prefix}_calibration.csv")
            totals = cal.group_by("group", "model", "target", "binning").agg(pl.col("n").sum())
            assert totals.height == len(groups)*4*4*2
            for row in totals.iter_rows(named=True):
                assert row["n"] == int(groups[row["group"]].sum())
            assert cal.filter(pl.col("binning") == "baseline_prediction").group_by("group", "target", "bin").agg(
                pl.col("observed").n_unique(), pl.col("n").n_unique()).select((pl.col("observed") == 1).all() & (pl.col("n") == 1).all()).item()
            mc = pl.read_csv(f"outputs/tables/{prefix}_mc.csv").filter(pl.col("metric") == "log_loss")
            assert mc["chain"].n_unique() == 4
            mcse = mc["diff_vs_selected"].std()/np.sqrt(4)
            se = table.filter(pl.col("group") == "all", pl.col("vs") == "selected", pl.col("metric") == "log_loss")["se"].item()
            assert mcse/se < .1, (prefix, mcse, se)
            rows.append({"speed": speed, "fold": fold, "kind": kind, "n": len(d), "swap_error": swap_error,
                         "rhat_max": diag["r_hat"].max(), "ess_bulk_min": diag["ess_bulk"].min(), "mcse_to_dyadic_se": mcse/se})
suffix = "_"+speeds[0] if len(speeds) == 1 else ""
pl.DataFrame(rows).write_csv(f"outputs/tables/hierarchical_audit{suffix}.csv")
print(pl.DataFrame(rows))
