# 替代對手的白方計分須重現既有模型在白黑雙方視角的完整預測，包含缺歷史情境。
import os
os.environ["JAX_PLATFORMS"] = "cpu"
import numpy as np
import polars as pl

from matching_probabilities import measures
import ordinal
import temporal_models as tm

t, folds, _ = tm.sample("Lichess")
coef = pl.read_csv("outputs/tables/temporal_prediction_coef.csv")
state = pl.read_csv("outputs/tables/temporal_prediction_state_coef.csv")
cols = ["d_prev_draw", "d_prev_win", "opp_missing", "d_dr"]
rows = []
for speed in ["Blitz", "Bullet", "Classical"]:
    for fold, *_ in folds:
        th = coef.filter(pl.col("game") == "Lichess", pl.col("speed") == speed, pl.col("fold") == fold, pl.col("fit") == "base_refit").sort("param_index")["est"].to_numpy()
        bc = dict(state.filter(pl.col("game") == "Lichess", pl.col("speed") == speed, pl.col("fold") == fold, pl.col("model") == "recent").select("param", "est").iter_rows())
        assert set(bc) <= set(cols + ["baseline_offset"])
        beta = np.array([bc.get(c, 0.) for c in cols])
        for sign in [-1, 1]:
            for missing in [0., 1.]:
                d = t.filter(pl.col("speed") == speed, pl.col("sign") == sign, pl.col("opp_missing") == missing).head(1000)
                a = d.select("elo", "rd", (pl.col("prev_result") == "draw").cast(pl.Float64).alias("draw"), (pl.col("prev_result") == "win").cast(pl.Float64).alias("win"), pl.lit(0.).alias("missing"), "dr").to_numpy()
                b = d.select("opp_elo", "opp_rd", (pl.col("opp_prev_result") == "draw").fill_null(False).cast(pl.Float64).alias("draw"), (pl.col("opp_prev_result") == "win").fill_null(False).cast(pl.Float64).alias("win"), "opp_missing", "opp_dr").to_numpy()
                v = measures(a, b, th, beta) if sign == 1 else measures(b, a, th, beta)
                d = tm.offset(d, th, "Lichess")
                for model, ci in [("baseline", 1), ("recent", 2)]:
                    eta = d["eta"].to_numpy().copy()
                    if model == "recent":
                        for name, value in bc.items():
                            if name != "baseline_offset":
                                eta += value * d[name].to_numpy()
                    p = ordinal.probs(eta, d["width"].to_numpy())
                    error = np.max(abs(v[:, ci] - abs(p[:, 2] + p[:, 1] / 2 - .5)))
                    assert error < 1e-12
                    rows.append(dict(speed=speed, fold=fold, model=model, sign=sign, opponent_missing=missing, rows=d.height, max_error=error))
pl.DataFrame(rows).write_csv("outputs/tables/matching_policy_prediction_checks.csv")
print(pl.DataFrame(rows).group_by("model").agg(cells=pl.len(), max_error=pl.col("max_error").max()))
