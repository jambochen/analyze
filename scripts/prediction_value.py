# 固定時間外模型的改善幅度、校準與匿名情境；不重估、不選項、不使用前瞻樣本。
import os
import time

os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import polars as pl

import ordinal
import state_adjusted as sa
import temporal_models as tm

coef = pl.read_csv("outputs/tables/temporal_prediction_coef.csv")
state = pl.read_csv("outputs/tables/temporal_prediction_state_coef.csv")
reported = pl.read_csv("outputs/tables/temporal_prediction_prediction.csv")
samples = pl.read_csv("outputs/tables/temporal_prediction_sample.csv")
score_rows = reported.with_columns(reference=pl.col("mean") - pl.col("diff")).with_columns(
    gain=-pl.col("diff"), relative_gain_pct=-100 * pl.col("diff") / pl.col("reference"),
    # 分母是固定的參考損失，只轉換既有條件區間，不把它當作比值的新推論。
    relative_gain_lo_pct=-100 * pl.col("hi") / pl.col("reference"),
    relative_gain_hi_pct=-100 * pl.col("lo") / pl.col("reference"),
)
score_rows.write_csv("outputs/tables/prediction_value_scores.csv")
changes, calibration, bins_rows, scenarios, checks = [], [], [], [], []
started = time.monotonic()

for game in ["Lichess", "OGS", "AoE2"]:
    tick = time.monotonic()
    t, folds, _ = tm.sample(game)
    t = t.sort("time", "game_id", "a_id")
    assert t.select(pl.struct("game_id", "a_id").n_unique()).item() == t.height
    print(game, "sample", t.height, "seconds", time.monotonic() - tick, flush=True)
    for speed in sorted(t["speed"].unique().to_list()):
        ts = t.filter(pl.col("speed") == speed)
        for fold, _, start, end in folds:
            lo, hi = [int(np.datetime64(x, "s").astype(np.int64)) for x in [start, end]]
            key = {"game": game, "speed": speed, "fold": fold}
            cc = coef.filter(pl.col("game") == game, pl.col("speed") == speed, pl.col("fold") == fold, pl.col("fit") == "base_refit").sort("param_index")
            te = tm.offset(ts.filter(pl.col("time").is_between(lo, hi, closed="left")), cc["est"].to_numpy(), game)
            ns = samples.filter(pl.col("game") == game, pl.col("speed") == speed, pl.col("fold") == fold, pl.col("segment") == "test").row(0, named=True)
            assert te.height == ns["n"] and te["time"].min() == ns["time_min"] and te["time"].max() == ns["time_max"]
            P = {"baseline": ordinal.probs(te["eta"].to_numpy(), te["width"].to_numpy())}
            coefficients = {}
            for model in ["recent", "novice_extension"]:
                sc = state.filter(pl.col("game") == game, pl.col("speed") == speed, pl.col("fold") == fold, pl.col("model") == model)
                coefficients[model] = dict(zip(sc["param"], sc["est"]))
                assert coefficients[model]["baseline_offset"] == 1
                eta = te["eta"].to_numpy().copy()
                for name, value in coefficients[model].items():
                    if name != "baseline_offset":
                        eta += te[name].to_numpy() * value
                P[model] = ordinal.probs(eta, te["width"].to_numpy())
            assert all(np.isfinite(p).all() and (p >= 0).all() and np.allclose(p.sum(1), 1) for p in P.values())
            E = {model: p[:, 2] + p[:, 1] / 2 for model, p in P.items()}
            E["system"] = te["p_sys"].to_numpy()
            y = te["y"].to_numpy()
            loss = {model: sa.losses(p, y) for model, p in P.items()}
            if game != "Lichess":
                loss["system"] = sa.losses(np.c_[1 - E["system"], np.zeros(te.height), E["system"]], y)
            for model in E:
                loss.setdefault(model, {})["score_mse"] = (y / 2 - E[model])**2
            targets = {"score": (y / 2, E)}
            if game == "Lichess":
                for k, target in enumerate(["loss", "draw", "win"]):
                    targets[target] = ((y == k).astype(float), {model: p[:, k] for model, p in P.items()})
            for group, mask in [("all", np.ones(te.height, bool)), ("novice", te["novice"].to_numpy()), ("established", ~te["novice"].to_numpy())]:
                # 重建值須逐格重現既有損失與差值，才能解讀新增摘要。
                rr = reported.filter(pl.col("game") == game, pl.col("speed") == speed, pl.col("fold") == fold, pl.col("group") == group)
                errors = []
                for row in rr.iter_rows(named=True):
                    assert row["n"] == int(mask.sum())
                    val = loss[row["model"]][row["metric"]][mask].mean()
                    diff = (loss[row["model"]][row["metric"]] - loss[row["vs"]][row["metric"]])[mask].mean()
                    errors.extend([abs(val - row["mean"]), abs(diff - row["diff"])])
                assert max(errors) < 1e-10, (key, group, max(errors))
                checks.append({**key, "group": group, "n": int(mask.sum()), "max_loss_error": max(errors)})
                for model, reference in [("recent", "baseline"), ("novice_extension", "recent"), ("novice_extension", "baseline")]:
                    win = (P[model][:, 2] - P[reference][:, 2])[mask] * 100
                    expected = (E[model] - E[reference])[mask] * 100
                    changes.append({**key, "group": group, "model": model, "vs": reference, "n": int(mask.sum()),
                                    "mean_abs_win_pp": np.abs(win).mean(), "median_abs_win_pp": np.median(np.abs(win)), "p90_abs_win_pp": np.quantile(np.abs(win), .9),
                                    "mean_abs_score_pp": np.abs(expected).mean(), "median_abs_score_pp": np.median(np.abs(expected)), "p90_abs_score_pp": np.quantile(np.abs(expected), .9)})
                for target, (observed, predictions) in targets.items():
                    for binning in ["own_prediction", "baseline_prediction"]:
                        for model, expected in predictions.items():
                            anchor = expected if binning == "own_prediction" else predictions["baseline"]
                            bins = np.minimum((anchor * 10).astype(int), 9)
                            errors, counts = [], []
                            for b in np.unique(bins[mask]):
                                keep = mask & (bins == b)
                                pred, obs = expected[keep].mean(), observed[keep].mean()
                                n = int(keep.sum())
                                errors.append(obs - pred)
                                counts.append(n)
                                bins_rows.append({**key, "group": group, "target": target, "binning": binning, "model": model, "bin": int(b), "n": n, "predicted": pred, "observed": obs, "residual": obs - pred})
                            calibration.append({**key, "group": group, "target": target, "binning": binning, "model": model, "n": sum(counts),
                                                "weighted_abs_gap_pp": 100 * np.average(np.abs(errors), weights=counts),
                                                "weighted_rms_gap_pp": 100 * np.sqrt(np.average(np.square(errors), weights=counts))})
            if game == "Lichess":
                # 只用訓練期局前特徵選代表列，避免挑測試改善最大的案例，也不拼接不可能的中位數組合。
                tr = ts.filter(pl.col("time") < lo, pl.col("sign") == 1, pl.col("opp_missing") == 0, pl.col("p_sys").is_between(.4, .6))
                features = ["elo", "opp_elo", "rd", "opp_rd", "dr", "opp_dr"]
                for group in ["novice", "established"]:
                    for own, opponent in [("loss", "win"), ("win", "loss"), ("loss", "loss"), ("win", "win")]:
                        cell = tr.filter(pl.col("novice") == (group == "novice"), pl.col("prev_result") == own, pl.col("opp_prev_result") == opponent)
                        # 缺少該情境時不外插或從別的分組借列。
                        if cell.height == 0:
                            continue
                        X = cell.select(features).to_numpy()
                        scale = np.quantile(X, .75, axis=0) - np.quantile(X, .25, axis=0)
                        scale[scale == 0] = 1
                        closest = int(np.argmin((np.abs(X - np.median(X, axis=0)) / scale).sum(1)))
                        d = tm.offset(cell.slice(closest, 1), cc["est"].to_numpy(), game)
                        feature_row = {name: float(d[name][0]) for name in features}
                        eta0, width = d["eta"][0], d["width"][0]
                        base_p = ordinal.probs(np.array([eta0]), np.array([width]))[0]
                        for model in ["baseline", "recent", "novice_extension"]:
                            eta = eta0 + sum(value * d[name][0] for name, value in coefficients.get(model, {}).items() if name != "baseline_offset")
                            p = ordinal.probs(np.array([eta]), np.array([width]))[0]
                            scenarios.append({**key, "group": group, "previous": own, "opponent_previous": opponent, "training_cell_n": cell.height, "model": model, **feature_row,
                                              "p_loss": p[0], "p_draw": p[1], "p_win": p[2], "expected_score": p[2] + p[1] / 2,
                                              "win_change_pp": 100 * (p[2] - base_p[2]), "score_change_pp": 100 * (p[2] + p[1] / 2 - base_p[2] - base_p[1] / 2)})
            print(game, speed, fold, "verified", te.height, "elapsed", time.monotonic() - started, flush=True)
    for label, rows in [("changes", changes), ("calibration", calibration), ("calibration_bins", bins_rows), ("scenarios", scenarios), ("checks", checks)]:
        pl.DataFrame(rows).write_csv(f"outputs/tables/prediction_value_{label}.csv")
    del t, ts, te
print("seconds", time.monotonic() - started)
