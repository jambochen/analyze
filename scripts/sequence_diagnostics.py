# 固定觀察設計的人工參考，不是重建評分與續下機制的生成檢定。
import os
import time

os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import polars as pl

import ordinal
import state_adjusted as sa
import temporal_models as tm
from sequence_summary import summarize, window_sum

reps = int(os.environ.get("SEQUENCE_REPS", "200"))
prefix = os.environ.get("SEQUENCE_PREFIX", "outputs/tables/sequence_diagnostics")
coef = pl.read_csv("outputs/tables/temporal_prediction_coef.csv")
state = pl.read_csv("outputs/tables/temporal_prediction_state_coef.csv")
reported = pl.read_csv("outputs/tables/temporal_prediction_prediction.csv")
samples = pl.read_csv("outputs/tables/temporal_prediction_sample.csv")
started = time.monotonic()
t, folds, _ = tm.sample("Lichess")
print("sample", t.height, "seconds", time.monotonic() - started, flush=True)
rows, checks, timings = [], [], []
for speed in ["Bullet", "Blitz"]:
    for fold, _, start, end in folds:
        lo, hi = [int(np.datetime64(x, "s").astype(np.int64)) for x in [start, end]]
        key = {"game": "Lichess", "speed": speed, "fold": fold}
        cc = coef.filter(pl.col("game") == "Lichess", pl.col("speed") == speed, pl.col("fold") == fold, pl.col("fit") == "base_refit").sort("param_index")
        te = tm.offset(t.filter(pl.col("speed") == speed, pl.col("time").is_between(lo, hi, closed="left")), cc["est"].to_numpy(), "Lichess").sort("a_id", "tournament_id", "time", "game_id")
        ns = samples.filter(pl.col("game") == "Lichess", pl.col("speed") == speed, pl.col("fold") == fold, pl.col("segment") == "test").row(0, named=True)
        assert te.height == ns["n"] and te["time"].min() == ns["time_min"] and te["time"].max() == ns["time_max"]
        assert te.select(pl.struct("game_id", "a_id").n_unique()).item() == te.height
        y = te["y"].to_numpy()
        ids = te["a_id"].to_numpy()
        tour = te["tournament_id"].to_numpy()
        pos = te["games_in_tour"].to_numpy().astype(np.int64)
        boundary = np.r_[True, (ids[1:] != ids[:-1]) | (tour[1:] != tour[:-1]) | (np.diff(pos) != 1)]
        segment = np.cumsum(boundary)
        _, first, game_index = np.unique(te["game_id"].to_numpy(), return_index=True, return_inverse=True)
        white = te["sign"].to_numpy() == 1
        wy = np.where(white, y, 2 - y)
        assert np.array_equal(wy, wy[first][game_index])
        windows, metadata = [], []
        for metric, k in [("loss_run", k) for k in [2, 3, 4, 5]] + [("cumulative_square", k) for k in [5, 10, 20]]:
            valid = np.flatnonzero(segment[k-1:] == segment[:len(segment)-k+1])
            start_ids = ids[valid]
            own_novice = te["rd"].to_numpy()[valid] > 110
            masks, weights = [], []
            for group, mask in [("all", np.ones(len(valid), bool)), ("own_novice", own_novice), ("own_established", ~own_novice)]:
                _, inv, counts = np.unique(start_ids[mask], return_inverse=True, return_counts=True)
                weight = 1 / counts[inv] / len(counts)
                assert len(counts) == 0 or np.isclose(weight.sum(), 1)
                masks.append(mask)
                weights.append(weight)
                for weighting in ["window", "player"]:
                    metadata.append({**key, "metric": metric, "length": k, "group": group, "weighting": weighting, "n_windows": int(mask.sum()), "n_players": len(counts)})
            windows.append((metric, k, valid, masks, weights))
        for model in ["baseline", "recent", "novice_extension"]:
            eta = te["eta"].to_numpy().copy()
            if model != "baseline":
                sc = state.filter(pl.col("game") == "Lichess", pl.col("speed") == speed, pl.col("fold") == fold, pl.col("model") == model)
                pars = dict(zip(sc["param"], sc["est"]))
                assert pars.pop("baseline_offset") == 1
                for name, value in pars.items():
                    eta += te[name].to_numpy() * value
            p = ordinal.probs(eta, te["width"].to_numpy())
            assert np.isfinite(p).all() and (p > 0).all() and np.allclose(p.sum(1), 1)
            wp = np.where(white[:, None], p, p[:, ::-1])
            symmetry_error = np.max(np.abs(wp - wp[first][game_index]))
            assert symmetry_error < 1e-10
            losses = sa.losses(p, y)
            losses["score_mse"] = (y / 2 - p[:, 2] - p[:, 1] / 2)**2
            errors = []
            for group, mask in [("all", np.ones(len(y), bool)), ("novice", te["novice"].to_numpy()), ("established", ~te["novice"].to_numpy())]:
                rr = reported.filter(pl.col("game") == "Lichess", pl.col("speed") == speed, pl.col("fold") == fold, pl.col("group") == group, pl.col("model") == model)
                for row in rr.iter_rows(named=True):
                    assert row["n"] == int(mask.sum())
                    errors.append(abs(losses[row["metric"]][mask].mean() - row["mean"]))
            # baseline 是比較參考，既有表未必單獨列出；用 recent 的差值還原。
            if model == "baseline":
                rr = reported.filter(pl.col("game") == "Lichess", pl.col("speed") == speed, pl.col("fold") == fold, pl.col("group") == "all", pl.col("vs") == "baseline")
                for row in rr.iter_rows(named=True):
                    errors.append(abs(losses[row["metric"]].mean() - (row["mean"] - row["diff"])))
            assert errors and max(errors) < 1e-10
            expected = p[:, 2] + p[:, 1] / 2
            variance = p[:, 2] + p[:, 1] / 4 - expected ** 2
            observed = summarize(y / 2, expected, variance, windows)
            analytic = []
            for metric, k, valid, masks, weights in windows:
                ref = np.exp(window_sum(np.log(p[:, 0]), k)[valid]) if metric == "loss_run" else np.ones(len(valid))
                for mask, weight in zip(masks, weights):
                    analytic.extend([ref[mask].mean(), np.dot(ref[mask], weight)] if mask.any() else [np.nan, np.nan])
            rng = np.random.default_rng(20261113)
            sim = []
            tick = time.monotonic()
            for rep in range(reps):
                u = rng.random(len(first))
                gp = wp[first]
                draw = (u >= gp[:, 0]).astype(np.int8) + (u >= gp[:, :2].sum(1)).astype(np.int8)
                sy = np.where(white, draw[game_index], 2 - draw[game_index]) / 2
                sim.append(summarize(sy, expected, variance, windows))
                if rep == 0:
                    unit = time.monotonic() - tick
                    print(key, model, "one_draw_seconds", unit, "projected_draw_seconds", unit * reps, flush=True)
            sim = np.array(sim)
            for j, meta in enumerate(metadata):
                rows.append({**meta, "model": model, "observed": observed[j], "analytic_reference": analytic[j], "simulation_mean": sim[:, j].mean(), "reference_lo": np.quantile(sim[:, j], .025), "reference_hi": np.quantile(sim[:, j], .975), "mc_se_mean": sim[:, j].std(ddof=1) / np.sqrt(reps), "replicates": reps})
            checks.append({**key, "model": model, "n_player_games": len(y), "n_games": len(first), "n_segments": int(boundary.sum()), "max_loss_error": max(errors), "max_symmetry_error": symmetry_error})
            timings.append({**key, "model": model, "replicates": reps, "seconds": time.monotonic() - tick})
            print(key, model, "seconds", timings[-1]["seconds"], flush=True)
pl.DataFrame(rows).write_csv(prefix + "_summary.csv")
pl.DataFrame(checks).write_csv(prefix + "_checks.csv")
pl.DataFrame(timings).write_csv(prefix + "_timing.csv")
print("total_seconds", time.monotonic() - started, flush=True)
