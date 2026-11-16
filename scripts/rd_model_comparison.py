# RD 樣條與既有交互作用的有限共同流程比較；只新增樣條配適，不重新使用前瞻樣本。
import os
import json
import time
import resource
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import polars as pl

import likelihood as lk
import ordinal
import rd_spline as spline
import state_adjusted as sa
import temporal_models as tm

SMOKE = os.environ.get("RD_COMPARISON_SMOKE") == "1"
PREFIX = "rd_model_comparison_smoke" if SMOKE else "rd_model_comparison"
coef = pl.read_csv("outputs/tables/temporal_prediction_coef.csv").filter(pl.col("game") == "Lichess")
steps = pl.read_csv("outputs/tables/temporal_prediction_selection.csv").filter(pl.col("game") == "Lichess")
state = pl.read_csv("outputs/tables/temporal_prediction_state_coef.csv").filter(pl.col("game") == "Lichess")
models = pl.read_csv("outputs/tables/temporal_prediction_models.csv").filter(pl.col("game") == "Lichess")
reported = pl.read_csv("outputs/tables/temporal_prediction_prediction.csv").filter(pl.col("game") == "Lichess")
reported_samples = pl.read_csv("outputs/tables/temporal_prediction_sample.csv").filter(pl.col("game") == "Lichess")
fits, selections, predictions, curves, calibration, checks, samples, coefficients, transforms = [], [], [], [], [], [], [], [], []
exports, changes, scenarios = [], [], []
example_inputs = pl.read_csv("outputs/tables/prediction_value_scenarios.csv").filter(pl.col("game") == "Lichess", pl.col("model") == "baseline")
started = time.monotonic()


def difference(a, b, mask, d):
    assert np.isfinite(a).all() and np.isfinite(b).all()
    delta = (a - b)[mask]
    se = lk.dyad_se((delta - delta.mean()) / len(delta), d["a_id"].to_numpy()[mask], d["b_id"].to_numpy()[mask])
    assert np.isfinite(se) and np.isfinite(delta).all()
    return float(delta.mean()), float(se)


def saved_model(columns, values):
    return {"columns": columns, "coef": list(values), "spline_coef": [], "knots": [], "center": []}


def fit_spline(d, columns, values, label):
    tick = time.monotonic()
    pooled = np.r_[d["rd"].to_numpy(), d["opp_rd"].to_numpy()]
    knots = np.quantile(np.unique(pooled), np.linspace(0, 1, 5))
    assert np.all(np.diff(knots) > 0)
    center = spline.basis(pooled, knots).mean(0)
    X = np.column_stack([d["eta"].to_numpy(), d.select(columns).to_numpy(), spline.design(d, knots, center)])
    assert np.linalg.matrix_rank(X[:, 1:].T @ X[:, 1:]) == X.shape[1] - 1
    t0 = np.r_[1., values, np.zeros(4)]
    th, _, _, _, score, H = lk.fit(tm.state_ll, t0, np.arange(1, len(t0)), (X, d["width"].to_numpy(), d["y"].to_numpy()), d["a_id"].to_numpy(), d["b_id"].to_numpy())
    scaled = np.max(np.abs(score.sum(0)) / np.sqrt(np.diag(H)))
    assert np.isfinite(th).all() and scaled < .01 and np.linalg.eigvalsh(H).min() > 0
    fits.append({**key, "fit": label, "n": d.height, "seconds": time.monotonic()-tick, "score_scaled_max": scaled, "hessian_condition": np.linalg.cond(H), "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024})
    for j, (knot, mean) in enumerate(zip(knots, center)):
        transforms.append({**key, "fit": label, "basis": j, "knot": knot, "center": mean})
    for name, value in zip(["baseline_offset"]+columns+[f"rd_spline_{j}" for j in range(4)], th):
        coefficients.append({**key, "fit": label, "param": name, "est": value})
    print(speed, fold, label, fits[-1], flush=True)
    return {"columns": columns, "coef": th[1:1+len(columns)].tolist(), "spline_coef": th[1+len(columns):].tolist(), "knots": knots.tolist(), "center": center.tolist()}


t, folds, _ = tm.sample("Lichess")
t = t.filter(pl.col("speed").is_in(["Bullet", "Blitz"]))
if SMOKE:
    t = t.filter(pl.col("game_id").hash(seed=42) % 40 == 0)
t = t.sort("time", "game_id", "a_id")
assert t.select(pl.struct("game_id", "a_id").n_unique()).item() == t.height
print("sample", t.height, "seconds", time.monotonic()-started, flush=True)
for speed in ["Bullet", "Blitz"]:
    ts = t.filter(pl.col("speed") == speed)
    for fold, inner, start, end in folds:
        key = {"speed": speed, "fold": fold}
        lo, mid, hi = [int(np.datetime64(x, "s").astype(np.int64)) for x in [inner, start, end]]
        raw = {"fit": ts.filter(pl.col("time") < lo), "validation": ts.filter(pl.col("time").is_between(lo, mid, closed="left")), "train_refit": ts.filter(pl.col("time") < mid), "test": ts.filter(pl.col("time").is_between(mid, hi, closed="left"))}
        assert raw["fit"]["time"].max() < raw["validation"]["time"].min()
        assert raw["train_refit"]["time"].max() < raw["test"]["time"].min()
        for segment, d in raw.items():
            samples.append({**key, "segment": segment, "n": d.height, "games": d["game_id"].n_unique(), "novice": d["novice"].sum(), "time_min": d["time"].min(), "time_max": d["time"].max()})
            if not SMOKE:
                old = reported_samples.filter(pl.col("speed") == speed, pl.col("fold") == fold, pl.col("segment") == segment).row(0, named=True)
                assert all(samples[-1][c] == old[c] for c in ["n", "games", "novice", "time_min", "time_max"])
        cc = coef.filter(pl.col("speed") == speed, pl.col("fold") == fold)
        bs_inner = cc.filter(pl.col("fit") == "base_inner").sort("param_index")["est"].to_numpy()
        bs_refit = cc.filter(pl.col("fit") == "base_refit").sort("param_index")["est"].to_numpy()
        f, v = [tm.offset(raw[segment], bs_inner, "Lichess") for segment in ["fit", "validation"]]
        tr, te = [tm.offset(raw[segment], bs_refit, "Lichess") for segment in ["train_refit", "test"]]
        ss = steps.filter(pl.col("speed") == speed, pl.col("fold") == fold)
        inner_models = {}
        for model, labels in [("recent", ["recent_results", "rating_change"]), ("interaction", ["recent_results", "rating_change", "rd_interaction", "direction_interaction"])]:
            kept = ss.filter(pl.col("step").is_in(labels), pl.col("kept"))
            if kept.height:
                row = kept.row(-1, named=True)
                cols = row["columns"].split(",")
                values = cc.filter(pl.col("fit") == "select_"+row["step"]).sort("param_index")["est"].to_numpy()
                assert values[0] == 1 and len(values) == len(cols)+1
                inner_models[model] = saved_model(cols, values[1:])
            else:
                inner_models[model] = saved_model([], [])
        for model in inner_models:
            old_model = "novice_extension" if model == "interaction" else model
            old_cols = models.filter(pl.col("speed") == speed, pl.col("fold") == fold, pl.col("model") == old_model)["columns"].item()
            assert ",".join(inner_models[model]["columns"]) == old_cols
        inner_models["spline"] = fit_spline(f, inner_models["recent"]["columns"], inner_models["recent"]["coef"], "spline_inner")
        valid_prob = {nm: spline.probabilities(v, m) for nm, m in inner_models.items()}
        assert all(np.isfinite(p).all() and (p >= 0).all() and np.allclose(p.sum(1), 1) and (p[np.arange(v.height), v["y"].to_numpy()] > 0).all() for p in valid_prob.values())
        valid_loss = {nm: sa.losses(p, v["y"].to_numpy())["log_loss"] for nm, p in valid_prob.items()}
        mask = v["novice"].to_numpy()
        accepted = {}
        for model, reference in [("interaction", "recent"), ("spline", "recent"), ("spline", "interaction")]:
            diff, se = difference(valid_loss[model], valid_loss[reference], mask, v)
            accepted[(model, reference)] = diff+1.96*se < 0
            selections.append({**key, "model": model, "vs": reference, "group": "novice", "n": int(mask.sum()), "diff": diff, "se": se, "lo": diff-1.96*se, "hi": diff+1.96*se, "improved": accepted[(model, reference)]})
        selected = "recent"
        if accepted[("interaction", "recent")]:
            selected = "interaction"
        if accepted[("spline", "recent")] and (selected == "recent" or accepted[("spline", "interaction")]):
            selected = "spline"
        final = {}
        for model, old_model in [("recent", "recent"), ("interaction", "novice_extension")]:
            sc = state.filter(pl.col("speed") == speed, pl.col("fold") == fold, pl.col("model") == old_model)
            assert sc.filter(pl.col("param") == "baseline_offset")["est"].item() == 1
            sc = sc.filter(pl.col("param") != "baseline_offset")
            final[model] = saved_model(sc["param"].to_list(), sc["est"].to_list())
        final["spline"] = fit_spline(tr, final["recent"]["columns"], final["recent"]["coef"], "spline_refit")
        export = {**key, "selected": selected, "train_end_exclusive": start, "test_end_exclusive": end, "baseline_coef": bs_refit.tolist(), "spline_df": 5, "dropped_basis": 4, "knot_rule": "quantiles_of_distinct_pooled_training_rd", "models": final}
        # JSON 重建與記憶體模型須相同，避免圖表有結果卻無法重現交付模型。
        roundtrip = json.loads(json.dumps(export))
        P = {nm: spline.probabilities(te, m) for nm, m in final.items()}
        P["baseline"] = ordinal.probs(te["eta"].to_numpy(), te["width"].to_numpy())
        P["selected"] = P[selected]
        for nm in final:
            assert np.allclose(P[nm], spline.probabilities(te, roundtrip["models"][nm]), atol=1e-14)
        assert all(np.isfinite(p).all() and (p >= 0).all() and np.allclose(p.sum(1), 1) and (p[np.arange(te.height), te["y"].to_numpy()] > 0).all() for p in P.values())
        loss = {nm: sa.losses(p, te["y"].to_numpy()) for nm, p in P.items()}
        for nm, p in P.items():
            loss[nm]["score_mse"] = (te["y"].to_numpy()/2-p[:, 2]-p[:, 1]/2)**2
        errors = []
        for group, mask in [("all", np.ones(te.height, bool)), ("novice", te["novice"].to_numpy()), ("established", ~te["novice"].to_numpy())]:
            if not SMOKE:
                rr = reported.filter(pl.col("speed") == speed, pl.col("fold") == fold, pl.col("group") == group, pl.col("model").is_in(["recent", "novice_extension"]))
                for row in rr.iter_rows(named=True):
                    nm = "interaction" if row["model"] == "novice_extension" else row["model"]
                    mean = loss[nm][row["metric"]][mask].mean()
                    diff = (loss[nm][row["metric"]]-loss[row["vs"]][row["metric"]])[mask].mean()
                    errors.extend([abs(mean-row["mean"]), abs(diff-row["diff"])])
                    assert row["n"] == int(mask.sum())
            for nm, reference in [("interaction", "recent"), ("spline", "recent"), ("spline", "interaction"), ("selected", "recent"), ("selected", "baseline")]:
                for metric in ["log_loss", "rps", "score_mse"]:
                    diff, se = difference(loss[nm][metric], loss[reference][metric], mask, te)
                    predictions.append({**key, "group": group, "model": nm, "vs": reference, "selected": selected, "metric": metric, "n": int(mask.sum()), "mean": loss[nm][metric][mask].mean(), "diff": diff, "se": se, "lo": diff-1.96*se, "hi": diff+1.96*se})
            for target in ["score", "loss", "draw", "win"]:
                observed = te["y"].to_numpy()/2 if target == "score" else (te["y"].to_numpy() == ["loss", "draw", "win"].index(target)).astype(float)
                expected = {nm: p[:, 2]+p[:, 1]/2 if target == "score" else p[:, ["loss", "draw", "win"].index(target)] for nm, p in P.items()}
                for nm, e in expected.items():
                    for binning in ["own_prediction", "baseline_prediction"]:
                        anchor = e if binning == "own_prediction" else expected["baseline"]
                        bins = np.minimum((anchor*10).astype(int), 9)
                        gaps, counts = [], []
                        for b in np.unique(bins[mask]):
                            keep = mask & (bins == b)
                            gap = observed[keep].mean()-e[keep].mean()
                            gaps.append(gap)
                            counts.append(int(keep.sum()))
                            curves.append({**key, "group": group, "model": nm, "target": target, "binning": binning, "bin": int(b), "n": int(keep.sum()), "predicted": e[keep].mean(), "observed": observed[keep].mean(), "residual": gap})
                        calibration.append({**key, "group": group, "model": nm, "target": target, "binning": binning, "n": sum(counts), "weighted_abs_gap_pp": 100*np.average(np.abs(gaps), weights=counts)})
        reconstruction_error = max(errors) if errors else 0.
        assert reconstruction_error < 1e-10
        # 比較樣條修正項的雙方反轉，不把缺歷史時僅單方有列的樣本誤當成雙方配對。
        delta = spline.design(te, final["spline"]["knots"], np.asarray(final["spline"]["center"]))
        swapped = te.with_columns(rd=pl.col("opp_rd"), opp_rd=pl.col("rd"))
        reversal_error = np.abs(delta+spline.design(swapped, final["spline"]["knots"], np.asarray(final["spline"]["center"]))).max()
        assert reversal_error < 1e-12
        checks.append({**key, "n": te.height, "selected": selected, "max_loss_error": reconstruction_error, "max_spline_reversal_error": reversal_error})
        for group, mask in [("all", np.ones(te.height, bool)), ("novice", te["novice"].to_numpy()), ("established", ~te["novice"].to_numpy())]:
            for reference in ["baseline", "recent"]:
                delta = 100*(P["selected"][:, 2]-P[reference][:, 2])[mask]
                changes.append({**key, "group": group, "model": "selected", "vs": reference, "n": int(mask.sum()), "mean_abs_win_pp": np.abs(delta).mean(), "p90_abs_win_pp": np.quantile(np.abs(delta), .9)})
        for row in example_inputs.filter(pl.col("speed") == speed, pl.col("fold") == fold).iter_rows(named=True):
            d = pl.DataFrame([row]).with_columns(
                sign=pl.lit(1.), xr=((pl.col("elo")+pl.col("opp_elo"))/2-1500)/100,
                xd=(pl.col("elo")-pl.col("opp_elo"))/100,
                gxd=ordinal.g("rd", "opp_rd")*(pl.col("elo")-pl.col("opp_elo"))/100,
                d_prev_draw=(pl.col("previous")=="draw").cast(pl.Float64)-(pl.col("opponent_previous")=="draw").cast(pl.Float64),
                d_prev_win=(pl.col("previous")=="win").cast(pl.Float64)-(pl.col("opponent_previous")=="win").cast(pl.Float64),
                opp_missing=pl.lit(0.), d_dr=pl.col("dr")-pl.col("opp_dr"),
                rd_dr=pl.col("dr")*pl.col("rd")/100-pl.col("opp_dr")*pl.col("opp_rd")/100,
                rd_abs_dr=pl.col("dr").abs()*pl.col("rd")/100-pl.col("opp_dr").abs()*pl.col("opp_rd")/100,
            )
            d = tm.offset(d, bs_refit, "Lichess")
            base_p = ordinal.probs(d["eta"].to_numpy(), d["width"].to_numpy())[0]
            assert np.allclose(base_p, [row["p_loss"], row["p_draw"], row["p_win"]], atol=1e-12)
            for nm, m in [("baseline", saved_model([], [])), ("recent", final["recent"]), ("selected", final[selected])]:
                p = spline.probabilities(d, m)[0]
                scenarios.append({**{k: row[k] for k in ["game", "speed", "fold", "group", "previous", "opponent_previous", "training_cell_n", "elo", "opp_elo", "rd", "opp_rd", "dr", "opp_dr"]}, "model": nm, "selected": selected, "p_loss": p[0], "p_draw": p[1], "p_win": p[2], "expected_score": p[2]+p[1]/2})
        exports.append(export)
        for label, rows in [("fits", fits), ("sample", samples), ("selection", selections), ("prediction", predictions), ("calibration_bins", curves), ("calibration", calibration), ("checks", checks), ("coef", coefficients), ("transforms", transforms), ("changes", changes), ("scenarios", scenarios)]:
            pl.DataFrame(rows).write_csv(f"outputs/tables/{PREFIX}_{label}.csv")
        Path(f"outputs/tables/{PREFIX}_models.json").write_text(json.dumps(exports, ensure_ascii=False, indent=2))
        print(speed, fold, "selected", selected, "elapsed", time.monotonic()-started, flush=True)
print("complete", PREFIX, "seconds", time.monotonic()-started, flush=True)
