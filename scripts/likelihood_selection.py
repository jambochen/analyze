# 同樣本聯合估計；CL-BIC（唯一對局數）主選、CL-AIC 對照，巢狀 LRT 只解釋增項證據。
# 不經 LRT 或驗證 NLL 篩選候選；玩家數與玩家局數的 BIC 另作尺度敏感度。
import os
import json
import time
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import polars as pl

import likelihood as lk
import ordinal
import rd_spline as spline
import state_adjusted as sa
import temporal_models as tm

SMOKE = os.environ.get("LIKELIHOOD_SMOKE") == "1"
PREFIX = "likelihood_selection_smoke" if SMOKE else "likelihood_selection"
coef = pl.read_csv("outputs/tables/temporal_prediction_coef.csv").filter(pl.col("game") == "Lichess")
old_samples = pl.read_csv("outputs/tables/temporal_prediction_sample.csv").filter(pl.col("game") == "Lichess")
example_inputs = pl.read_csv("outputs/tables/prediction_value_scenarios.csv").filter(pl.col("game") == "Lichess", pl.col("model") == "baseline")
fits, tests, selections, samples, predictions, curves, calibration, changes, scenarios, exports = [], [], [], [], [], [], [], [], [], []
started = time.monotonic()


def fit_model(d, columns, name, use_spline=False):
    tick = time.monotonic()
    knots, center = [], []
    extra = np.zeros((d.height, 0))
    if use_spline:
        pooled = np.r_[d["rd"].to_numpy(), d["opp_rd"].to_numpy()]
        knots = np.quantile(np.unique(pooled), np.linspace(0, 1, 5)).tolist()
        center = spline.basis(pooled, knots).mean(0).tolist()
        extra = spline.design(d, knots, np.asarray(center))
    xr, xd, gx, sign = [d[c].to_numpy() for c in ["xr", "xd", "gxd", "sign"]]
    X = np.column_stack([sign, sign*xr, sign*xr**2, xd, xr*xd, gx, *[d[c].to_numpy() for c in columns], extra])
    Z = np.column_stack([np.ones(d.height), xr, xr**2])
    k = X.shape[1]+Z.shape[1]
    t0 = np.r_[beta[[0, 1, 2, 3, 4, 8]], np.zeros(k-9), beta[[5, 6, 7]]]
    th, ll, V, Hi, score, H = lk.fit(tm.joint_ll, t0, np.arange(k), (X, Z, d["y"].to_numpy()), d["a_id"].to_numpy(), d["b_id"].to_numpy())
    scaled = float(np.max(np.abs(score.sum(0))/np.sqrt(np.diag(H))))
    assert scaled < .01 and np.isfinite(ll) and np.isfinite(th).all()
    assert np.linalg.eigvalsh(H).min() > 0 and np.linalg.eigvalsh(V).min() > 0, (name, np.linalg.eigvalsh(H), np.linalg.eigvalsh(V))
    effective = float(np.trace(H @ V))
    players = pl.concat([d["a_id"], d["b_id"]]).n_unique()
    info = {**key, "model": name, "n": d.height, "games": d["game_id"].n_unique(), "players": players, "k": k, "effective_k": effective, "nll": -ll, "mean_nll": -ll/d.height, "aic_naive": -2*ll+2*k, "bic_naive_rows": -2*ll+np.log(d.height)*k, "cl_aic": -2*ll+2*effective, "cl_bic_players": -2*ll+np.log(players)*effective, "cl_bic_games": -2*ll+np.log(d["game_id"].n_unique())*effective, "cl_bic_rows": -2*ll+np.log(d.height)*effective, "score_scaled_max": scaled, "seconds": time.monotonic()-tick}
    fits.append(info)
    model = {"baseline_coef": th[[0, 1, 2, 3, 4, k-3, k-2, k-1, 5]].tolist(), "columns": columns, "coef": th[6:6+len(columns)].tolist(), "knots": knots, "center": center, "spline_coef": th[6+len(columns):-3].tolist()}
    prob = spline.probabilities(tm.offset(d, model["baseline_coef"], "Lichess"), model)
    assert np.isclose(np.log(prob[np.arange(d.height), d["y"].to_numpy()]).sum(), ll, atol=1e-7, rtol=1e-10)
    print(key, name, info, flush=True)
    return {"name": name, "model": model, "info": info, "ll": ll, "V": V, "Hi": Hi, "parameters": ["white", "white_xr", "white_xr2", "xd", "xdxr", "gxd"]+columns+[f"rd_spline_{i}" for i in range(extra.shape[1])]+["width", "width_xr", "width_xr2"]}


def nested(restricted, full, step):
    assert set(restricted["parameters"]) < set(full["parameters"])
    idx = [i for i, name in enumerate(full["parameters"]) if name not in restricted["parameters"]]
    lam = lk.lr_weights(full["Hi"], full["V"], idx)
    lr = 2*(full["ll"]-restricted["ll"])
    assert lr >= -1e-7 and (lam > 0).all()
    out = lk.adjusted_lrt(lr, lam) if lr > 1e-10 else {"lr": max(0., lr), "df": len(lam), "p_naive": 1., "p": 1., "p_method": "zero", "p_upper": 1., "p_satt": 1., "lam_min": lam.min(), "lam_max": lam.max()}
    assert 0 <= out["p"] <= 1
    tests.append({**key, "step": step, "restricted": restricted["name"], "full": full["name"], **out, "reject_raw_005": out["p"] < .05})


def difference(a, b, mask, d):
    delta = (a-b)[mask]
    se = lk.dyad_se((delta-delta.mean())/len(delta), d["a_id"].to_numpy()[mask], d["b_id"].to_numpy()[mask])
    assert np.isfinite(delta).all() and np.isfinite(se)
    return float(delta.mean()), float(se)


t, folds, _ = tm.sample("Lichess")
if SMOKE:
    t = t.filter(pl.col("game_id").hash(seed=42) % 10 == 0)
t = t.sort("time", "game_id", "a_id")
assert t.select(pl.struct("game_id", "a_id").n_unique()).item() == t.height
for speed in ["Bullet", "Blitz", "Classical"]:
    ts = t.filter(pl.col("speed") == speed)
    for fold, _, start, end in folds:
        key = {"speed": speed, "fold": fold}
        mid, hi = [int(np.datetime64(x, "s").astype(np.int64)) for x in [start, end]]
        raw = {"train_refit": ts.filter(pl.col("time") < mid), "test": ts.filter(pl.col("time").is_between(mid, hi, closed="left"))}
        assert raw["train_refit"]["time"].max() < raw["test"]["time"].min()
        for segment, d in raw.items():
            row = {**key, "segment": segment, "n": d.height, "games": d["game_id"].n_unique(), "novice": d["novice"].sum(), "time_min": d["time"].min(), "time_max": d["time"].max()}
            samples.append(row)
            if not SMOKE:
                old = old_samples.filter(pl.col("speed") == speed, pl.col("fold") == fold, pl.col("segment") == segment).row(0, named=True)
                assert all(row[c] == old[c] for c in ["n", "games", "novice", "time_min", "time_max"])
        beta = coef.filter(pl.col("speed") == speed, pl.col("fold") == fold, pl.col("fit") == "base_refit").sort("param_index")["est"].to_numpy()
        tr = raw["train_refit"]
        results = ["d_prev_draw", "d_prev_win", "opp_missing"]
        specs = {"baseline": [], "results": results, "rating_change": ["d_dr"], "recent": results+["d_dr"], "rd_change": ["d_dr", "rd_dr"], "interaction": results+["d_dr", "rd_dr"], "rd_direction": ["d_dr", "rd_dr", "rd_abs_dr"], "asymmetric": results+["d_dr", "rd_dr", "rd_abs_dr"]}
        if speed != "Classical":
            specs |= {"spline_baseline": [], "spline_results": results, "spline_change": ["d_dr"], "spline_recent": results+["d_dr"]}
        candidates = {nm: fit_model(tr, cols, nm, nm.startswith("spline")) for nm, cols in specs.items()}
        for small, large in [("baseline", "results"), ("baseline", "rating_change"), ("results", "recent"), ("rating_change", "recent"), ("rating_change", "rd_change"), ("recent", "interaction"), ("recent", "asymmetric"), ("rd_change", "rd_direction"), ("interaction", "asymmetric"), ("rd_change", "interaction"), ("rd_direction", "asymmetric")]:
            nested(candidates[small], candidates[large], small+"_to_"+large)
        if speed != "Classical":
            for small, large in [("baseline", "spline_baseline"), ("results", "spline_results"), ("rating_change", "spline_change"), ("recent", "spline_recent"), ("spline_baseline", "spline_results"), ("spline_change", "spline_recent"), ("spline_results", "spline_recent")]:
                nested(candidates[small], candidates[large], small+"_to_"+large)
        criteria = ["cl_aic", "cl_bic_players", "cl_bic_games", "cl_bic_rows", "aic_naive", "bic_naive_rows"]
        choices = {criterion: min(candidates, key=lambda nm: candidates[nm]["info"][criterion]) for criterion in criteria}
        selected, aic = choices["cl_bic_games"], choices["cl_aic"]
        recent_name = min(["baseline", "results", "rating_change", "recent"], key=lambda nm: candidates[nm]["info"]["cl_bic_games"])
        interaction_name = min(["rd_change", "interaction", "rd_direction", "asymmetric"], key=lambda nm: candidates[nm]["info"]["cl_bic_games"])
        aliases = {"recent": recent_name, "interaction": interaction_name, "selected": selected, "aic": aic}
        if speed != "Classical":
            aliases["spline"] = min([nm for nm in specs if nm.startswith("spline")], key=lambda nm: candidates[nm]["info"]["cl_bic_games"])
        for nm, fit in candidates.items():
            selections.append({**key, "model": nm, "selected_bic": nm == selected, "selected_aic": nm == aic, "selected_bic_players": nm == choices["cl_bic_players"], "selected_bic_rows": nm == choices["cl_bic_rows"], **{c: fit["info"][c] for c in ["nll", "k", "effective_k", *criteria]}, **{"delta_"+c: fit["info"][c]-candidates[choices[c]]["info"][c] for c in criteria}})
        export = {**key, "selected": selected, "selected_aic": aic, "selected_bic_players": choices["cl_bic_players"], "selected_bic_rows": choices["cl_bic_rows"], "selection_rule": "minimum_cl_bic_unique_games", "estimation": "joint_arena_baseline_and_recent", "train_end_exclusive": start, "test_end_exclusive": end, "aliases": aliases, "models": {nm: fit["model"] for nm, fit in candidates.items()}}
        exports.append(export)
        # 全候選依訓練資訊準則選完，才計算測試預測；LRT 不作額外去留門檻。
        te = raw["test"]
        raw_prob = {nm: spline.probabilities(tm.offset(te, model["baseline_coef"], "Lichess"), model) for nm, model in export["models"].items()}
        P = {"baseline": raw_prob["baseline"], **{nm: raw_prob[target] for nm, target in aliases.items()}}
        assert all(np.isfinite(p).all() and (p >= 0).all() and np.allclose(p.sum(1), 1) and (p[np.arange(te.height), te["y"].to_numpy()] > 0).all() for p in P.values())
        restored = json.loads(json.dumps(export))
        m = restored["models"][selected]
        assert np.allclose(P["selected"], spline.probabilities(tm.offset(te, m["baseline_coef"], "Lichess"), m), atol=1e-14)
        loss = {nm: sa.losses(p, te["y"].to_numpy()) for nm, p in P.items()}
        for nm, p in P.items():
            loss[nm]["score_mse"] = (te["y"].to_numpy()/2-p[:, 2]-p[:, 1]/2)**2
        pairs = [("recent", "baseline"), ("interaction", "recent"), ("selected", "recent"), ("selected", "baseline"), ("aic", "baseline"), ("aic", "selected")]
        if "spline" in P:
            pairs += [("spline", "recent"), ("spline", "interaction")]
        for group, mask in [("all", np.ones(te.height, bool)), ("novice", te["novice"].to_numpy()), ("established", ~te["novice"].to_numpy())]:
            for nm, reference in pairs:
                for metric in ["log_loss", "rps", "score_mse"]:
                    diff, se = difference(loss[nm][metric], loss[reference][metric], mask, te)
                    predictions.append({**key, "group": group, "model": nm, "vs": reference, "selected": selected, "metric": metric, "n": int(mask.sum()), "mean": loss[nm][metric][mask].mean(), "diff": diff, "se": se, "lo": diff-1.96*se, "hi": diff+1.96*se})
            delta = 100*(P["selected"][:, 2]-P["baseline"][:, 2])[mask]
            changes.append({**key, "group": group, "model": "selected", "vs": "baseline", "n": int(mask.sum()), "mean_abs_win_pp": np.abs(delta).mean(), "p90_abs_win_pp": np.quantile(np.abs(delta), .9)})
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
            for nm, model in [("baseline", candidates["baseline"]["model"]), ("recent", candidates[recent_name]["model"]), ("selected", candidates[selected]["model"])]:
                p = spline.probabilities(tm.offset(d, model["baseline_coef"], "Lichess"), model)[0]
                scenarios.append({**{c: row[c] for c in ["game", "speed", "fold", "group", "previous", "opponent_previous", "training_cell_n", "elo", "opp_elo", "rd", "opp_rd", "dr", "opp_dr"]}, "model": nm, "selected": selected, "p_loss": p[0], "p_draw": p[1], "p_win": p[2], "expected_score": p[2]+p[1]/2})
        for label, rows in [("fits", fits), ("lrt", tests), ("selection", selections), ("sample", samples), ("prediction", predictions), ("changes", changes), ("calibration_bins", curves), ("calibration", calibration), ("scenarios", scenarios)]:
            out = pl.DataFrame(rows)
            if label == "lrt":
                out = pl.concat([g.with_columns(p_holm=pl.Series(lk.holm(g["p"].to_numpy()))) for g in out.partition_by("speed", "fold")]).with_columns(reject_holm_005=pl.col("p_holm") < .05)
            out.write_csv(f"outputs/tables/{PREFIX}_{label}.csv")
        Path(f"outputs/tables/{PREFIX}_models.json").write_text(json.dumps(exports, ensure_ascii=False, indent=2))
        print(key, "selected", selected, "aic", aic, "elapsed", time.monotonic()-started, flush=True)
print("complete", PREFIX, "seconds", time.monotonic()-started, flush=True)
