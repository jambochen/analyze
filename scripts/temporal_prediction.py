# 完整時間外驗證與新手誤差；切分與候選固定於 docs/temporal-validation.md。
import os
import time

os.environ["JAX_PLATFORMS"] = "cpu"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import baseline_sample
import likelihood as lk
import ordinal
import state_adjusted as sa
import temporal_models as tm

SMOKE = os.environ.get("TEMPORAL_SMOKE") == "1"
PREFIX = "temporal_prediction_smoke" if SMOKE else "temporal_prediction"
T0 = np.array([0.1, 0, 0, 0.6, 0, -2.5, 0, 0, 0])
BASE = tm.BASE
started = time.monotonic()
fit_rows, sample_rows, steps, coef_rows, pred_rows, curves = [], [], [], [], [], []


def fit_model(ll, t0, free, args, d, label):
    tick = time.monotonic()
    th, _, _, _, scores, H = lk.fit(ll, t0, free, args, d["a_id"].to_numpy(), d["b_id"].to_numpy())
    z = np.max(np.abs(scores.sum(0)) / np.sqrt(np.diag(H))) if len(free) else 0.0
    assert np.isfinite(th).all() and z < 0.01, (label, z)
    fit_rows.append({"game": game, "speed": speed, "fold": fold, "fit": label, "n": d.height, "seconds": time.monotonic() - tick, "score_scaled_max": z})
    for k, value in enumerate(th):
        coef_rows.append({"game": game, "speed": speed, "fold": fold, "fit": label, "param_index": k, "est": value})
    print(game, speed, fold, label, d.height, fit_rows[-1]["seconds"], flush=True)
    return th


def base_fit(d, label):
    if game == "Lichess":
        args = tuple(d[c].to_numpy().astype(float) for c in ["xr", "xd", "gxd", "y"])
        return fit_model(baseline_sample.ll_ord, T0, np.arange(9), args, d, label)
    cols = BASE if game == "OGS" else BASE[:-1]
    return fit_model(lk.logit_ll, np.zeros(len(cols)), np.arange(len(cols)), (d.select(cols).to_numpy().astype(float), d["y"].to_numpy() / 2), d, label)


def state_ll(th, X, width, y):
    return ordinal.ll(X @ th, width, y)


def state_fit(tr, te, cols, label):
    cols = ["eta"] + cols
    Xtr, Xte = [d.select(cols).to_numpy().astype(float) for d in [tr, te]]
    t0 = np.r_[1.0, np.zeros(len(cols) - 1)]
    if game == "Lichess":
        th = fit_model(state_ll, t0, np.arange(1, len(cols)), (Xtr, tr["width"].to_numpy(), tr["y"].to_numpy()), tr, label)
    else:
        th = fit_model(lk.logit_ll, t0, np.arange(1, len(cols)), (Xtr, tr["y"].to_numpy() / 2), tr, label)
    p = ordinal.probs(Xte @ th, te["width"].to_numpy())
    assert np.isfinite(p).all() and (p >= 0).all() and np.allclose(p.sum(1), 1)
    assert (p[np.arange(te.height), te["y"].to_numpy()] > 0).all()
    return p


def difference(a, b, mask, d):
    delta = (a - b)[mask]
    se = lk.dyad_se((delta - delta.mean()) / len(delta), d["a_id"].to_numpy()[mask], d["b_id"].to_numpy()[mask])
    return float(delta.mean()), float(se)


for game in ["Lichess", "OGS", "AoE2"]:
    t, folds, state_cols = tm.sample(game)
    base = baseline_sample.sample().rename({"w": "a_id", "b": "b_id"}).with_columns(time=pl.col("start").dt.epoch("s")) if game == "Lichess" else t
    if SMOKE:
        # 依 game_id 固定抽樣，保留同局雙方，涵蓋全部時間段。
        t = t.filter(pl.col("game_id").hash(seed=42) % 40 == 0)
        base = base.filter(pl.col("game_id").hash(seed=42) % 40 == 0)
    t, base = t.sort("time", "game_id", "a_id"), base.sort("time", "game_id")
    assert t.select(pl.struct("game_id", "a_id").n_unique()).item() == t.height
    for speed in sorted(t["speed"].unique().to_list()):
        ts, bs = [d.filter(pl.col("speed") == speed) for d in [t, base]]
        for fold, inner, test_start, test_end in folds:
            cut = [int(np.datetime64(x, 's').astype(np.int64)) for x in [inner, test_start, test_end]]
            train = ts.filter(pl.col("time") < cut[1])
            fit = ts.filter(pl.col("time") < cut[0])
            valid = ts.filter(pl.col("time").is_between(cut[0], cut[1], closed="left"))
            test = ts.filter(pl.col("time").is_between(cut[1], cut[2], closed="left"))
            for label, d in [("fit", fit), ("validation", valid), ("train_refit", train), ("test", test)]:
                sample_rows.append({"game": game, "speed": speed, "fold": fold, "segment": label, "n": d.height, "games": d["game_id"].n_unique(), "players": pl.concat([d["a_id"], d["b_id"]]).n_unique(), "novice": d["novice"].sum(), "time_min": d["time"].min(), "time_max": d["time"].max()})
            assert fit["time"].max() < valid["time"].min() and train["time"].max() < test["time"].min()
            th = base_fit(bs.filter(pl.col("time") < cut[0]), "base_inner")
            f, v = tm.offset(fit, th, game), tm.offset(valid, th, game)
            current = sa.losses(ordinal.probs(v["eta"].to_numpy(), v["width"].to_numpy()), v["y"].to_numpy())["log_loss"]
            kept = []
            stages = [("recent_results", state_cols, "all"), ("rating_change", ["d_dr"], "all")]
            if game != "AoE2":
                stages += [("rd_interaction", ["d_dr", "rd_dr"], "novice"), ("direction_interaction", ["d_dr", "rd_dr", "rd_abs_dr"], "novice")]
            for label, add, group in stages:
                if label == "rd_interaction":
                    simple = kept.copy()
                candidate = list(dict.fromkeys(kept + add))
                pp = state_fit(f, v, candidate, "select_" + label)
                ls = sa.losses(pp, v["y"].to_numpy())["log_loss"]
                mask = v["novice"].to_numpy() if group == "novice" else np.ones(v.height, bool)
                diff, se = difference(ls, current, mask, v)
                accept = diff + 1.96 * se < 0
                steps.append({"game": game, "speed": speed, "fold": fold, "step": label, "group": group, "columns": ",".join(candidate), "diff": diff, "se": se, "kept": accept})
                if accept:
                    kept, current = candidate, ls
            if game == "AoE2":
                simple = kept.copy()
            th = base_fit(bs.filter(pl.col("time") < cut[1]), "base_refit")
            tr, te = tm.offset(train, th, game), tm.offset(test, th, game)
            P = {"baseline": ordinal.probs(te["eta"].to_numpy(), te["width"].to_numpy())}
            P["recent"] = state_fit(tr, te, simple, "recent_refit")
            P["novice_extension"] = state_fit(tr, te, kept, "novice_refit") if kept != simple else P["recent"]
            E = {nm: p[:, 2] + p[:, 1] / 2 for nm, p in P.items()}
            E["system"] = te["p_sys"].to_numpy()
            loss = {nm: sa.losses(p, te["y"].to_numpy()) for nm, p in P.items()}
            if game != "Lichess":
                loss["system"] = sa.losses(np.c_[1 - E["system"], np.zeros(te.height), E["system"]], te["y"].to_numpy())
            for nm in E:
                loss.setdefault(nm, {})["score_mse"] = (te["y"].to_numpy() / 2 - E[nm])**2
            for group, mask in [("all", np.ones(te.height, bool)), ("novice", te["novice"].to_numpy()), ("established", ~te["novice"].to_numpy())]:
                for nm, reference in [("baseline", "system"), ("recent", "baseline"), ("novice_extension", "recent")]:
                    for metric in loss[nm].keys() & loss[reference].keys():
                        diff, se = difference(loss[nm][metric], loss[reference][metric], mask, te)
                        pred_rows.append({"game": game, "speed": speed, "fold": fold, "group": group, "model": nm, "vs": reference, "metric": metric, "n": int(mask.sum()), "mean": loss[nm][metric][mask].mean(), "diff": diff, "se": se, "lo": diff - 1.96 * se, "hi": diff + 1.96 * se})
                for nm, expected in E.items():
                    bins = np.minimum((expected * 10).astype(int), 9)
                    for k in np.unique(bins[mask]):
                        m = mask & (bins == k)
                        residual = te["y"].to_numpy() / 2 - expected
                        mean, se = difference(residual, np.zeros(te.height), m, te)
                        curves.append({"game": game, "speed": speed, "fold": fold, "group": group, "model": nm, "bin": int(k), "n": int(m.sum()), "predicted": expected[m].mean(), "observed": (te["y"].to_numpy()[m] / 2).mean(), "residual": mean, "se": se})
            for label, rows in [("fits", fit_rows), ("sample", sample_rows), ("selection", steps), ("coef", coef_rows), ("prediction", pred_rows), ("calibration", curves)]:
                pl.DataFrame(rows).write_csv(f"outputs/tables/{PREFIX}_{label}.csv")
            print(game, speed, fold, "complete", time.monotonic() - started, flush=True)
    del t, base, ts, bs

cal = pl.DataFrame(curves)
for game in ["Lichess", "OGS", "AoE2"]:
    g = cal.filter(pl.col("game") == game, pl.col("group") == "all")
    panels = g.select("speed", "fold").unique().sort("speed", "fold").rows()
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4), squeeze=False)
    for ax, (speed, fold) in zip(axes[0], panels):
        for nm in ["system", "baseline", "recent", "novice_extension"]:
            d = g.filter(pl.col("speed") == speed, pl.col("fold") == fold, pl.col("model") == nm).sort("predicted")
            ax.plot(d["predicted"], d["observed"], marker=".", label=nm)
        ax.plot([0, 1], [0, 1], color="grey", linestyle="--")
        ax.set(xlabel="Predicted score", ylabel="Observed score", title=f"{game} {speed}\n{fold}", xlim=(0, 1), ylim=(0, 1))
    axes[0][-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"outputs/figures/{PREFIX}_{game.lower()}_calibration.png", dpi=150)
    plt.close(fig)
print(pl.DataFrame(pred_rows).filter(pl.col("metric") == "log_loss", pl.col("group") == "all"))
print("seconds", time.monotonic() - started)
