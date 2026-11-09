# 首局／開局間隔的時間外增量，及限制適用範圍的覆蓋代價。
import os
import time
os.environ["JAX_PLATFORMS"] = "cpu"
import jax.numpy as jnp
import numpy as np
import polars as pl

import arena_batches as ab
import likelihood as lk
import novice
import ordinal
import temporal_models as tm


def ll(t, eta, width, X, y):
    return ordinal.ll(eta + X @ t[:-1], width * jnp.exp(t[-1]), y)


started = time.monotonic()
smoke = os.environ.get("ENTRY_SMOKE") == "1"
coef = pl.read_csv("outputs/tables/temporal_prediction_coef.csv")
state = pl.read_csv("outputs/tables/temporal_prediction_state_coef.csv")
grp = ["player", "tournament_id"]
base = ab.player_sample_rd("2014").sort(*grp, "start", "game_id").with_columns(
    prev_result=pl.col("result").shift().over(grp),
    dr=((pl.col("elo") - pl.col("elo").shift().over(grp)) / 100).fill_null(0.),
    own_first=pl.col("games_in_tour") == 0,
    gap_bin=pl.when(pl.col("prev_gap").is_null()).then(5).when(pl.col("prev_gap") <= 300).then(0).when(pl.col("prev_gap") <= 1800).then(1).when(pl.col("prev_gap") <= 21600).then(2).when(pl.col("prev_gap") <= 86400).then(3).otherwise(4),
)
opp = base.select("game_id", opponent="player", opp_prev="prev_result", opp_dr="dr", opp_first="own_first", opp_gap="prev_gap", opp_bin="gap_bin")
base = base.filter(pl.col("white") == 1, pl.col("speed").is_in(["Bullet", "Blitz"])).join(opp, on=["game_id", "opponent"]).with_columns(
    time=pl.col("start").dt.epoch("s"), y=(pl.col("score") * 2).cast(pl.Int64),
    d_prev_draw=(pl.col("prev_result") == "draw").fill_null(False).cast(pl.Float64) - (pl.col("opp_prev") == "draw").fill_null(False).cast(pl.Float64),
    d_prev_win=(pl.col("prev_result") == "win").fill_null(False).cast(pl.Float64) - (pl.col("opp_prev") == "win").fill_null(False).cast(pl.Float64),
    opp_missing=pl.col("opp_prev").is_null().cast(pl.Float64) - pl.col("prev_result").is_null().cast(pl.Float64),
    d_dr=pl.col("dr") - pl.col("opp_dr"),
    first_diff=pl.col("own_first").cast(pl.Float64) - pl.col("opp_first").cast(pl.Float64),
    *[((pl.col("gap_bin") == k).cast(pl.Float64) - (pl.col("opp_bin") == k).cast(pl.Float64)).alias(f"gap{k}") for k in range(1, 6)],
    any_first=pl.col("own_first") | pl.col("opp_first"),
    any_long=(pl.col("prev_gap") > 21600).fill_null(False) | (pl.col("opp_gap") > 21600).fill_null(False),
    any_missing=pl.col("prev_gap").is_null() | pl.col("opp_gap").is_null(),
    high_rd=novice.novice_rd("rd", "opp_rd"), sign=pl.lit(1.), const=pl.lit(1.),
    xr=((pl.col("elo") + pl.col("opp_elo")) / 2 - 1500) / 100,
    xd=(pl.col("elo") - pl.col("opp_elo")) / 100,
    gxd=ordinal.g("rd", "opp_rd") * (pl.col("elo") - pl.col("opp_elo")) / 100,
).sort("time", "game_id")
assert base["game_id"].n_unique() == base.height
print("loaded", base.height, "seconds", time.monotonic() - started, flush=True)
cols = ["const", "eta", "first_diff", "gap1", "gap2", "gap3", "gap4", "gap5"]
models = {"recalibrated": [0, 1, 8], "first": [0, 1, 2, 8], "interval": [0, 1, 3, 4, 5, 6, 7, 8], "combined": list(range(9))}
fits, parameters, scores, calibration, coverage, samples = [], [], [], [], [], []
folds = [("Jul-Sep", "2014-07-01", "2014-10-01"), ("Oct-Dec", "2014-10-01", "2015-01-01")]
for speed in (["Bullet"] if smoke else ["Bullet", "Blitz"]):
    for fold, start, end in (folds[:1] if smoke else folds):
        lo, hi = [int(np.datetime64(x, "s").astype(np.int64)) for x in [start, end]]
        th = coef.filter(pl.col("game") == "Lichess", pl.col("speed") == speed, pl.col("fold") == fold, pl.col("fit") == "base_refit").sort("param_index")["est"].to_numpy()
        d = tm.offset(base.filter(pl.col("speed") == speed, pl.col("time") < hi), th, "Lichess")
        eta = d["eta"].to_numpy().copy()
        bc = state.filter(pl.col("game") == "Lichess", pl.col("speed") == speed, pl.col("fold") == fold, pl.col("model") == "recent")
        for name, value in bc.select("param", "est").iter_rows():
            if name != "baseline_offset":
                eta += d[name].to_numpy() * value
        d = d.with_columns(eta=pl.Series(eta))
        train, test = d.filter(pl.col("time") < lo), d.filter(pl.col("time") >= lo)
        if smoke:
            train = train.gather_every(10)
            test = test.gather_every(5)
        assert train["time"].max() < lo <= test["time"].min() and test["time"].max() < hi
        samples.append(dict(speed=speed, fold=fold, train=train.height, test=test.height, train_max_time=train["time"].max(), test_min_time=test["time"].min(), test_max_time=test["time"].max()))
        Xtr, Xte = train.select(cols).to_numpy(), test.select(cols).to_numpy()
        args = (train["eta"].to_numpy(), train["width"].to_numpy(), Xtr, train["y"].to_numpy())
        P = {"recent": ordinal.probs(test["eta"].to_numpy(), test["width"].to_numpy())}
        for model, free in models.items():
            tick = time.monotonic()
            assert np.linalg.matrix_rank(Xtr[:, np.array(free[:-1])]) == len(free) - 1
            t, logl, V, Hi, S, H = lk.fit(ll, np.zeros(9), np.array(free), args, train["player"].to_numpy(), train["opponent"].to_numpy())
            gradient = np.max(abs(S.sum(axis=0)) / np.sqrt(np.diag(H)))
            assert gradient < 1e-3
            p = ordinal.probs(test["eta"].to_numpy() + Xte @ t[:-1], test["width"].to_numpy() * np.exp(t[-1]))
            assert np.isfinite(p).all() and (p > 0).all() and np.allclose(p.sum(axis=1), 1)
            P[model] = p
            fits.append(dict(speed=speed, fold=fold, model=model, log_likelihood=logl, max_scaled_score=gradient, seconds=time.monotonic() - tick))
            parameters.extend(dict(speed=speed, fold=fold, model=model, param=name, value=value) for name, value in zip(cols + ["log_width"], t))
            print(speed, fold, model, "seconds", time.monotonic() - tick, flush=True)
        ranges = np.quantile(train["xr"].to_numpy(), [.01, .99])
        gap_limit = np.quantile(abs(train["xd"].to_numpy()), .99)
        supported = test["xr"].to_numpy() >= ranges[0]
        supported &= (test["xr"].to_numpy() <= ranges[1]) & (abs(test["xd"].to_numpy()) <= gap_limit)
        low_rd = ~test["high_rd"].to_numpy()
        groups = {"all": np.ones(test.height, bool), "any_first": test["any_first"].to_numpy(), "neither_first": ~test["any_first"].to_numpy(), "any_long_gap": test["any_long"].to_numpy(), "known_interval": ~test["any_missing"].to_numpy(), "missing_interval": test["any_missing"].to_numpy(), "low_rd": low_rd, "supported": supported, "low_rd_supported": low_rd & supported}
        y = test["y"].to_numpy()
        expected = {name: p[:, 2] + .5 * p[:, 1] for name, p in P.items()}
        losses = {name: -np.log(p[np.arange(test.height), y]) for name, p in P.items()}
        common_bin = np.minimum((expected["recalibrated"] * 10).astype(int), 9)
        for group, mask in groups.items():
            n = int(mask.sum())
            a, b = test["player"].to_numpy()[mask], test["opponent"].to_numpy()[mask]
            ids, inverse, counts = np.unique(np.r_[a, b], return_inverse=True, return_counts=True)
            pw = (1 / counts[inverse[:n]] + 1 / counts[inverse[n:]]) / 2
            for target in ["any_first", "high_rd", "any_missing"]:
                flags = test[target].to_numpy()
                coverage.append(dict(speed=speed, fold=fold, group=group, target=target, n=n, coverage=n / test.height, target_total=int(flags.sum()), retained_target=int((mask & flags).sum()), target_retention=(mask & flags).sum() / flags.sum(), mean_rating_lo=ranges[0] * 100 + 1500, mean_rating_hi=ranges[1] * 100 + 1500, abs_gap_hi=gap_limit * 100))
            for weighting, weight in [("games", np.ones(n)), ("players", pw)]:
                weight = weight / weight.sum()
                for model, reference in [("recalibrated", "recent"), ("first", "recalibrated"), ("interval", "recalibrated"), ("combined", "recalibrated"), ("first", "recent"), ("interval", "recent"), ("combined", "recent")]:
                    for metric, values, previous in [("log_loss", losses[model], losses[reference]), ("score_mse", (y / 2 - expected[model])**2, (y / 2 - expected[reference])**2)]:
                        delta = (values - previous)[mask]
                        mean = weight @ delta
                        se = lk.dyad_se((weight * (delta - mean))[:, None], a, b)
                        scores.append(dict(speed=speed, fold=fold, group=group, weighting=weighting, model=model, reference=reference, metric=metric, n=n, mean=weight @ values[mask], diff=mean, lo=mean - 1.96 * se, hi=mean + 1.96 * se))
                for model, e in expected.items():
                    residual = y[mask] / 2 - e[mask]
                    bins = common_bin[mask]
                    gap = sum(abs(np.sum(weight[bins == k] * residual[bins == k])) for k in np.unique(bins))
                    calibration.append(dict(speed=speed, fold=fold, group=group, weighting=weighting, model=model, n=n, mean_residual=weight @ residual, common_bin_abs_gap=gap))
        print(speed, fold, "complete", time.monotonic() - started, flush=True)

out = "outputs/tables/entry_context_smoke" if smoke else "outputs/tables/entry_context"
for name, rows in [("fits", fits), ("parameters", parameters), ("scores", scores), ("calibration", calibration), ("coverage", coverage), ("sample", samples)]:
    pl.DataFrame(rows).write_csv(out + "_" + name + ".csv")
print("seconds", time.monotonic() - started, flush=True)
