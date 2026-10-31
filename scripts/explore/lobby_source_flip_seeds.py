# 排除邀戰與無 source 的對局之後，校準與切點形式的檢定有五項 Holm 決策翻轉。
# 翻轉可能來自兩件事：子樣本重抽（母體變了，種子相同也抽到不同的局），或樣本組成。
# 新舊樣本各以同一組種子重抽，另加不抽的全樣本，分開兩者。
# 各種子內以完整的檢定族做 Holm，族與 calibration.py、threshold_rating_form.py 相同；
# fractional logit 的檢定本來就用全部老手對局，不隨種子變，每個樣本只算一次。

import os
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, "scripts")

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl

import game_source
import likelihood as lk
import novice
import ordinal

UTC = "UTC"
CLASSES = ["Bullet", "Blitz", "Classical"]
N_FIT = 1_000_000
SEEDS = list(range(1, 11))
ELO = ordinal.Q
K = 20

# 校準的樣本，照 calibration.py：累計局數含全部對局
g = (
    pl.scan_parquet("data/processed/lichess_games.parquet")
    .filter(pl.col("speed").is_in(CLASSES), pl.col("base_seconds").is_not_null(), pl.col("start") < pl.datetime(2015, 1, 1, time_zone=UTC))
    .select(
        "game_id", "start", "speed", "white_elo", "black_elo",
        arena=pl.col("tournament_id").is_not_null(),
        y=pl.col("result").replace_strict({"0-1": 0, "1/2-1/2": 1, "1-0": 2}, default=None, return_dtype=pl.Int8),
        w="white_id", b="black_id",
    )
    .with_row_index("i")
    .collect(engine="streaming")
)
nprev = pl.concat([
    pl.concat([g.filter(pl.col("speed") == tc).select("i", "start", p="w"), g.filter(pl.col("speed") == tc).select("i", "start", p="b")])
    .sort("i")
    .with_columns(n_prev=pl.int_range(pl.len(), dtype=pl.UInt32).over("p"))
    .filter(pl.col("start") >= pl.datetime(2014, 1, 1, time_zone=UTC))
    .select("i", "p", "n_prev")
    for tc in CLASSES
])
cal = (
    game_source.with_source(g.lazy())
    .filter(pl.col("start") >= pl.datetime(2014, 1, 1, time_zone=UTC), pl.col("y").is_not_null(), pl.col("white_elo").is_not_null(), pl.col("black_elo").is_not_null())
    .join(nprev.lazy().rename({"p": "w", "n_prev": "nw"}), on=["i", "w"])
    .join(nprev.lazy().rename({"p": "b", "n_prev": "nb"}), on=["i", "b"])
    .filter(pl.min_horizontal("nw", "nb") >= novice.GAMES)
    .sort("i")
    .with_columns(
        diff=(pl.col("white_elo") - pl.col("black_elo")).cast(pl.Float64),
        xr=((pl.col("white_elo") + pl.col("black_elo")) / 2 - 1500) / 100,
        keep=game_source.KEEP.fill_null(False),
    )
    .with_columns(xd=pl.col("diff") / 100, score=pl.col("y") / 2, E=1 / (1 + 10 ** (-pl.col("diff") / 400)))
    .select("speed", "w", "b", "y", "score", "xr", "xd", "E", "keep")
    .collect(engine="streaming")
)
del g, nprev

# 切點形式的樣本，照 threshold_rating_form.py
thr = (
    pl.scan_parquet("data/processed/lichess_games.parquet")
    .filter(
        pl.col("speed").is_in(CLASSES), pl.col("base_seconds").is_not_null(),
        pl.col("start") >= pl.datetime(2014, 1, 1, time_zone=UTC), pl.col("start") < pl.datetime(2015, 1, 1, time_zone=UTC),
        pl.col("white_elo").is_not_null(), pl.col("black_elo").is_not_null(),
    )
    .select(
        "game_id", "speed", "white_elo", "black_elo",
        y=pl.col("result").replace_strict({"0-1": 0, "1/2-1/2": 1, "1-0": 2}, default=None, return_dtype=pl.Int8),
        w="white_id", b="black_id", arena=pl.col("tournament_id").is_not_null(),
    )
    .filter(pl.col("y").is_not_null())
    .pipe(game_source.with_source)
    .join(pl.scan_parquet("data/processed/glicko_replay.parquet").select("game_id", "perf", "white_rd", "black_rd"), on="game_id")
    .filter(pl.col("perf") == pl.col("speed"))
    .with_columns(
        xr=((pl.col("white_elo") + pl.col("black_elo")) / 2 - 1500) / 100,
        xd=(pl.col("white_elo") - pl.col("black_elo")).cast(pl.Float64) / 100,
        gxd=ordinal.g("white_rd", "black_rd") * (pl.col("white_elo") - pl.col("black_elo")).cast(pl.Float64) / 100,
        keep=game_source.KEEP.fill_null(False),
    )
    .sort("game_id")
    .select("speed", "w", "b", "y", "xr", "xd", "gxd", "keep")
    .collect(engine="streaming")
)
print(cal.group_by("speed", "keep").len().sort(pl.all()), thr.group_by("speed", "keep").len().sort(pl.all()))


def ll_ord(t, xr, xd, y):
    eta = t[0] + t[1] * xr + t[2] * xr**2 + (t[3] + t[4] * xr) * xd
    return ordinal.ll(eta, jnp.exp(t[5] + t[6] * xr + t[7] * xr**2), y)


def ll_frac(t, xr, L, s):
    z = t[0] + t[1] * xr + (t[2] + t[3] * xr) * L
    return s * jax.nn.log_sigmoid(z) + (1 - s) * jax.nn.log_sigmoid(-z)


def ll_ppo(t, xr, xd, y):
    eta = t[0] + t[1] * xr + t[2] * xr**2 + (t[3] + t[4] * xr) * xd
    return ordinal.ll(eta, jnp.exp(t[5] + t[6] * xr + t[7] * xr**2), y, shift=(t[8] + t[9] * xr) * xd)


def ll_thr(t, Z, xr, xd, gxd, y):
    p = Z.shape[1]
    eta = Z @ t[3 : 3 + p] + (t[0] + t[1] * xr) * xd + t[2] * gxd
    return ordinal.ll(eta, jnp.exp(Z @ t[3 + p :]), y)


T0 = [0.1, 0, 0, 0.6, 0, -2.5, 0, 0]
IDX = np.arange(8)
rows = []
for sample in ["new", "old"]:
    c_s = cal if sample == "old" else cal.filter("keep")
    t_s = thr if sample == "old" else thr.filter("keep")
    # fractional logit 的三項：全部老手對局，與種子無關
    frac = []
    for tc in CLASSES:
        mf = c_s.filter(pl.col("speed") == tc)
        L = np.log(mf["E"].to_numpy() / (1 - mf["E"].to_numpy()))
        af = (mf["xr"].to_numpy(), L, mf["score"].to_numpy())
        wf, bf = mf["w"].to_numpy(), mf["b"].to_numpy()
        tf, llf, Vf, Hf, *_ = lk.fit(ll_frac, [0, 0, 1, 0], np.arange(4), af, wf, bf)
        for name, idx, val in [("elo_mean_calibrated", [0, 1, 2, 3], [0, 0, 1, 0]), ("no_white_adv_in_mean", [0, 1], [0, 0]), ("elo_slope_in_mean", [2, 3], [1, 0])]:
            t0f = np.array([0, 0, 1, 0], dtype=float)
            t0f[idx] = val
            _, ll0f, *_ = lk.fit(ll_frac, t0f, np.setdiff1d(np.arange(4), idx), af, wf, bf)
            frac.append({"speed": tc, "test": name, "p": lk.adjusted_lrt(2 * (llf - ll0f), lk.lr_weights(Hf, Vf, idx))["p"]})
    for seed in SEEDS + ["full"]:
        fam = list(frac)
        for tc in CLASSES:
            m = c_s.filter(pl.col("speed") == tc)
            m = m if seed == "full" else m.sample(min(N_FIT, m.height), seed=seed)
            w, b = m["w"].to_numpy(), m["b"].to_numpy()
            xr, xd, y = (m[c].to_numpy().astype(np.float64) for c in ["xr", "xd", "y"])
            th, ll1, V, Hi, *_ = lk.fit(ll_ord, T0, IDX, (xr, xd, y), w, b)
            for name, idx, val in [
                ("white_adv_constant_in_rating", [1, 2], [0, 0]),
                ("draw_width_constant_in_rating", [6, 7], [0, 0]),
                ("slope_constant_in_rating", [4], [0]),
                ("all_rating_terms_zero", [1, 2, 4, 6, 7], [0, 0, 0, 0, 0]),
                ("naive_elo_slope", [3], [100 * ELO]),
            ]:
                t0 = np.array(T0, dtype=float)
                t0[idx] = val
                _, ll0, *_ = lk.fit(ll_ord, t0, np.setdiff1d(IDX, idx), (xr, xd, y), w, b)
                fam.append({"speed": tc, "test": name, "p": lk.adjusted_lrt(2 * (ll1 - ll0), lk.lr_weights(Hi, V, idx))["p"]})
            tp, llp, Vp, Hip, *_ = lk.fit(ll_ppo, T0 + [0, 0], np.arange(10), (xr, xd, y), w, b)
            fam.append({"speed": tc, "test": "proportional_odds_in_diff", "p": lk.adjusted_lrt(2 * (llp - ll1), lk.lr_weights(Hip, Vp, [8, 9]))["p"]})
        f = pl.DataFrame(fam)
        rows.append(f.with_columns(p_holm=lk.holm(f["p"]), family=pl.lit("calibration"), sample=pl.lit(sample), seed=pl.lit(str(seed))))

        fam = []
        for tc in CLASSES:
            m = t_s.filter(pl.col("speed") == tc)
            m = m if seed == "full" else m.sample(min(N_FIT, m.height), seed=seed)
            w, b = m["w"].to_numpy(), m["b"].to_numpy()
            xr, xd, gxd, y = (m[c].to_numpy().astype(np.float64) for c in ["xr", "xd", "gxd", "y"])
            edges = np.quantile(xr, np.linspace(0, 1, K + 1)[1:-1])
            D = np.eye(K)[np.searchsorted(edges, xr, side="right")]
            Z = np.column_stack([np.ones_like(xr), xr, xr**2, D[:, 1:]])
            del D
            p = Z.shape[1]
            A, W = 3 + np.arange(p), 3 + p + np.arange(p)
            t0 = np.r_[0.6, 0, 0, 0.1, np.zeros(p - 1), -2.5, np.zeros(p - 1)]
            allf = np.arange(len(t0))
            fits = {
                "quad_plus_bins": allf,
                "quad": np.setdiff1d(allf, np.r_[A[3:], W[3:]]),
                "a_linear": np.setdiff1d(allf, np.r_[A[2:], W[3:]]),
                "w_linear": np.setdiff1d(allf, np.r_[A[3:], W[2:]]),
                "a_quad_w_bins": np.setdiff1d(allf, A[3:]),
                "w_quad_a_bins": np.setdiff1d(allf, W[3:]),
            }
            res = {nm: lk.fit(ll_thr, t0, fr, (Z, xr, xd, gxd, y), w, b) for nm, fr in fits.items()}
            for name, full, rest in [("a_quad_term", "quad", "a_linear"), ("w_quad_term", "quad", "w_linear"), ("a_bins_beyond_quad", "quad_plus_bins", "a_quad_w_bins"), ("w_bins_beyond_quad", "quad_plus_bins", "w_quad_a_bins")]:
                _, l1, V1, Hi1, *_ = res[full]
                idx = np.flatnonzero(~np.isin(fits[full], fits[rest]))
                fam.append({"speed": tc, "test": name, "p": lk.adjusted_lrt(2 * (l1 - res[rest][1]), lk.lr_weights(Hi1, V1, idx))["p"]})
            del Z, res
        f = pl.DataFrame(fam)
        rows.append(f.with_columns(p_holm=lk.holm(f["p"]), family=pl.lit("threshold_form"), sample=pl.lit(sample), seed=pl.lit(str(seed))))
        print(sample, seed, flush=True)

out = pl.concat(rows)
out.write_csv("outputs/tables/lobby_source_flip_seeds.csv")
FOCUS = [("Bullet", "white_adv_constant_in_rating"), ("Classical", "slope_constant_in_rating"), ("Bullet", "w_bins_beyond_quad"), ("Blitz", "w_quad_term"), ("Blitz", "w_bins_beyond_quad")]
foc = out.filter(pl.struct("speed", "test").is_in([{"speed": s, "test": t} for s, t in FOCUS]))
summary = (
    foc.filter(pl.col("seed") != "full")
    .group_by("speed", "test", "sample")
    .agg(share_sig=(pl.col("p_holm") < 0.05).mean(), p_holm_min=pl.col("p_holm").min(), p_holm_median=pl.col("p_holm").median(), p_holm_max=pl.col("p_holm").max())
    .join(foc.filter(pl.col("seed") == "full").select("speed", "test", "sample", p_holm_full="p_holm"), on=["speed", "test", "sample"])
    .sort("test", "speed", "sample")
)
summary.write_csv("outputs/tables/lobby_source_flip_seeds_summary.csv")
print(summary)
