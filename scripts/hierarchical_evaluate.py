# Posterior predictive 平均；新玩家用 Normal 積分，不將零 random effect 當成其預測。
import os
os.environ["JAX_PLATFORMS"] = "cpu"
import json
from pathlib import Path
import numpy as np
import polars as pl
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from hierarchical_predictive import chain_probabilities, calibration_rows, game_rows
import hierarchical as hr
import likelihood as lk
import state_adjusted as sa
import temporal_models as tm
import rd_spline as spline

speed, fold, kind = [os.environ[k] for k in ["HIERARCHICAL_SPEED", "HIERARCHICAL_FOLD", "HIERARCHICAL_KIND"]]
prefix = f"hierarchical_{speed}_{fold}_{kind}"
fit = np.load(f"data/processed/{prefix}.npz")
metadata = json.loads(Path(f"outputs/tables/{prefix}_fit.json").read_text())
diagnostic = pl.read_csv(f"outputs/tables/{prefix}_diagnostics.csv")
assert np.isfinite(diagnostic.select("r_hat", "ess_bulk").to_numpy()).all()
assert metadata["divergences"] == 0
assert diagnostic["r_hat"].max() < 1.01, diagnostic["r_hat"].max()
assert diagnostic["ess_bulk"].min() > 400, diagnostic["ess_bulk"].min()
start, end = {"Jul-Sep": (1404172800, 1412121600), "Oct-Dec": (1412121600, 1420070400)}[fold]
d = (pl.read_parquet("data/processed/hierarchical_prediction.parquet")
     .filter(pl.col("speed") == speed, pl.col("time").is_between(start, end, closed="left")))
expected = (pl.read_csv("outputs/tables/likelihood_selection_sample.csv")
            .filter(pl.col("speed") == speed, pl.col("fold") == fold, pl.col("segment") == "test").row(0, named=True))
assert all([d.height == expected["n"], d["game_id"].n_unique() == expected["games"], d["time"].min() == expected["time_min"], d["time"].max() == expected["time_max"]])
players = fit["players"]
ai, bi = [np.searchsorted(players, d[c].to_numpy()) for c in ["a_id", "b_id"]]
ak, bk = [players[np.minimum(i, len(players)-1)] == d[c].to_numpy() for i, c in [(ai, "a_id"), (bi, "b_id")]]
ai, bi = [np.minimum(i, len(players)-1) for i in [ai, bi]]
ha, hb = [d[c].to_numpy() for c in ["h_a", "h_b"]]
X, Z = hr.design(d)
X, Z = X@fit["TX"], Z@fit["TZ"]
chains, draws, _ = fit["beta"].shape
samples = {k: fit[k] for k in ["beta", "width_beta", "u", "sd"]+(["rho"] if kind == "slope" else [])}
first, inverse, reverse = game_rows(d, X, Z, ha, hb)
worker = partial(chain_probabilities, samples=samples, X=X[first], Z=Z[first], ai=ai[first], bi=bi[first],
                 ak=ak[first], bk=bk[first], ha=ha[first], hb=hb[first], kind=kind)
with ThreadPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(worker, range(chains)))
chain_prob = np.stack([r[0] for r in results])[:, inverse]
chain_prob[:, reverse] = chain_prob[:, reverse, ::-1]
quadrature_error = max(r[1] for r in results)
print(prefix, "predictive chains complete", flush=True)
p = chain_prob.mean(0)
assert np.isfinite(p).all() and (p > 0).all() and np.allclose(p.sum(1), 1)
assert quadrature_error < 1e-7, quadrature_error
exports = json.loads(Path("outputs/tables/likelihood_selection_models.json").read_text())
export = next(m for m in exports if m["speed"] == speed and m["fold"] == fold)
P = {kind: p}
for name in ["baseline", "recent", "selected"]:
    m = export["models"][name if name == "baseline" else export["aliases"][name]]
    P[name] = spline.probabilities(tm.offset(d, m["baseline_coef"], "Lichess"), m)
y = d["y"].to_numpy()
loss = {name: {**sa.losses(prob, y), "score_mse": (y/2-prob[:, 2]-prob[:, 1]/2)**2} for name, prob in P.items()}
published = (pl.read_csv("outputs/tables/likelihood_selection_prediction.csv")
             .filter(pl.col("speed") == speed, pl.col("fold") == fold, pl.col("group") == "all",
                     pl.col("model") == "selected", pl.col("vs") == "baseline"))
for row in published.iter_rows(named=True):
    reproduced = (loss["selected"][row["metric"]]-loss["baseline"][row["metric"]]).mean()
    assert np.isclose(reproduced, row["diff"], atol=1e-12, rtol=1e-9), (reproduced, row["diff"])
rows, curves, mc = [], [], []
groups = [("all", np.ones(len(d), bool)), ("novice", d["novice"].to_numpy()),
          ("established", ~d["novice"].to_numpy()), ("both_seen", ak & bk), ("any_unseen", ~(ak & bk))]
for group, mask in groups:
    for reference in ["baseline", "recent", "selected"]:
        for metric in ["log_loss", "rps", "score_mse"]:
            delta = (loss[kind][metric]-loss[reference][metric])[mask]
            se = lk.dyad_se((delta-delta.mean())/len(delta), d["a_id"].to_numpy()[mask], d["b_id"].to_numpy()[mask])
            rows.append({"speed": speed, "fold": fold, "kind": kind, "group": group, "vs": reference, "metric": metric,
                         "n": len(delta), "mean": float(loss[kind][metric][mask].mean()), "diff": delta.mean(), "se": se,
                         "lo": delta.mean()-1.96*se, "hi": delta.mean()+1.96*se})
curves = [{"speed": speed, "fold": fold, "kind": kind, **r} for r in calibration_rows(y, P, groups)]
for chain in range(chains):
    lp = sa.losses(chain_prob[chain], y)
    for metric in ["log_loss", "rps"]:
        mc.append({"speed": speed, "fold": fold, "kind": kind, "chain": chain, "metric": metric,
                   "mean": lp[metric].mean(), "diff_vs_selected": (lp[metric]-loss["selected"][metric]).mean()})
pl.DataFrame(rows).write_csv(f"outputs/tables/{prefix}_prediction.csv")
pl.DataFrame(curves).write_csv(f"outputs/tables/{prefix}_calibration.csv")
pl.DataFrame(mc).write_csv(f"outputs/tables/{prefix}_mc.csv")
# 保留逐列預測，讓兩段合併的 dyadic uncertainty 與 random-intercept 對照可直接重算。
d.select("game_id", "a_id", "b_id", "y", "novice").with_columns(
    p_loss=pl.Series(p[:, 0]), p_draw=pl.Series(p[:, 1]), p_win=pl.Series(p[:, 2]),
    both_seen=pl.Series(ak & bk),
    **{f"{name}_{metric}": pl.Series(values) for name, metrics in loss.items() for metric, values in metrics.items()}
).write_parquet(f"data/processed/{prefix}_prediction.parquet")
print(pl.DataFrame(rows).filter(pl.col("group") == "all", pl.col("vs") == "selected"), flush=True)
print("quadrature_error", quadrature_error, flush=True)
