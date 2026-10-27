# 把對局當獨立時，停止規則的回測與依狀態調整勝率的逐步保留，判定是否與相依修正後的相同。
# 樣本與模型複製自 folk_stop_rules.py、state_adjusted_probs.py，只換標準誤；ML 參照不重跑。

import os
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, "scripts")

import jax.numpy as jnp
import numpy as np
import polars as pl
from scipy.stats import norm

import likelihood as lk
import ordinal
from arena_batches import STOP_RULES, player_sample_rd, stop_rule_sample

UTC = "UTC"
SPEEDS = ["Bullet", "Blitz", "Classical"]

d, _ = stop_rule_sample()
rows = []
for r in STOP_RULES:
    for s in SPEEDS:
        z = d.filter(pl.col("speed") == s)
        a = z.filter(pl.col(f"del_{r}"))["resid"].to_numpy()
        b = z.filter(~pl.col(f"del_{r}"))["resid"].to_numpy()
        diff = a.mean() - b.mean()
        se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        rows.append({"rule": r, "speed": s, "diff": diff, "se_iid": se, "p_iid": 2 * norm.sf(abs(diff) / se)})
folk = pl.DataFrame(rows)
fam = folk["speed"].is_in(["Bullet", "Blitz"])
folk = folk.with_columns(p_holm_iid=lk.holm(folk["p_iid"], fam))
ref = pl.read_csv("outputs/tables/folk_stop_rules_backtest.csv").filter(pl.col("group") == "all").select("rule", "speed", "se", "p_holm")
folk = folk.join(ref, on=["rule", "speed"])
with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250):
    print(folk)
folk.write_csv("outputs/tables/explore_naive_se_folk.csv")

grp = ["player", "tournament_id"]
t = (
    player_sample_rd()
    .sort("player", "tournament_id", "start", "game_id")
    .with_columns(
        prev_result=pl.col("result").shift(1).over(grp),
        dr=((pl.col("elo") - pl.col("elo").shift(1).over(grp)) / 100).cast(pl.Float64),
    )
    .filter(pl.col("prev_result").is_not_null())
    .with_columns(
        y=(pl.col("score") * 2).cast(pl.Int64),
        prev_draw=(pl.col("prev_result") == "draw").cast(pl.Float64),
        prev_win=(pl.col("prev_result") == "win").cast(pl.Float64),
        train=pl.col("start") < pl.datetime(2014, 7, 1, time_zone=UTC),
    )
)
XCOLS = ["prev_draw", "prev_win", "dr"]


def ll(t, eta, w, X, y):
    return ordinal.ll(eta + t[0] + X @ t[2:], w * jnp.exp(t[1]), y)


def probs(t, eta, w, X):
    return ordinal.probs(eta + t[0] + X @ t[2:], w * np.exp(t[1]))


rows = []
for spd in SPEEDS:
    ts = t.filter(pl.col("speed") == spd)
    tr, te = ts.filter(pl.col("train")), ts.filter(~pl.col("train"))
    args = lambda d: (d["eta"].to_numpy(), d["w"].to_numpy(), d.select(XCOLS).to_numpy().astype(np.float64), d["y"].to_numpy())
    atr, ate = args(tr), args(te)
    fits = {}

    def logloss(free):
        th = lk.fit(ll, np.zeros(5), np.array(free, dtype=int), atr, tr["player"].to_numpy(), tr["opponent"].to_numpy())[0]
        p = probs(th, *ate[:3])
        return -np.log(p[np.arange(len(ate[3])), ate[3]])

    # 兩種判定的保留路徑可能分岔，各自沿自己的路徑走
    for crit in ["dyadic", "iid"]:
        kept, cur = [], "baseline"
        fits[(crit, "baseline")] = logloss([])
        for nm, idx in [("arena", [0, 1]), ("prev_result", [2, 3]), ("dr", [4])]:
            key = tuple(kept + idx)
            if key not in fits:
                fits[key] = logloss(kept + idx)
            dlt = fits[key] - fits[(crit, "baseline")] if cur == "baseline" else fits[key] - fits[tuple(kept)]
            se_iid = dlt.std(ddof=1) / np.sqrt(len(dlt))
            se_dy = lk.dyad_se((dlt - dlt.mean()) / len(dlt), te["player"].to_numpy(), te["opponent"].to_numpy())
            se = se_dy if crit == "dyadic" else se_iid
            keep = dlt.mean() + 1.96 * se < 0
            rows.append({"speed": spd, "crit": crit, "step": nm, "vs": cur, "diff": dlt.mean(), "se_iid": se_iid, "se_dyadic": se_dy, "kept": keep})
            if keep:
                kept, cur = kept + idx, nm
steps = pl.DataFrame(rows)
with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250):
    print(steps)
steps.write_csv("outputs/tables/explore_naive_se_steps.csv")
