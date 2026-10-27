# 依狀態調整的勝率：主基準加近況的有序 logistic（規格見 docs/methods.md「依狀態調整的勝率」）。
# 主基準的 eta、w 作 offset，各速度類別在 2014 上半年估計、下半年評估。
# arena 的整體校正與事先寫定的近況候選項（上一局結果、上一局等級分的變動 dr）依序加入，
# 再加入對手的同一組近況，雙方共用係數、以差的形式進入。每局以雙方的視角各出現一次、結果互為鏡像，
# 對手的近況另給一組係數時，最大概似估計因對稱而落在共用係數上，兩者等價，所以只估共用的形式。
# 下半年的對數分數改善且 dyadic 區間不含 0 才保留。LightGBM、XGBoost 與 MLP 輸入相同，只作參照。

import os
import sys

# 先於 import jax 設定：float64 在這張 GPU 上很慢
os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, "scripts")

import jax
import jax.numpy as jnp
import lightgbm
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import xgboost

import likelihood as lk
import novice
import ordinal
from arena_batches import player_sample_rd

UTC = "UTC"
SPEEDS = ["Bullet", "Blitz", "Classical"]
RESULTS = ["loss", "draw", "win"]

grp = ["player", "tournament_id"]
code = pl.col("result").replace_strict({"loss": 0, "draw": 1, "win": 2}, return_dtype=pl.Float64)
base = (
    player_sample_rd()
    .sort("player", "tournament_id", "start", "game_id")
    .with_columns(
        prev_result=pl.col("result").shift(1).over(grp),
        prev2=code.shift(2).over(grp),
        prev3=code.shift(3).over(grp),
        prev1=code.shift(1).over(grp),
        dr=((pl.col("elo") - pl.col("elo").shift(1).over(grp)) / 100).cast(pl.Float64),
    )
)
# 對手的近況取自對手在同一局的那一列；對手在這場沒有上一局時 opp_missing 為 1，近況項記為 0
opp = base.select("game_id", opponent="player", opp_prev_result="prev_result", opp_prev1="prev1", opp_dr="dr")
t = (
    base.join(opp, on=["game_id", "opponent"], how="left")
    .filter(pl.col("prev_result").is_not_null())
    .with_columns(
        y=(pl.col("score") * 2).cast(pl.Int64),
        prev_draw=(pl.col("prev_result") == "draw").cast(pl.Float64),
        prev_win=(pl.col("prev_result") == "win").cast(pl.Float64),
        opp_missing=pl.col("opp_prev_result").is_null().cast(pl.Float64),
        opp_prev_draw=(pl.col("opp_prev_result") == "draw").fill_null(False).cast(pl.Float64),
        opp_prev_win=(pl.col("opp_prev_result") == "win").fill_null(False).cast(pl.Float64),
        opp_dr=pl.col("opp_dr").fill_null(0.0),
        novice_rd=novice.novice_rd("rd", "opp_rd"),
        train=pl.col("start") < pl.datetime(2014, 7, 1, time_zone=UTC),
        # 梯度提升的 early stopping 用上半年的最後一個月，下半年完全不碰
        valid=pl.col("start") >= pl.datetime(2014, 6, 1, time_zone=UTC),
        white=(pl.col("color") == "white").cast(pl.Float64),
    )
    .with_columns(
        d_prev_draw=pl.col("prev_draw") - pl.col("opp_prev_draw"),
        d_prev_win=pl.col("prev_win") - pl.col("opp_prev_win"),
        d_dr=pl.col("dr") - pl.col("opp_dr"),
    )
)
sample = t.group_by("speed", "train", "prev_result").agg(n=pl.len(), players=pl.col("player").n_unique(), opp_missing=pl.col("opp_missing").mean()).sort("speed", "train", "prev_result")
print(sample)
sample.write_csv("outputs/tables/state_adjusted_probs_sample.csv")

# sym_* 乘雙方之差，即兩人共用的係數
PARAMS = ["arena_shift", "arena_log_draw_width", "prev_draw", "prev_win", "dr", "sym_prev_draw", "sym_prev_win", "sym_dr", "opp_missing"]
XCOLS = ["prev_draw", "prev_win", "dr", "d_prev_draw", "d_prev_win", "d_dr", "opp_missing"]


# 主基準的 eta、w 為 offset。t0 平移 eta、t1 縮放和局帶寬：主基準在全站對局上估計，
# arena 的玩家與大廳不同，這兩項吸收 arena 的整體差異
def ll(t, eta, w, X, y):
    return ordinal.ll(eta + t[0] + X @ t[2:], w * jnp.exp(t[1]), y)


def probs(t, eta, w, X):
    return ordinal.probs(eta + t[0] + X @ t[2:], w * np.exp(t[1]))


# MLP 的輸出層給線性預測式與和局帶寬的對數，經與有序 logistic 相同的累積 logit 得到機率
def mlp_out(p, X):
    h = jax.nn.relu(X @ p[0] + p[1])
    h = jax.nn.relu(h @ p[2] + p[3])
    o = h @ p[4] + p[5]
    return o[:, 0], jnp.exp(o[:, 1])


def mlp_nll(p, X, y):
    return -jnp.mean(ordinal.ll(*mlp_out(p, X), y))


@jax.jit
def mlp_step(p, m, v, k, X, y):
    g = jax.grad(mlp_nll)(p, X, y)
    m = [0.9 * a + 0.1 * b for a, b in zip(m, g)]
    v = [0.999 * a + 0.001 * b**2 for a, b in zip(v, g)]
    p = [a - 1e-3 * (mm / (1 - 0.9**k)) / (jnp.sqrt(vv / (1 - 0.999**k)) + 1e-8) for a, mm, vv in zip(p, m, v)]
    return p, m, v


mlp_val = jax.jit(mlp_nll)


def mlp_x(d):
    cont = d.select("elo", "opp_elo", "rd", "opp_rd", "eta", "w", "dr", "opp_dr", pl.col("games_in_tour").log1p()).to_numpy().astype(np.float32)
    # 前幾局不存在時歸入第四類，不以 0 或平均填補
    cat = np.concatenate([np.eye(4, dtype=np.float32)[d[c].fill_null(3).cast(pl.Int64).to_numpy()] for c in ["prev1", "prev2", "prev3", "opp_prev1"]], 1)
    return cont, np.c_[d["white"].to_numpy().astype(np.float32), cat]


def losses(p, y):
    O = np.eye(3)[y]
    C, Co = np.cumsum(p, 1)[:, :2], np.cumsum(O, 1)[:, :2]
    return {"log_loss": -np.log(p[np.arange(len(y)), y]), "rps": ((C - Co) ** 2).sum(1) / 2}


def diff_row(spd, a, b, la, lb, msk, wg, bg, group):
    rows = []
    for metric in ["log_loss", "rps"]:
        dlt = la[metric][msk] - lb[metric][msk]
        se = lk.dyad_se((dlt - dlt.mean()) / msk.sum(), wg[msk], bg[msk])
        rows.append({"speed": spd, "group": group, "model": a, "vs": b, "metric": metric, "n": int(msk.sum()), "mean": la[metric][msk].mean(), "diff": dlt.mean(), "diff_se_dyadic": se})
    return rows


pred_rows, coef_rows, step_rows, gap_rows, curve_parts = [], [], [], [], []
for spd in SPEEDS:
    ts = t.filter(pl.col("speed") == spd)
    tr, te = ts.filter(pl.col("train")), ts.filter(~pl.col("train"))
    args = lambda d: (d["eta"].to_numpy(), d["w"].to_numpy(), d.select(XCOLS).to_numpy().astype(np.float64), d["y"].to_numpy())
    atr, ate = args(tr), args(te)
    wte, bte = te["player"].to_numpy(), te["opponent"].to_numpy()
    T0 = np.zeros(len(PARAMS))
    fits, loss = {}, {}

    def run(nm, free):
        fits[nm] = lk.fit(ll, T0, np.array(free, dtype=int), atr, tr["player"].to_numpy(), tr["opponent"].to_numpy())
        loss[nm] = losses(probs(fits[nm][0], *ate[:3]), ate[3])

    run("baseline", [])
    # 依序加入：通過才併入，下一項以目前保留的模型為比較基準。arena 的整體校正不保留時，
    # 上一局結果的兩項以剛輸為參照，剛輸者的機率即主基準
    kept, cur = [], "baseline"
    for nm, idx in [("arena", [0, 1]), ("prev_result", [2, 3]), ("dr", [4])]:
        run(nm, kept + idx)
        r = diff_row(spd, nm, cur, loss[nm], loss[cur], np.ones(te.height, bool), wte, bte, "all")[0]
        keep = r["diff"] + 1.96 * r["diff_se_dyadic"] < 0
        step_rows.append({"speed": spd, "step": nm, "vs": cur, "diff": r["diff"], "diff_se_dyadic": r["diff_se_dyadic"], "kept": keep})
        if keep:
            kept, cur = kept + idx, nm
    # 對手的近況只取自身已保留的項
    own = [j for j in kept if 2 <= j <= 4]
    sym = [j for j in kept if j < 2] + [j + 3 for j in own] + [8]
    run("opp_sym", sym)
    r = diff_row(spd, "opp_sym", cur, loss["opp_sym"], loss[cur], np.ones(te.height, bool), wte, bte, "all")[0]
    keep = r["diff"] + 1.96 * r["diff_se_dyadic"] < 0
    step_rows.append({"speed": spd, "step": "opp_sym", "vs": cur, "diff": r["diff"], "diff_se_dyadic": r["diff_se_dyadic"], "kept": keep})
    if keep:
        kept, cur = sym, "opp_sym"
    final = cur
    th, llk, V, Hi, *_ = fits[final]
    for k, j in enumerate(kept):
        coef_rows.append({"speed": spd, "model": final, "param": PARAMS[j], "est": th[j], "se_dyadic": np.sqrt(V[k, k]), "se_iid": np.sqrt(Hi[k, k]), "n": tr.height})

    # 預測上限的參照：主基準的輸入加近況，全為對局前的資訊
    F = ["elo", "opp_elo", "rd", "opp_rd", "white", "eta", "w", "prev1", "prev2", "prev3", "dr", "opp_prev1", "opp_dr", "games_in_tour"]
    fit_, val_ = tr.filter(~pl.col("valid")), tr.filter(pl.col("valid"))
    Xf, Xv, Xt = (d.select(F).to_numpy().astype(np.float64) for d in [fit_, val_, te])
    gbm = lightgbm.train(
        {"objective": "multiclass", "num_class": 3, "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 500, "verbose": -1, "num_threads": 8},
        lightgbm.Dataset(Xf, fit_["y"].to_numpy()),
        num_boost_round=2000,
        valid_sets=[lightgbm.Dataset(Xv, val_["y"].to_numpy())],
        callbacks=[lightgbm.early_stopping(50, verbose=False)],
    )
    loss["lightgbm"] = losses(gbm.predict(Xt, num_iteration=gbm.best_iteration), ate[3])
    xgb = xgboost.train(
        {"objective": "multi:softprob", "num_class": 3, "eval_metric": "mlogloss", "eta": 0.05, "max_depth": 6, "min_child_weight": 500, "tree_method": "hist", "nthread": 8},
        xgboost.DMatrix(Xf, fit_["y"].to_numpy()),
        num_boost_round=2000,
        evals=[(xgboost.DMatrix(Xv, val_["y"].to_numpy()), "valid")],
        early_stopping_rounds=50,
        verbose_eval=False,
    )
    loss["xgboost"] = losses(xgb.predict(xgboost.DMatrix(Xt), iteration_range=(0, xgb.best_iteration + 1)), ate[3])

    (cf, kf), (cv, kv), (ct, kt) = mlp_x(fit_), mlp_x(val_), mlp_x(te)
    mu, sd = cf.mean(0), cf.std(0)
    Mf, Mv, Mt = (np.c_[(c - mu) / sd, k] for c, k in [(cf, kf), (cv, kv), (ct, kt)])
    yf, yv = fit_["y"].to_numpy(), val_["y"].to_numpy()
    rng = np.random.default_rng(0)
    H = 64
    p = [
        rng.normal(0, np.sqrt(2 / Mf.shape[1]), (Mf.shape[1], H)), np.zeros(H),
        rng.normal(0, np.sqrt(2 / H), (H, H)), np.zeros(H),
        rng.normal(0, 0.01, (H, 2)), np.array([0.0, np.log(fit_["w"].mean())]),
    ]
    p = [jnp.asarray(a, dtype=jnp.float32) for a in p]
    m, v = [jnp.zeros_like(a) for a in p], [jnp.zeros_like(a) for a in p]
    best, best_p, wait, k = np.inf, p, 0, 0
    for epoch in range(200):
        for idx in np.array_split(rng.permutation(len(yf)), len(yf) // 4096):
            k += 1
            p, m, v = mlp_step(p, m, v, k, Mf[idx], yf[idx])
        lv = float(mlp_val(p, Mv, yv))
        if lv < best:
            best, best_p, wait = lv, p, 0
        else:
            wait += 1
            if wait == 10:
                break
    e_mlp, w_mlp = mlp_out(best_p, Mt)
    loss["mlp"] = losses(ordinal.probs(np.asarray(e_mlp, dtype=np.float64), np.asarray(w_mlp, dtype=np.float64)), ate[3])
    print(spd, "mlp epochs", epoch + 1, "best valid", best, flush=True)

    nov = te["novice_rd"].to_numpy()
    for group, msk in [("all", np.ones(te.height, bool)), ("novice", nov), ("established", ~nov)]:
        for nm in ["baseline", "arena", "prev_result", "dr", "opp_sym", "lightgbm", "xgboost", "mlp"]:
            pred_rows += diff_row(spd, nm, "baseline", loss[nm], loss["baseline"], msk, wte, bte, group)
        for nm in ["lightgbm", "xgboost", "mlp"]:
            pred_rows += diff_row(spd, nm, final, loss[nm], loss[final], msk, wte, bte, group)

    # 「再開一局，預期比系統的期望低多少」：修正後與主基準的期望得分之差，以主基準的斜率換算成等級分
    p = probs(th, *ate[:3])
    e_adj = p[:, 2] + p[:, 1] / 2
    te = te.with_columns(e_adj=pl.Series(e_adj)).with_columns(gap=pl.col("e_adj") - pl.col("e_base"))
    gap_rows.append(
        te.group_by("prev_result", "opp_prev_result", "novice_rd")
        .agg(n=pl.len(), e_base=pl.col("e_base").mean(), e_adj=pl.col("e_adj").mean(), score=pl.col("score").mean(), gap=pl.col("gap").mean(), gap_elo=(pl.col("gap") / pl.col("score_per_elo")).mean())
        .with_columns(speed=pl.lit(spd), model=pl.lit(final))
    )
    curve_parts.append(
        pl.concat([
            te.select("prev_result", "score", pred=pl.col("e_base"), model=pl.lit("baseline")),
            te.select("prev_result", "score", pred=pl.col("e_adj"), model=pl.lit(final)),
        ])
        .with_columns(ebin=(pl.col("pred") * 10).floor().clip(0, 9), speed=pl.lit(spd))
        .group_by("speed", "model", "prev_result", "ebin")
        .agg(n=pl.len(), pred_mean=pl.col("pred").mean(), resid_mean=(pl.col("score") - pl.col("pred")).mean(), resid_se=(pl.col("score") - pl.col("pred")).std() / pl.len().sqrt())
        .filter(pl.col("n") >= 30)
    )
    print(spd, "done", final, flush=True)

steps = pl.DataFrame(step_rows)
coef = pl.DataFrame(coef_rows)
pred = pl.DataFrame(pred_rows)
gap = pl.concat(gap_rows, how="diagonal").sort("speed", "novice_rd", "prev_result", "opp_prev_result")
curve = pl.concat(curve_parts).sort("speed", "model", "prev_result", "ebin")
with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250):
    print(steps)
    print(coef)
    print(pred.filter(pl.col("metric") == "log_loss"))
    print(gap)
steps.write_csv("outputs/tables/state_adjusted_probs_steps.csv")
coef.write_csv("outputs/tables/state_adjusted_probs_coef.csv")
pred.write_csv("outputs/tables/state_adjusted_probs_prediction.csv")
gap.write_csv("outputs/tables/state_adjusted_probs_gap.csv")
curve.write_csv("outputs/tables/state_adjusted_probs_curve.csv")

fig, axes = plt.subplots(2, len(SPEEDS), figsize=(13, 6), sharex=True, sharey=True)
for j, spd in enumerate(SPEEDS):
    for i, nm in enumerate(["baseline", "adjusted"]):
        ax = axes[i, j]
        for res, mk in zip(RESULTS, ["v", "o", "^"]):
            z = curve.filter(pl.col("speed") == spd, pl.col("prev_result") == res, (pl.col("model") == "baseline") == (nm == "baseline"))
            ax.errorbar(z["pred_mean"], z["resid_mean"], yerr=1.96 * z["resid_se"], marker=mk, ms=3, capsize=2, lw=1, label=f"previous {res}")
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_title(f"{spd}, {'main baseline' if nm == 'baseline' else 'state-adjusted'}")
        if i == 1:
            ax.set_xlabel("predicted expected score")
axes[0, 0].legend(fontsize=8)
axes[0, 0].set_ylabel("mean score minus predicted")
axes[1, 0].set_ylabel("mean score minus predicted")
fig.tight_layout()
fig.savefig("outputs/figures/state_adjusted_probs_curve.png", dpi=150)
