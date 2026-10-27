# 續下的選擇：被隨機分到較強的對手，是否改變同一場錦標賽中是否再開一局；之後幾局的表現只在續下者身上有定義，
# crossover 模型的結論要推到「兩種對手強弱下都會續下的玩家」，須經界限（規格見 docs/methods.md「續下的選擇模型」
# 「續下的選擇：界限與對照」）。
# - 續下模型：logistic 迴歸，以局加權的修正概似比檢定；總效果只放 h 與配對前的變數，給定結果的模型另加本局結果
# - 兩組：e_bar 的五等分格內，h 的最強與最弱五分之一。h 的分布依玩家在候選者中的位置而定，全體五等分的組成不同
# - 界限：Lee（單調性）、不假設單調性的 worst-case、inverse probability weighting、選擇上的有參數敏感度分析；
#   反應變數為之後第 k 局的主基準殘差。玩家 block bootstrap 的 Imbens–Manski 區間

import os
import sys

# 先於 likelihood import jax 設定：float64 在這張 GPU 上很慢
os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, "scripts")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import weighted_trim_rs
from scipy import optimize, stats

import likelihood as lk
from arena_batches import continuation_sample

B = 999
SPEEDS = ["Bullet", "Blitz", "Classical"]
LAYERS = ["multi", "single"]
LAGS = [1, 2, 3]
WEIGHTINGS = ["player", "game"]
NQ = 5
GAMMAS = np.round(np.linspace(-40, 40, 81), 1)


def holm(p):
    o = np.argsort(p)
    out = np.empty(len(p))
    out[o] = np.minimum(1, np.maximum.accumulate(p[o] * (len(p) - np.arange(len(p)))))
    return out


d = continuation_sample(LAGS)

sample = pl.concat([
    d.filter(pl.col(f"cont{k}").is_not_null())
    .group_by("speed", "layer")
    .agg(rows=pl.len(), players=pl.col("player").n_unique(), cont_rate=pl.col(f"cont{k}").mean())
    .with_columns(lag=pl.lit(k))
    for k in LAGS
]).sort("lag", "speed", "layer")
print(sample)
sample.write_csv("outputs/tables/continuation_selection_sample.csv")

# 續下的 logistic 迴歸。總效果只放配對前已定的變數；給定結果的模型另加本局結果與「結果 × h」（負為參照）
BASE = ["const", "e_bar", "elo_c", "elo_c2", "log_games_in_tour", "log_remaining", "novice_rd", "white"]
RES = ["win", "draw"]
HX = ["h_win", "h_draw"]
XCOLS = BASE + ["h"] + RES + HX


def free_of(cols):
    return np.array([XCOLS.index(c) for c in cols])


T0 = np.zeros(len(XCOLS))
test_rows, coef_rows = [], []
p_hat = {}
for spd in SPEEDS:
    for layer in LAYERS:
        s = d.filter(pl.col("speed") == spd, pl.col("layer") == layer, pl.col("cont1").is_not_null()).with_columns(const=pl.lit(1.0))
        X = s.select(XCOLS).to_numpy().astype(np.float64)
        a = (X, s["cont1"].cast(pl.Float64).to_numpy())
        w_, b_ = s["player"].to_numpy(), s["opponent"].to_numpy()
        for model, c0, c1 in [("total", BASE, BASE + ["h"]), ("given_result", BASE + RES, BASE + ["h"] + RES + HX)]:
            th1, V, res = lk.lrt_player_dyad(lk.logit_ll, T0, free_of(c0), free_of(c1), a, w_, b_)
            test_rows.append({"speed": spd, "layer": layer, "model": model, "n": s.height, **res})
            f1 = free_of(c1)
            for i, j in enumerate(f1):
                coef_rows.append({"speed": spd, "layer": layer, "model": model, "param": XCOLS[j], "est": th1[j], "se_dyadic": np.sqrt(V[i, i])})
            if model == "given_result":
                p_hat[(spd, layer)] = (s.select("game_id", "player"), 1 / (1 + np.exp(-(X @ th1))))
        print(spd, layer, "models", flush=True)
tests = pl.DataFrame(test_rows)
# 升級規則的診斷：總效果模型中 h 一項，Bullet 與 Blitz 的兩層為一族
fam = tests.select((pl.col("model") == "total") & pl.col("speed").is_in(["Bullet", "Blitz"])).to_series().to_numpy()
ph = np.full(tests.height, np.nan)
ph[fam] = holm(tests["p"].to_numpy()[fam])
tests = tests.with_columns(p_holm=pl.Series(ph))
coef = pl.DataFrame(coef_rows)
with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250):
    print(tests)
    print(coef.filter(pl.col("param").str.starts_with("h")))
tests.write_csv("outputs/tables/continuation_selection_tests.csv")
coef.write_csv("outputs/tables/continuation_selection_coef.csv")
h_sign = {(r["speed"], r["layer"]): np.sign(r["est"]) for r in coef.filter(pl.col("model") == "total", pl.col("param") == "h").iter_rows(named=True)}

# 玩家 block bootstrap 的重抽次數：每位玩家一列，所有樣本共用同一組重抽
players = np.unique(d["player"].to_numpy())
rng = np.random.default_rng(1)
C = rng.multinomial(len(players), np.full(len(players), 1 / len(players)), size=B).astype(np.float64)


# Imbens–Manski（2004）的區間：涵蓋部分識別的參數本身
def im_ci(lo, hi, se_lo, se_hi):
    sd = max(se_lo, se_hi)
    c = optimize.brentq(lambda c: stats.norm.cdf(c + (hi - lo) / sd) - stats.norm.cdf(-c) - 0.95, 1e-6, 10) if sd > 0 else 1.96
    return lo - c * se_lo, hi + c * se_hi


rate_rows, mono_rows, bound_rows, sens_rows, cell_rows = [], [], [], [], []
for spd in SPEEDS:
    for layer in LAYERS:
        for k in LAGS:
            s = d.filter(pl.col("speed") == spd, pl.col("layer") == layer, pl.col(f"cont{k}").is_not_null())
            # e_bar 的五等分格，格內再依 h 分五等分；只用處理與配對前的變數
            eb, h = s["e_bar"].to_numpy(), s["h"].to_numpy()
            cell = np.searchsorted(np.quantile(eb, np.linspace(0, 1, NQ + 1)[1:-1]), eb, side="right")
            arm = np.zeros(len(h), dtype=int)
            for c in range(NQ):
                kc = cell == c
                arm[kc] = np.searchsorted(np.quantile(h[kc], np.linspace(0, 1, NQ + 1)[1:-1]), h[kc], side="right")
            cont = s[f"cont{k}"].to_numpy()
            y = s[f"resid_next{k}"].to_numpy()
            pi_ = np.searchsorted(players, s["player"].to_numpy())
            n_p = np.bincount(pi_)[pi_]
            spe = np.nanmean(s[f"spe_next{k}"].to_numpy())
            if k == 1:
                ids, ph_ = p_hat[(spd, layer)]
                p_cont = s.select("game_id", "player").join(ids.with_columns(p=pl.Series(ph_)), on=["game_id", "player"], how="left")["p"].to_numpy()
            for wt in WEIGHTINGS:
                wr = 1 / n_p if wt == "player" else np.ones(len(cont))

                # 每次重抽在各（格, 等分）的加權和：先依玩家加總再乘重抽次數，不展開成「重抽 × 列」的矩陣。
                # 全部組合疊成一個矩陣一次相乘：逐次的矩陣乘向量每次都從記憶體重讀整個 C，受頻寬限制
                gp = (cell * NQ + arm) * len(players) + pi_
                V = np.vstack([np.bincount(gp, wr * v, NQ * NQ * len(players)).reshape(NQ * NQ, len(players)) for v in [cont, np.ones(len(cont))]])
                Sc, Sn = np.split(C @ V.T, 2, axis=1)

                # 各等分的續下率，以格的權重平均
                cw = np.array([wr[cell == c].sum() for c in range(NQ)])
                rate, rate_b = np.zeros(NQ), np.zeros((B, NQ))
                for q in range(NQ):
                    for c in range(NQ):
                        kk = (cell == c) & (arm == q)
                        rate[q] += cw[c] / cw.sum() * (wr[kk] * cont[kk]).sum() / wr[kk].sum()
                        rate_b[:, q] += cw[c] / cw.sum() * Sc[:, c * NQ + q] / Sn[:, c * NQ + q]
                    rate_rows.append({"speed": spd, "layer": layer, "lag": k, "weighting": wt, "arm": q + 1, "rate": rate[q], "se": rate_b[:, q].std()})
                # 單調性：相鄰差中方向與總效果模型 h 係數相反者，單尾檢定、Holm 校正
                sg = h_sign[(spd, layer)]
                dif, dif_b = np.diff(rate), np.diff(rate_b, axis=1)
                z = -sg * dif / dif_b.std(0)
                p_rev = stats.norm.sf(z)
                p_rev_h = holm(p_rev)
                for j in range(NQ - 1):
                    mono_rows.append({"speed": spd, "layer": layer, "lag": k, "weighting": wt, "pair": f"{j + 1}-{j + 2}", "diff": dif[j], "se": dif_b[:, j].std(), "expected_sign": sg, "p_reversal": p_rev[j], "p_holm": p_rev_h[j]})
                monotone = bool((p_rev_h >= 0.05).all())

                # 每一格內，最強（arm 4）與最弱（arm 0）兩組的界限
                res = {m: [] for m in ["continuers", "lee_lo", "lee_hi", "wc_lo", "wc_hi", "ipw"]}
                wts, wts_ipw, ess, wmax = [], [], [], []
                sens = np.zeros(len(GAMMAS))
                for c in range(NQ):
                    A, Bm = (cell == c) & (arm == NQ - 1), (cell == c) & (arm == 0)
                    ia, ib = c * NQ + NQ - 1, c * NQ
                    pa_b = Sc[:, ia] / Sn[:, ia]
                    pb_b = Sc[:, ib] / Sn[:, ib]
                    pa, pb = (wr[A] * cont[A]).sum() / wr[A].sum(), (wr[Bm] * cont[Bm]).sum() / wr[Bm].sum()
                    # 實際資料放在第 0 列，bootstrap 接在其後
                    P_a, P_b = np.r_[pa, pa_b], np.r_[pb, pb_b]
                    # Lee：續下率較高的一組修剪 1 - p_低 / p_高；方向逐列（含每次重抽）判斷
                    trim_a = P_a > P_b
                    sh = np.minimum(P_a, P_b) / np.maximum(P_a, P_b)
                    # 不假設單調性：兩組都會續下者占一組續下者的比例至少為 (p_a + p_b - 1) / p_a
                    # p_a + p_b <= 1 時這一層可能是空的，界限退化為反應變數的極值；比例取極小的正數即得極值
                    sa, sb = np.maximum((P_a + P_b - 1) / P_a, 1e-9), np.maximum((P_a + P_b - 1) / P_b, 1e-9)
                    # 續下者依 y 排序後，每列權重為玩家權重乘該次重抽的次數，取加權平均與兩端的修剪平均；
                    # 第 0 欄為 Lee 的比例 sh，第 1 欄為 worst-case 的比例
                    grp_ = {}
                    for nm, msk, s_wc in [("A", A, sa), ("B", Bm, sb)]:
                        ix = np.flatnonzero(msk & cont)
                        ix = ix[np.argsort(y[ix])]
                        ys, ws = y[ix], wr[ix]
                        grp_[nm] = (ys, ws, *weighted_trim_rs.trim_means(ys, pi_[ix], ws, C, np.column_stack([sh, s_wc])))
                    (ya, wa, ma, lo_a, hi_a), (yb, wb, mb, lo_b, hi_b) = grp_["A"], grp_["B"]
                    res["continuers"].append(ma - mb)
                    lo = np.where(trim_a, lo_a[:, 0] - mb, ma - hi_b[:, 0])
                    hi = np.where(trim_a, hi_a[:, 0] - mb, ma - lo_b[:, 0])
                    res["lee_lo"].append(lo)
                    res["lee_hi"].append(hi)
                    res["wc_lo"].append(lo_a[:, 1] - hi_b[:, 1])
                    res["wc_hi"].append(hi_a[:, 1] - lo_b[:, 1])
                    wcell = np.r_[wr[A | Bm].sum(), Sn[:, ia] + Sn[:, ib]]
                    wts.append(wcell * np.minimum(P_a, P_b))
                    wts_ipw.append(wcell)
                    cell_rows.append({"speed": spd, "layer": layer, "lag": k, "weighting": wt, "cell": c + 1, "p_strong": pa, "p_weak": pb, "continuers": res["continuers"][-1][0], "lee_lo": lo[0], "lee_hi": hi[0]})
                    if k == 1:
                        mi = []
                        for msk in [A, Bm]:
                            kk = msk & cont
                            wi = wr[kk] / p_cont[kk]
                            mi.append(weighted_trim_rs.trim_means(y[kk], pi_[kk], wi, C, np.zeros((B + 1, 0)))[0])
                            ess.append(wi.sum() ** 2 / (wi**2).sum())
                            # 只看續下機率倒數的部分：玩家權重本身的差異不是 inverse probability weighting 造成的
                            wmax.append((1 / p_cont[kk]).max() / (1 / p_cont[kk]).mean())
                        res["ipw"].append(mi[0] - mi[1])
                    yh, wh, ylo, wl = (ya, wa, yb, wb) if trim_a[0] else (yb, wb, ya, wa)
                    ml = (wl * ylo).sum() / wl.sum()
                    # 選擇上的有參數模型：續下者屬於「兩組都會續下」的機率為 expit(alpha + gamma y)，平均等於 sh
                    mh = weighted_trim_rs.tilt_means(yh, wh, sh[0], GAMMAS)
                    sens += wts[-1][0] * ((ml - mh) if not trim_a[0] else (mh - ml))
                Wt, Wi = np.array(wts), np.array(wts_ipw)
                sens /= Wt[:, 0].sum()
                out = {m: (np.array(v) * (Wi if m == "ipw" else Wt)).sum(0) / (Wi if m == "ipw" else Wt).sum(0) for m, v in res.items() if v}
                row0 = {"speed": spd, "layer": layer, "lag": k, "weighting": wt, "monotone": monotone, "score_per_elo": spe}
                for m, lo_, hi_ in [("continuers", "continuers", "continuers"), ("lee", "lee_lo", "lee_hi"), ("worst_case", "wc_lo", "wc_hi"), ("ipw", "ipw", "ipw")]:
                    if lo_ not in out:
                        continue
                    L, U = out[lo_][0], out[hi_][0]
                    sl, su = out[lo_][1:].std(), out[hi_][1:].std()
                    cl, cu = im_ci(L, U, sl, su)
                    extra = {"ess_min": min(ess), "w_max": max(wmax)} if m == "ipw" else {}
                    bound_rows.append(row0 | {"method": m, "lo": L, "hi": U, "se_lo": sl, "se_hi": su, "ci_lo": cl, "ci_hi": cu, "ci_excludes_0": bool(cl > 0 or cu < 0), "lo_elo": L / spe, "hi_elo": U / spe, "ci_lo_elo": cl / spe, "ci_hi_elo": cu / spe} | extra)
                sens_rows.append(pl.DataFrame({"gamma": GAMMAS, "effect": sens}).with_columns(speed=pl.lit(spd), layer=pl.lit(layer), lag=pl.lit(k), weighting=pl.lit(wt)))
            print(spd, layer, k, "bounds", flush=True)

rates = pl.DataFrame(rate_rows)
mono = pl.DataFrame(mono_rows)
bounds = pl.DataFrame(bound_rows, infer_schema_length=None)
sens = pl.concat(sens_rows)
# 效果的點估計變號時的 gamma：每組只取變號的第一個格點
flip = (
    sens.sort("speed", "layer", "lag", "weighting", "gamma")
    .with_columns(cross=(pl.col("effect").sign() != pl.col("effect").sign().shift().over("speed", "layer", "lag", "weighting")) & pl.col("effect").shift().over("speed", "layer", "lag", "weighting").is_not_null())
    .filter(pl.col("cross"))
)
with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250):
    print(rates.filter(pl.col("lag") == 1))
    print(mono.filter(pl.col("p_holm") < 0.05))
    print(bounds.select("speed", "layer", "lag", "weighting", "monotone", "method", "lo", "hi", "ci_lo", "ci_hi", "ci_excludes_0", "lo_elo", "hi_elo"))
    print(flip)
rates.write_csv("outputs/tables/continuation_selection_rates.csv")
mono.write_csv("outputs/tables/continuation_selection_monotone.csv")
bounds.write_csv("outputs/tables/continuation_selection_bounds.csv")
sens.write_csv("outputs/tables/continuation_selection_sensitivity.csv")
flip.drop("cross").write_csv("outputs/tables/continuation_selection_flip.csv")
pl.DataFrame(cell_rows).write_csv("outputs/tables/continuation_selection_cells.csv")

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for ax, spd in zip(axes, ["Bullet", "Blitz"]):
    # 不假設單調性的界限寬到會壓扁其他兩者，只留在表中
    for off, (m, mk) in zip([-0.1, 0.1], [("continuers", "o"), ("lee", "s")]):
        z = bounds.filter(pl.col("speed") == spd, pl.col("layer") == "multi", pl.col("weighting") == "player", pl.col("method") == m).sort("lag")
        x = z["lag"].to_numpy() + off
        ax.vlines(x, z["ci_lo"], z["ci_hi"], lw=1)
        ax.vlines(x, z["lo"], z["hi"], lw=4)
        ax.plot(x, (z["lo"] + z["hi"]) / 2, mk, ms=4, label="continuers only" if m == "continuers" else "Lee bounds")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title(spd)
    ax.set_xticks(LAGS)
    ax.set_xlabel("k-th game after the treated game")
axes[0].set_ylabel("strongest minus weakest fifth (score)")
axes[0].legend()
fig.tight_layout()
fig.savefig("outputs/figures/continuation_selection_bounds.png", dpi=150)
