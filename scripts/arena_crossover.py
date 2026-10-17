# arena 的 crossover 模型：被隨機分到較強的對手，是否改變同一場錦標賽中之後幾局的表現（carryover）。
# 處理 h = e_bar - e：期望匹配品質減本局的期望得分，正值為對手比依配對機制所期望的強；只含隨機配對產生的部分。
# 反應為之後第 k 局的勝、和、負，以主基準（加入 RD 的有序 logistic）的線性預測值為 offset，本局對手的強弱由基準吸收。
# 主推論：第 k 局之前那一局屬於兩局以上的配對批次時，在該批內依 lila 規則重新配對的 permutation test。
# 統計量取 offset 有序模型在 beta = 0 下的一步估計（score 除以 information）；虛無模型不含 h，
# 每局的 score 與 information 在重新配對下不變，每次重抽只需重算 h。
# sharp null 是「之前的對手完全不影響之後那一局的紀錄」，含有沒有那一局；續下的選擇另以界限處理。

import os
import sys

# 先於 import jax 設定：float64 在這張 GPU 上很慢
os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, "scripts")

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import optimize, stats
from statsmodels.stats.contingency_tables import StratifiedTable

import likelihood as lk
from arena_batches import e_grid, multi_null, player_sample, repair
from calibrated import rd_baseline

B = 2000
B_BOOT = 999
CHUNK_DRAWS = 50
SPEEDS = ["Bullet", "Blitz", "Classical"]
GROUPS = SPEEDS + ["all"]
# 統計量另在「上一局屬於純隨機規則的批」上算：2014 年 12 人以下的批有小幅偏離隨機（docs/random-pairing.md）
BATCHES = ["all", "random_rule"]
LAGS = [1, 2, 3]
NULLS = ["main", "unrestricted_before_1130"]
BAL = ["elo_c", "log_games_in_tour", "novice", "rd"]
C_GRID = np.round(np.arange(-0.5, 0.5001, 0.001), 3)

ids = pl.scan_parquet("data/processed/lichess_games.parquet").select("game_id", "white_id", "black_id")
rp = pl.scan_parquet("data/processed/glicko_replay.parquet").select("game_id", "white_rd", "black_rd")
d = (
    player_sample()
    .join(ids.collect(), on="game_id")
    .join(rp.collect(), on="game_id")
    .with_columns(
        rd=pl.when(pl.col("color") == "white").then("white_rd").otherwise("black_rd"),
        opp_rd=pl.when(pl.col("color") == "white").then("black_rd").otherwise("white_rd"),
        opponent=pl.when(pl.col("color") == "white").then("black_id").otherwise("white_id"),
    )
    .drop("white_id", "black_id", "white_rd", "black_rd")
)
d = rd_baseline(d).with_columns(
    h=pl.col("e_bar") - pl.col("e"),
    y=pl.col("result").replace_strict({"loss": 0, "draw": 1, "win": 2}, return_dtype=pl.Int8),
    resid=pl.col("score") - pl.col("e_base"),
    sp=pl.col("speed").replace_strict({s: i for i, s in enumerate(SPEEDS)}, return_dtype=pl.Int64),
)

# 兩局以上的批：每局兩列相鄰，依 bid、game_id、color 排序，與 multi_null 的要求相同
m = d.filter(pl.col("layer") == "multi").sort("bid", "game_id", "color").with_row_index("midx")
elos = np.unique(np.r_[d["elo"].to_numpy(), d["opp_elo"].to_numpy()])
E = e_grid(elos)
ns = multi_null(m, E, elos)
ui, tci, bi, nb = ns["ui"], ns["tci"], ns["bi"], ns["nb"]
e_bar_m = m["e_bar"].to_numpy()
h_act_m = e_bar_m - E[tci, ui, ui[np.arange(m.height) ^ 1]]
assert np.abs(h_act_m - m["h"].to_numpy()).max() < 1e-9

grp = ["player", "tournament_id"]
d = (
    d.join(m.select("game_id", "player", pl.col("midx").cast(pl.Int64)), on=["game_id", "player"], how="left")
    .with_columns(pl.col("midx").fill_null(-1))
    .sort("player", "tournament_id", "start", "game_id")
    .with_columns(
        *[pl.col("h").shift(k).over(grp).alias(f"h_lag{k}") for k in LAGS],
        *[pl.col("midx").shift(k).over(grp).alias(f"midx_lag{k}") for k in LAGS],
        *[pl.col("layer").shift(k).over(grp).alias(f"layer_lag{k}") for k in LAGS],
        bid_lag1=pl.col("bid").shift(1).over(grp),
        score_lag1=pl.col("score").shift(1).over(grp),
        opp_lag1=pl.col("opponent").shift(1).over(grp),
    )
)
t = d.filter(pl.col("h_lag1").is_not_null())

sample = pl.concat([
    t.filter(pl.col(f"h_lag{k}").is_not_null())
    .group_by("speed", layer=pl.col(f"layer_lag{k}"))
    .agg(rows=pl.len(), players=pl.col("player").n_unique(), tournaments=pl.col("tournament_id").n_unique())
    .with_columns(lag=pl.lit(k))
    for k in LAGS
]).sort("lag", "speed", "layer")
print(sample)
sample.write_csv("outputs/tables/arena_crossover_sample.csv")


# 虛無模型只有兩個參數（位移 alpha、和局帶寬的對數倍率 delta），以 numpy 估計：
# 之後要 fork 出工作行程，jax 的執行緒在 fork 前不能啟動
def ord_parts(off, wo, a, dl):
    eta, w = off + a, wo * np.exp(dl)
    return 1 / (1 + np.exp(-(eta - w))), 1 / (1 + np.exp(-(eta + w)))


def nll_null(p, off, wo, y):
    hi, nl = ord_parts(off, wo, *p)
    return -np.log(np.where(y == 2, hi, np.where(y == 0, 1 - nl, nl - hi))).sum()


# 每局對 eta 的 score 與 information（有序 logistic 的封閉形式）
off_t, wo_t, y_t, sp_t = (t[c].to_numpy() for c in ["eta", "w", "y", "sp"])
s_t, v_t = np.zeros(t.height), np.zeros(t.height)
null_rows = []
for j, spd in enumerate(SPEEDS):
    k = sp_t == j
    r = optimize.minimize(nll_null, [0.0, 0.0], args=(off_t[k], wo_t[k], y_t[k]), method="BFGS", options={"gtol": 1e-10})
    hi, nl = ord_parts(off_t[k], wo_t[k], *r.x)
    yk = y_t[k]
    s_t[k] = np.where(yk == 2, 1 - hi, np.where(yk == 0, -nl, 1 - hi - nl))
    v_t[k] = np.where(yk == 2, hi * (1 - hi), np.where(yk == 0, nl * (1 - nl), hi * (1 - hi) + nl * (1 - nl)))
    null_rows.append({"speed": spd, "alpha": r.x[0], "log_draw_scale": r.x[1], "score_sum": s_t[k].sum(), "n": int(k.sum())})
print(pl.DataFrame(null_rows))
resid_t, score_lag1 = t["resid"].to_numpy(), t["score_lag1"].to_numpy()

# 各 lag 的隨機化樣本：第 k 局之前那一局屬於兩局以上的批
rr_m = (m["rule"] == "random").to_numpy()
R = {}
for k in LAGS:
    mk = t[f"midx_lag{k}"].fill_null(-1).to_numpy()
    rows = np.flatnonzero(mk >= 0)
    R[k] = (rows, mk[rows], sp_t[rows], rr_m[mk[rows]])
# 劑量反應的五等分：切點取實際的 h（依速度類別），重新配對時固定
r1, m1, g1, _ = R[1]
cuts = np.stack([np.quantile(t["h_lag1"].to_numpy()[r1][g1 == j], [0.2, 0.4, 0.6, 0.8]) for j in range(len(SPEEDS))])
# 平衡檢驗：配對前已定的變數在批內置中，取其對 h 的斜率，在全部兩局以上的批的列上
sp_m = m["sp"].to_numpy()
Zc = []
for c in BAL:
    z = m[c].cast(pl.Float64).to_numpy()
    Zc.append(z - (np.bincount(bi, z, nb) / np.bincount(bi, minlength=nb))[bi])


def by_group(num, lab, rr):
    a = np.bincount(lab, num, len(SPEEDS))
    b = np.bincount(lab, num * rr, len(SPEEDS))
    return np.r_[a, a.sum(), b, b.sum()]


# 下一局的對手等級分與基準的期望得分：sharp null 下也不隨上一局的配對而變，用來看上一局經由下一局的配對與等級分更新影響下一局的管道
opp_next = ((t["opp_elo"] - 1500) / 100).to_numpy()
e_next = t["e_base"].to_numpy()


def stats_of(hm):
    out = {}
    for k in LAGS:
        rows, mk, g, rr = R[k]
        h = hm[mk]
        out[f"beta_lag{k}"] = by_group(h * s_t[rows], g, rr) / by_group(h * h * v_t[rows], g, rr)
    rows, mk, g, rr = R[1]
    h = hm[mk]
    qs = np.empty(len(h))
    for j in range(len(SPEEDS)):
        qs[g == j] = np.searchsorted(cuts[j], h[g == j]) - 2.0
    out["trend_quintile"] = by_group(qs * s_t[rows], g, rr) / by_group(qs * qs * v_t[rows], g, rr)
    hp, hn = np.maximum(h, 0), np.minimum(h, 0)
    bp = by_group(hp * s_t[rows], g, rr) / by_group(hp * hp * v_t[rows], g, rr)
    bn = by_group(hn * s_t[rows], g, rr) / by_group(hn * hn * v_t[rows], g, rr)
    out["beta_stronger"], out["beta_weaker"], out["asymmetry"] = bp, bn, bp - bn
    hh = by_group(h * h, g, rr)
    out["next_opp_elo"] = by_group(h * opp_next[rows], g, rr) / hh
    out["next_e_base"] = by_group(h * e_next[rows], g, rr) / hh
    out["next_resid_ols"] = by_group(h * resid_t[rows], g, rr) / hh
    # Anderson–Rubin 式的區間：H0(c) 下 resid - c * score_lag1 不隨上一局的配對而變
    out["ar_a"] = by_group(h * resid_t[rows], g, rr)
    out["ar_b"] = by_group(h * score_lag1[rows], g, rr)
    for c, z in zip(BAL, Zc):
        out[f"balance_{c}"] = by_group(hm * z, sp_m, rr_m) / by_group(hm * hm, sp_m, rr_m)
    return out


act = stats_of(h_act_m)
KEYS = list(act)


def draws(job):
    # 每次重新配對有自己的種子：平行時結果不隨工作分配到哪個行程而變
    kn, i0 = job
    return np.array([np.concatenate([stats_of(e_bar_m - E[tci, ui, ui[repair(ns, NULLS[kn], np.random.default_rng([kn, i]))]])[c] for c in KEYS]) for i in range(i0, i0 + CHUNK_DRAWS)])


# 單次重新配對是單執行緒的 numpy 運算；每個工作一次跑一批重抽，避免切得太碎
with ProcessPoolExecutor(os.cpu_count() - 2, mp_context=mp.get_context("fork")) as ex:
    nd = {nl: np.concatenate(list(ex.map(draws, [(kn, i) for i in range(0, B, CHUNK_DRAWS)]))) for kn, nl in enumerate(NULLS)}

ri_rows, ar_rows = [], []
for nl in NULLS:
    arr = nd[nl].reshape(B, len(KEYS), 2 * len(GROUPS))
    for ki, c in enumerate(KEYS):
        if c.startswith("ar_"):
            continue
        for gi in range(2 * len(GROUPS)):
            a, x = act[c][gi], arr[:, ki, gi]
            ri_rows.append({
                "null": nl, "stat": c, "speed": GROUPS[gi % 4], "batches": BATCHES[gi // 4], "est": a, "null_mean": x.mean(), "null_sd": x.std(), "est_centered": a - x.mean(),
                "p": (1 + (np.abs(x - x.mean()) >= abs(a - x.mean())).sum()) / (B + 1),
            })
    ia, ib = KEYS.index("ar_a"), KEYS.index("ar_b")
    for gi in range(2 * len(GROUPS)):
        Ta = act["ar_a"][gi] - C_GRID * act["ar_b"][gi]
        Tn = arr[:, ia, gi][:, None] - C_GRID[None, :] * arr[:, ib, gi][:, None]
        mu = Tn.mean(0)
        p = (1 + (np.abs(Tn - mu) >= np.abs(Ta - mu)).sum(0)) / (B + 1)
        keep = C_GRID[p > 0.05]
        ar_rows.append({
            "null": nl, "speed": GROUPS[gi % 4], "batches": BATCHES[gi // 4], "ratio": act["ar_a"][gi] / act["ar_b"][gi],
            "ci_lo": keep.min() if len(keep) else np.nan, "ci_hi": keep.max() if len(keep) else np.nan,
            "ci_at_grid_edge": bool(len(keep) and (keep.min() == C_GRID[0] or keep.max() == C_GRID[-1])),
            "ci_connected": bool(len(keep) and len(keep) == int(round((keep.max() - keep.min()) / 0.001)) + 1),
            "p_zero": p[C_GRID == 0][0],
        })
ri = pl.DataFrame(ri_rows)
# 主要假設事先指定為 Bullet 與 Blitz 的第一至三局 carryover（主虛無分布），以 Holm 法控制族錯誤率
fam = ri.filter(pl.col("null") == "main", pl.col("batches") == "all", pl.col("stat").str.starts_with("beta_lag"), pl.col("speed").is_in(["Bullet", "Blitz"])).sort("p")
holm = np.minimum(1, np.maximum.accumulate(fam["p"].to_numpy() * (fam.height - np.arange(fam.height))))
ri = ri.join(fam.select("null", "batches", "stat", "speed", p_holm=pl.Series(holm)), on=["null", "batches", "stat", "speed"], how="left")
slope_elo = {gname: t.filter(pl.col("speed") == gname)["slope_elo"].mean() if gname != "all" else t["slope_elo"].mean() for gname in GROUPS}
# 換算：h 增加 0.1 的效果相當於自身等級分高多少分（以基準的分差斜率換算）
ri = ri.with_columns(
    elo_per_0_1=pl.when(pl.col("stat").str.starts_with("beta_")).then(pl.col("est_centered") * 0.1 / pl.col("speed").replace_strict(slope_elo, return_dtype=pl.Float64))
)
with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250):
    print(ri)
ri.write_csv("outputs/tables/arena_crossover_ri.csv")
ar = pl.DataFrame(ar_rows)

# 輸棋本身的效果的 delta method 對照：比值 = cov(h, resid) / cov(h, score_lag1)，影響函數以 dyadic meat 估變異數；
# 同一局上一局的兩位玩家 h 幾乎互為相反數、得分相加為 1，故以上一局的對手組成 dyad
pl_t, opl_t, opn_t = (t[c].to_numpy() for c in ["player", "opp_lag1", "opponent"])
h1 = t["h_lag1"].to_numpy()
dm = []
for gi, gname in enumerate(GROUPS):
    k = r1 if gname == "all" else r1[g1 == gi]
    hc = h1[k] - h1[k].mean()
    c = (hc * resid_t[k]).sum() / (hc * score_lag1[k]).sum()
    phi = hc * (resid_t[k] - c * score_lag1[k]) / (hc * score_lag1[k]).sum()
    dm.append({"speed": gname, "ratio_delta": c, "se_delta": np.sqrt(lk.dyad_meat(phi[:, None], pl_t[k], opl_t[k])[0, 0])})
ar = ar.join(pl.DataFrame(dm).with_columns(batches=pl.lit("all")), on=["speed", "batches"], how="left")
with pl.Config(tbl_cols=-1, tbl_width_chars=250):
    print(ar)
ar.write_csv("outputs/tables/arena_crossover_ratio.csv")


# 模型的點估計與修正的概似比檢定。t0 位移、t1 和局帶寬的對數倍率、t2-t4 第一至三局之前的 h、
# t5 本局分差斜率的修正：上一局的配對會改變下一局的對手，基準的分差斜率在 arena 若有偏，偏誤會經由下一局的對手混入 carryover
def ll(t_, off, wo, x1, x2, x3, xd, y):
    eta = off + t_[0] + t_[2] * x1 + t_[3] * x2 + t_[4] * x3 + t_[5] * xd
    w = wo * jnp.exp(t_[1])
    u, v = eta + w, eta - w
    ldraw = jax.nn.log_sigmoid(u) + jax.nn.log_sigmoid(-v) + jnp.log(-jnp.expm1(-2 * w))
    return jnp.where(y == 2, jax.nn.log_sigmoid(v), jnp.where(y == 0, jax.nn.log_sigmoid(-u), ldraw))


model_rows = []
specs = {
    "lag1": ([1], pl.col("h_lag1").is_not_null(), False),
    "lag1_slope_adj": ([1], pl.col("h_lag1").is_not_null(), True),
    "lag1_3": ([1, 2, 3], pl.col("h_lag3").is_not_null(), False),
}
for spd in SPEEDS:
    for spec, (lags, keep, adj) in specs.items():
        for layer, lk_keep in [("multi", pl.col("layer_lag1") == "multi"), ("single", pl.col("layer_lag1") == "single"), ("all", pl.lit(True))]:
            s = t.filter(pl.col("speed") == spd, keep, lk_keep)
            xs = [s[f"h_lag{k}"].to_numpy() if k in lags else np.zeros(s.height) for k in LAGS]
            xd = ((s["elo"] - s["opp_elo"]) / 100).cast(pl.Float64).to_numpy()
            args = (s["eta"].to_numpy(), s["w"].to_numpy(), *xs, xd, s["y"].to_numpy().astype(np.float64))
            w_, b_ = s["player"].to_numpy(), s["opponent"].to_numpy()
            base = [0, 1, 5] if adj else [0, 1]
            free = np.r_[base, np.array(lags) + 1]
            th, ll1, V, Hi, *_ = lk.fit(ll, np.zeros(6), free, args, w_, b_)
            _, ll0, *_ = lk.fit(ll, np.zeros(6), base, args, w_, b_)
            test = lk.adjusted_lrt(2 * (ll1 - ll0), lk.lr_weights(Hi, V, np.arange(len(base), len(free))))
            se_elo = s["slope_elo"].mean()
            for j, k in enumerate(lags):
                j = len(base) + j
                model_rows.append({
                    "speed": spd, "spec": spec, "layer_lag1": layer, "lag": k, "n": s.height, "players": s["player"].n_unique(),
                    "beta": th[k + 1], "se_dyadic": np.sqrt(V[j, j]), "se_iid": np.sqrt(Hi[j, j]),
                    "elo_per_0_1": th[k + 1] * 0.1 / se_elo, "elo_per_0_1_se": np.sqrt(V[j, j]) * 0.1 / se_elo,
                    "alpha": th[0], "log_draw_scale": th[1], "slope_adj": th[5], **{f"lrt_{a}": b for a, b in test.items()},
                })
            print(spd, spec, layer, flush=True)
model = pl.DataFrame(model_rows)
with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250):
    print(model.select("speed", "spec", "layer_lag1", "lag", "n", "beta", "se_dyadic", "elo_per_0_1", "lrt_p", "lrt_lam_max"))
model.write_csv("outputs/tables/arena_crossover_model.csv")

# 劑量反應的描述：上一局 h 的五等分下，下一局得分減基準的平均；標準誤以 dyadic meat
dose_rows = []
hq = np.full(t.height, -1)
for j in range(len(SPEEDS)):
    k = sp_t == j
    hq[k] = np.searchsorted(cuts[j], h1[k])
for j, spd in enumerate(SPEEDS):
    for layer in ["multi", "all"]:
        for q in range(5):
            k = (sp_t == j) & (hq == q) & ((t["layer_lag1"] == "multi").to_numpy() if layer == "multi" else True)
            r = resid_t[k]
            dose_rows.append({
                "speed": spd, "layer_lag1": layer, "quintile": q + 1, "n": int(k.sum()), "h_mean": h1[k].mean(),
                "resid_mean": r.mean(), "resid_se": np.sqrt(lk.dyad_meat((r - r.mean())[:, None], pl_t[k], opn_t[k])[0, 0]) / k.sum(),
            })
dose = pl.DataFrame(dose_rows)
print(dose)
dose.write_csv("outputs/tables/arena_crossover_dose.csv")

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for ax, spd in zip(axes, ["Bullet", "Blitz"]):
    for layer, mk in [("multi", "o"), ("all", "s")]:
        z = dose.filter(pl.col("speed") == spd, pl.col("layer_lag1") == layer)
        ax.errorbar(z["h_mean"], z["resid_mean"], yerr=1.96 * z["resid_se"], marker=mk, capsize=3, label={"multi": "randomized batches", "all": "all batches"}[layer])
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title(spd)
    ax.set_xlabel("previous opponent strength h (expected score units)")
axes[0].set_ylabel("next-game score minus baseline")
axes[0].legend()
fig.tight_layout()
fig.savefig("outputs/figures/arena_crossover_dose.png", dpi=150)

# 對照：以上一局的批分層的 CMH 相關統計量與 Mantel–Haenszel 共同 odds ratio（下一局輸 × 上一局對手比期望強）；
# 只用上一局屬於兩局以上的批的列，批內至少兩列。CMH 的反應取下一局得分減基準；取原始得分（0、0.5、1，即有序 CMH）
# 時另含上一局經由下一局的對手與等級分更新的管道，MH 的「輸」也是原始結果
ctrl_rows = []
bid1, sc_t = t["bid_lag1"].to_numpy(), t["score"].to_numpy()
for gi, gname in enumerate(GROUPS):
    k = r1 if gname == "all" else r1[g1 == gi]
    st = np.unique(bid1[k], return_inverse=True)[1]
    n = np.bincount(st)
    ok = n[st] >= 2
    k, st = k[ok], np.unique(st[ok], return_inverse=True)[1]
    n = np.bincount(st)
    x = h1[k]
    xc = x - (np.bincount(st, x) / n)[st]
    for outcome, y in [("resid", resid_t[k]), ("raw_score", sc_t[k])]:
        yc = y - (np.bincount(st, y) / n)[st]
        T = (xc * yc).sum()
        var = (np.bincount(st, xc * xc) * np.bincount(st, yc * yc) / (n - 1)).sum()
        ctrl_rows.append({"speed": gname, "method": f"stratified_correlation_{outcome}", "stat": T**2 / var, "est": T / np.bincount(st, xc * xc).sum(), "p": stats.chi2.sf(T**2 / var, 1), "n": len(k), "strata": len(n)})
    y = sc_t[k]
    lo, st_ = (y == 0).astype(int), (x > 0).astype(int)
    tab = np.zeros((2, 2, len(n)))
    np.add.at(tab, (st_, lo, st), 1)
    keep = (tab.sum((0, 1)) >= 2) & (tab.sum(1) > 0).all(0) & (tab.sum(0) > 0).all(0)
    stt = StratifiedTable(tab[:, :, keep])
    ci = stt.oddsratio_pooled_confint()
    ctrl_rows.append({"speed": gname, "method": "mh_odds_ratio_loss_stronger", "stat": stt.test_null_odds().statistic, "est": stt.oddsratio_pooled, "est_lo": ci[0], "est_hi": ci[1], "p": stt.test_null_odds().pvalue, "n": int(tab[:, :, keep].sum()), "strata": int(keep.sum())})

    # 下一局殘差對上一局 h 的 OLS，以玩家為單位的 bootstrap；另排除列數最多的 1% 玩家看影響
    k = r1 if gname == "all" else r1[g1 == gi]
    pc = np.unique(pl_t[k], return_inverse=True)[1]
    P = pc.max() + 1
    x, y = h1[k], resid_t[k]
    S = np.stack([np.bincount(pc, None, P), np.bincount(pc, x, P), np.bincount(pc, y, P), np.bincount(pc, x * x, P), np.bincount(pc, x * y, P)])

    def ols(wt):
        n_, sx, sy, sxx, sxy = (wt @ S.T).T
        return (sxy - sx * sy / n_) / (sxx - sx**2 / n_)

    b0 = ols(np.ones(P))
    rng = np.random.default_rng(gi)
    bb = np.concatenate([ols(rng.multinomial(P, np.full(P, 1 / P), size=111).astype(float)) for _ in range(B_BOOT // 111)])
    top = np.bincount(pc) >= np.quantile(np.bincount(pc), 0.99)
    b_drop = ols((~top).astype(float))
    ctrl_rows.append({"speed": gname, "method": "ols_resid_player_bootstrap", "est": b0, "est_lo": np.quantile(bb, 0.025), "est_hi": np.quantile(bb, 0.975), "se": bb.std(), "n": len(k), "strata": P})
    ctrl_rows.append({"speed": gname, "method": "ols_drop_top1pct_players", "est": b_drop, "share_rows_dropped": np.bincount(pc)[top].sum() / len(k), "n": len(k), "strata": P})
ctrl = pl.DataFrame(ctrl_rows, infer_schema_length=None)
with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250):
    print(ctrl)
ctrl.write_csv("outputs/tables/arena_crossover_controls.csv")
