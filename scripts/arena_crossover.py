# arena 的 crossover 模型：被隨機分到較強的對手，是否改變同一場錦標賽中之後幾局的表現（carryover）。
# 處理 h = e_bar - e：期望匹配品質減本局的期望得分，正值為對手比依配對機制所期望的強；只含隨機配對產生的部分。
# 兩個目標量（規格見 docs/methods.md「arena 的 crossover 模型」）：
# - 總效果：殘差取主基準，含上一局經由下一局的對手與等級分更新的管道。以上一局所屬批內的重新配對作 permutation test，
#   統計量為主基準 offset 的有序模型在 beta = 0 下的一步估計（score 除以 information）；虛無模型不含 h，
#   每局的 score 在重新配對下不變，每次重抽只需重算 h。sharp null 是「之前的對手完全不影響之後那一局的紀錄」
# - 直接效果：固定下一局的對手。殘差取基準變數自由估計、另加自身等級分與下一局那一批 e_bar 的有序 logistic，
#   每位玩家算「上一局 h > 0 之後」減「h <= 0 之後」的平均殘差，對玩家做配對 t 檢定
# 以玩家加權為主、以局加權為次：crossover 樣本的活躍度極度集中，以局加權時有效玩家數只有數百

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
from arena_batches import e_grid, multi_null, player_sample_rd, player_slopes, repair
from calibrated import Q

B = 2000
B_BOOT = 999
CHUNK_DRAWS = 20
MIN_ROWS_SLOPE = 10
SPEEDS = ["Bullet", "Blitz", "Classical"]
GROUPS = SPEEDS + ["all"]
# 加權與批的組合。純隨機規則的批為敏感度分析：2014 年 12 人以下的批有小幅偏離隨機（docs/random-pairing.md）
COMBOS = [("game", "all"), ("game", "random_rule"), ("player", "all"), ("player", "random_rule")]
LAGS = [1, 2, 3]
NULLS = ["main", "unrestricted_before_1130"]
BAL = ["elo_c", "log_games_in_tour", "novice", "rd"]
C_GRID = np.round(np.arange(-0.5, 0.5001, 0.001), 3)


def holm(p):
    o = np.argsort(p)
    out = np.empty(len(p))
    out[o] = np.minimum(1, np.maximum.accumulate(p[o] * (len(p) - np.arange(len(p)))))
    return out


d = player_sample_rd().with_columns(
    y=pl.col("result").replace_strict({"loss": 0, "draw": 1, "win": 2}, return_dtype=pl.Int8),
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
        *[pl.col("rule").shift(k).over(grp).alias(f"rule_lag{k}") for k in LAGS],
        bid_lag1=pl.col("bid").shift(1).over(grp),
        score_lag1=pl.col("score").shift(1).over(grp),
        opp_lag1=pl.col("opponent").shift(1).over(grp),
    )
)
t = d.filter(pl.col("h_lag1").is_not_null())

sample = pl.concat([
    t.filter(pl.col(f"h_lag{k}").is_not_null())
    .group_by("speed", layer=pl.col(f"layer_lag{k}"), rule=pl.col(f"rule_lag{k}"))
    .agg(rows=pl.len(), players=pl.col("player").n_unique(), tournaments=pl.col("tournament_id").n_unique())
    .with_columns(lag=pl.lit(k))
    for k in LAGS
]).sort("lag", "speed", "layer", "rule")
print(sample)
sample.write_csv("outputs/tables/arena_crossover_sample.csv")


# 總效果的虛無模型只有兩個參數（位移 alpha、和局帶寬的對數倍率 delta），以 numpy 估計：
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
for j, spd in enumerate(SPEEDS):
    k = sp_t == j
    r = optimize.minimize(nll_null, [0.0, 0.0], args=(off_t[k], wo_t[k], y_t[k]), method="BFGS", options={"gtol": 1e-10})
    hi, nl = ord_parts(off_t[k], wo_t[k], *r.x)
    yk = y_t[k]
    s_t[k] = np.where(yk == 2, 1 - hi, np.where(yk == 0, -nl, 1 - hi - nl))
    v_t[k] = np.where(yk == 2, hi * (1 - hi), np.where(yk == 0, nl * (1 - nl), hi * (1 - hi) + nl * (1 - nl)))
    print(spd, "null", r.x, s_t[k].sum())
resid_t, score_lag1 = t["resid"].to_numpy(), t["score_lag1"].to_numpy()
pl_t, opn_t, opl_t = (t[c].to_numpy() for c in ["player", "opponent", "opp_lag1"])
opp_next = ((t["opp_elo"] - 1500) / 100).to_numpy()
e_next = t["e_base"].to_numpy()
rr_m = (m["rule"] == "random").to_numpy()
sp_m = m["sp"].to_numpy()


# 每列的權重：以局加權為 1；以玩家加權為 1 / (該玩家在該速度類別、該樣本中的列數)。純隨機規則的子樣本另算列數
def weights(player, sp, rr):
    key = player.astype(np.int64) * len(SPEEDS) + sp
    n_all = np.unique(key, return_inverse=True, return_counts=True)
    n_rr = np.unique(key[rr], return_inverse=True, return_counts=True)
    w_rr = np.zeros(len(key))
    w_rr[rr] = 1 / n_rr[2][n_rr[1]]
    return [np.ones(len(key)), rr.astype(float), 1 / n_all[2][n_all[1]], w_rr]


# 各 lag 的隨機化樣本：第 k 局之前那一局屬於兩局以上的批
R = {}
for k in LAGS:
    mk = t[f"midx_lag{k}"].fill_null(-1).to_numpy()
    rows = np.flatnonzero(mk >= 0)
    g = sp_t[rows]
    R[k] = (rows, mk[rows], g, weights(pl_t[rows], g, rr_m[mk[rows]]))
W_m = weights(m["player"].to_numpy(), sp_m, rr_m)
# 劑量反應的五等分：切點取實際的 h（依速度類別），重新配對時固定
r1, m1, g1, _ = R[1]
h1 = t["h_lag1"].to_numpy()
cuts = np.stack([np.quantile(h1[r1][g1 == j], [0.2, 0.4, 0.6, 0.8]) for j in range(len(SPEEDS))])
# 平衡檢驗：配對前已定的變數在批內置中，取其對 h 的斜率，在全部兩局以上的批的列上
Zc = []
for c in BAL:
    z = m[c].cast(pl.Float64).to_numpy()
    Zc.append(z - (np.bincount(bi, z, nb) / np.bincount(bi, minlength=nb))[bi])


def agg(num, g, W):
    out = []
    for wv in W:
        a = np.bincount(g, num * wv, len(SPEEDS))
        out.append(np.r_[a, a.sum()])
    return np.concatenate(out)


def stats_of(hm):
    out = {}
    for k in LAGS:
        rows, mk, g, W = R[k]
        h = hm[mk]
        out[f"beta_lag{k}"] = agg(h * s_t[rows], g, W) / agg(h * h * v_t[rows], g, W)
    rows, mk, g, W = R[1]
    h = hm[mk]
    qs = np.empty(len(h))
    for j in range(len(SPEEDS)):
        qs[g == j] = np.searchsorted(cuts[j], h[g == j]) - 2.0
    out["trend_quintile"] = agg(qs * s_t[rows], g, W) / agg(qs * qs * v_t[rows], g, W)
    hp, hn = np.maximum(h, 0), np.minimum(h, 0)
    bp = agg(hp * s_t[rows], g, W) / agg(hp * hp * v_t[rows], g, W)
    bn = agg(hn * s_t[rows], g, W) / agg(hn * hn * v_t[rows], g, W)
    out["beta_stronger"], out["beta_weaker"], out["asymmetry"] = bp, bn, bp - bn
    # 下一局的管道：sharp null 下下一局的對手與基準的期望得分也不隨上一局的配對而變
    hh = agg(h * h, g, W)
    out["next_opp_elo"] = agg(h * opp_next[rows], g, W) / hh
    out["next_e_base"] = agg(h * e_next[rows], g, W) / hh
    # Anderson–Rubin 式的區間：H0(c) 下 resid - c * score_lag1 不隨上一局的配對而變
    out["ar_a"] = agg(h * resid_t[rows], g, W)
    out["ar_b"] = agg(h * score_lag1[rows], g, W)
    for c, z in zip(BAL, Zc):
        out[f"balance_{c}"] = agg(hm * z, sp_m, W_m) / agg(hm * hm, sp_m, W_m)
    return out


act = stats_of(h_act_m)
KEYS = list(act)
NG = len(COMBOS) * len(GROUPS)


def draws(job):
    # 每次重新配對有自己的種子：平行時結果不隨工作分配到哪個行程而變
    kn, i0 = job
    out = []
    for i in range(i0, i0 + CHUNK_DRAWS):
        st = stats_of(e_bar_m - E[tci, ui, ui[repair(ns, NULLS[kn], np.random.default_rng([kn, i]))]])
        out.append(np.concatenate([st[c] for c in KEYS]))
    print(kn, i0 + CHUNK_DRAWS, "draws", flush=True)
    return np.array(out)


# 單次重新配對是單執行緒的 numpy 運算；每個工作一次跑一批重抽，避免切得太碎。
# 兩個虛無分布的工作一起送出，最後一波才不會只剩少數行程在跑
jobs = [(kn, i) for kn in range(len(NULLS)) for i in range(0, B, CHUNK_DRAWS)]
with ProcessPoolExecutor(os.cpu_count() - 2, mp_context=mp.get_context("fork")) as ex:
    res_jobs = list(ex.map(draws, jobs))
nd = {nl: np.concatenate([r_ for (kn, _), r_ in zip(jobs, res_jobs) if kn == knl]) for knl, nl in enumerate(NULLS)}


def label(gi):
    return {"weighting": COMBOS[gi // len(GROUPS)][0], "batches": COMBOS[gi // len(GROUPS)][1], "speed": GROUPS[gi % len(GROUPS)]}


slope_elo = t.group_by("speed").agg(pl.col("slope_elo").mean())
slope_elo = dict(zip(slope_elo["speed"], slope_elo["slope_elo"])) | {"all": t["slope_elo"].mean()}
ri_rows, ar_rows = [], []
for nl in NULLS:
    arr = nd[nl].reshape(B, len(KEYS), NG)
    for ki, c in enumerate(KEYS):
        if c.startswith("ar_"):
            continue
        for gi in range(NG):
            a, x = act[c][gi], arr[:, ki, gi]
            ri_rows.append({
                "null": nl, "stat": c, **label(gi), "est": a, "null_mean": x.mean(), "null_sd": x.std(), "est_centered": a - x.mean(),
                "p": (1 + (np.abs(x - x.mean()) >= abs(a - x.mean())).sum()) / (B + 1),
            })
    ia, ib = KEYS.index("ar_a"), KEYS.index("ar_b")
    for gi in range(NG):
        Ta = act["ar_a"][gi] - C_GRID * act["ar_b"][gi]
        Tn = arr[:, ia, gi][:, None] - C_GRID[None, :] * arr[:, ib, gi][:, None]
        mu = Tn.mean(0)
        p = (1 + (np.abs(Tn - mu) >= np.abs(Ta - mu)).sum(0)) / (B + 1)
        keep = C_GRID[p > 0.05]
        ar_rows.append({
            "null": nl, **label(gi), "ratio": act["ar_a"][gi] / act["ar_b"][gi],
            "ci_lo": keep.min() if len(keep) else np.nan, "ci_hi": keep.max() if len(keep) else np.nan,
            "ci_at_grid_edge": bool(len(keep) and (keep.min() == C_GRID[0] or keep.max() == C_GRID[-1])),
            "ci_connected": bool(len(keep) and len(keep) == int(round((keep.max() - keep.min()) / 0.001)) + 1),
            "p_zero": p[C_GRID == 0][0],
        })
ri = pl.DataFrame(ri_rows)
# 總效果的主要假設：Bullet 與 Blitz 的第一至三局 carryover（主虛無分布、以玩家加權、全部兩局以上的批），Holm 法
main_sel = (pl.col("null") == "main") & (pl.col("weighting") == "player") & (pl.col("batches") == "all") & pl.col("stat").str.starts_with("beta_lag") & pl.col("speed").is_in(["Bullet", "Blitz"])
fam = ri.filter(main_sel)
ri = ri.join(fam.select("null", "weighting", "batches", "stat", "speed", p_holm=pl.Series(holm(fam["p"].to_numpy()))), on=["null", "weighting", "batches", "stat", "speed"], how="left")
# 平衡檢驗為一族（主虛無分布、Bullet 與 Blitz、全部加權與批的組合），以 Holm 法校正後仍拒絕才觸發調整
bal = ri.filter(pl.col("null") == "main", pl.col("stat").str.starts_with("balance_"), pl.col("speed").is_in(["Bullet", "Blitz"]))
ri = ri.join(bal.select("null", "weighting", "batches", "stat", "speed", p_holm_diag=pl.Series(holm(bal["p"].to_numpy()))), on=["null", "weighting", "batches", "stat", "speed"], how="left")
# 換算：h 增加 0.1 的效果相當於自身等級分高多少分（以主基準的分差斜率換算）
ri = ri.with_columns(
    elo_per_0_1=pl.when(pl.col("stat").str.starts_with("beta_")).then(pl.col("est_centered") * 0.1 / pl.col("speed").replace_strict(slope_elo, return_dtype=pl.Float64))
)
with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250):
    print(ri.filter(pl.col("null") == "main", pl.col("speed") != "Classical"))
ri.write_csv("outputs/tables/arena_crossover_ri.csv")
ar = pl.DataFrame(ar_rows)
ar.write_csv("outputs/tables/arena_crossover_ratio_total.csv")
print(ar.filter(pl.col("null") == "main"))


# 直接效果的有序 logistic。t[0] 截距，t[1:NX+1] 各共變數，t[NX+1:NX+4] 和局帶寬的對數（常數、平均等級分、其平方），
# t[NX+4] partial proportional odds：分差對「勝」門檻的斜率位移。未列入 free 的參數固定為 0
t = t.with_columns(
    sign=pl.when(pl.col("color") == "white").then(1.0).otherwise(-1.0),
    xr=((pl.col("elo") + pl.col("opp_elo")) / 2 - 1500) / 100,
    xd=((pl.col("elo") - pl.col("opp_elo")) / 100).cast(pl.Float64),
).with_columns(
    sxr=pl.col("sign") * pl.col("xr"),
    sxr2=pl.col("sign") * pl.col("xr") ** 2,
    xdxr=pl.col("xd") * pl.col("xr"),
    gxd=pl.col("xd") / (1 + 3 * Q**2 * (pl.col("rd") ** 2 + pl.col("opp_rd") ** 2) / np.pi**2).sqrt(),
    elo_c2=pl.col("elo_c") ** 2,
    elo_c3=pl.col("elo_c") ** 3,
    ebar_xd=pl.col("e_bar") * pl.col("xd"),
    h_ebar=pl.col("h_lag1") * pl.col("e_bar"),
)
# 本場已下局數：12 人以下的批優先配對等待最久者，它與 h 的平衡不通過
BASE = ["sign", "sxr", "sxr2", "xd", "xdxr", "gxd", "elo_c", "elo_c2", "log_games_in_tour", "e_bar"]
EXTRA = ["h_lag1", "h_lag2", "h_lag3", "elo_c3", "ebar_xd", "h_ebar"]
XCOLS = BASE + EXTRA
NX = len(XCOLS)
P_DRAW, P_PPO = NX + 1, NX + 4


def ll(th, X, xr, xd, y):
    eta = th[0] + X @ th[1 : NX + 1]
    w = jnp.exp(th[P_DRAW] + th[P_DRAW + 1] * xr + th[P_DRAW + 2] * xr**2)
    u, v = eta + w, eta - w + th[P_PPO] * xd
    ldraw = jax.nn.log_sigmoid(u) + jax.nn.log_sigmoid(-v) + jnp.log(-jnp.expm1(-(u - v)))
    return jnp.where(y == 2, jax.nn.log_sigmoid(v), jnp.where(y == 0, jax.nn.log_sigmoid(-u), ldraw))


def free_of(cols, ppo=False):
    return np.r_[0, [1 + XCOLS.index(c) for c in cols], P_DRAW, P_DRAW + 1, P_DRAW + 2, [P_PPO] if ppo else []].astype(int)


T0 = np.zeros(NX + 5)
T0[P_DRAW] = -2.5


def args_of(s):
    X = s.select(XCOLS).fill_null(0.0).to_numpy().astype(np.float64)
    return (X, s["xr"].to_numpy(), s["xd"].to_numpy(), s["y"].to_numpy().astype(np.float64))


def lrt(s, cols0, cols1, ppo1=False):
    f1 = free_of(cols1, ppo1)
    th1, V, res = lk.lrt_player_dyad(ll, T0, free_of(cols0), f1, args_of(s), s["player"].to_numpy(), s["opponent"].to_numpy())
    return th1, f1, V, res


# 玩家內的對比：每位玩家「h > 0 之後」減「h <= 0 之後」的平均殘差，對玩家做配對 t 檢定。
# 同一局的兩位玩家（本局或上一局的對手）分屬兩個差值，以「有共同對局的玩家對」的共變數核對標準誤
def contrast(r, h, player, opp_a, opp_b):
    df = pl.DataFrame({"p": player, "pos": np.where(h > 0, "s", "w"), "r": r, "h": h})
    g = df.group_by("p", "pos").agg(r=pl.col("r").mean(), h=pl.col("h").mean()).pivot(on="pos", index="p", values=["r", "h"]).drop_nulls()
    dp = (g["r_s"] - g["r_w"]).to_numpy()
    dh = (g["h_s"] - g["h_w"]).to_numpy()
    P = len(dp)
    e = dp - dp.mean()
    pos = dict(zip(g["p"].to_list(), range(P)))
    links = set()
    for a, b in zip(np.r_[player, player], np.r_[opp_a, opp_b]):
        if a in pos and b in pos and a != b:
            links.add((min(a, b), max(a, b)))
    li = np.array([(pos[a], pos[b]) for a, b in links]).reshape(-1, 2)
    se_t = dp.std(ddof=1) / np.sqrt(P)
    se_link = np.sqrt(max((e**2).sum() + 2 * (e[li[:, 0]] * e[li[:, 1]]).sum(), 0)) / P
    se = max(se_t, se_link)
    tt = dp.mean() / se
    return {"players": P, "diff": dp.mean(), "se_t": se_t, "se_link": se_link, "t": tt, "p": 2 * stats.t.sf(abs(tt), P - 1), "h_diff": dh.mean(), "effect_per_h": dp.mean() / dh.mean()}


con_rows, slope_rows, game_rows, spec_rows, dose_rows, ratio_rows = [], [], [], [], [], []
rng = np.random.default_rng(1)
sc_t = t["score"].to_numpy()
for j, spd in enumerate(SPEEDS):
    ts = t.filter(pl.col("speed") == spd)
    a_s = args_of(ts)
    # 直接效果的殘差：不含 h 的模型，以該速度類別全部有上一局的列估計
    th0, *_ = lk.fit(ll, T0, free_of(BASE), a_s, ts["player"].to_numpy(), ts["opponent"].to_numpy())
    X, xr, xd, _ = a_s
    eta = th0[0] + X @ th0[1 : NX + 1]
    w = np.exp(th0[P_DRAW] + th0[P_DRAW + 1] * xr + th0[P_DRAW + 2] * xr**2)
    e_dir = (1 / (1 + np.exp(-(eta - w))) + 1 / (1 + np.exp(-(eta + w)))) / 2
    ts = ts.with_columns(r_dir=pl.Series(ts["score"].to_numpy() - e_dir), r_tot=pl.col("resid"))
    spe = ts["score_per_elo"].mean()
    for k in LAGS:
        for sample_name, keep in [
            ("multi", pl.col(f"layer_lag{k}") == "multi"),
            ("multi_random_rule", (pl.col(f"layer_lag{k}") == "multi") & (pl.col(f"rule_lag{k}") == "random")),
            ("single", pl.col(f"layer_lag{k}") == "single"),
        ]:
            s = ts.filter(keep, pl.col(f"h_lag{k}").is_not_null())
            hk = s[f"h_lag{k}"].to_numpy()
            for eff in ["direct", "total"]:
                r = s[f"r_{'dir' if eff == 'direct' else 'tot'}"].to_numpy()
                c = contrast(r, hk, s["player"].to_numpy(), s["opponent"].to_numpy(), s["opp_lag1"].to_numpy())
                con_rows.append({"speed": spd, "lag": k, "sample": sample_name, "effect": eff, **c, "elo_per_0_1": 0.1 * c["effect_per_h"] / spe})
                slope_rows.append({"speed": spd, "lag": k, "sample": sample_name, "effect": eff, **player_slopes(r, hk, s["player"].to_numpy(), MIN_ROWS_SLOPE)})
    print(spd, "contrasts", flush=True)

    s = ts.filter(pl.col("layer_lag1") == "multi")
    # 以局加權的直接效果：完整模型的係數與修正的概似比檢定
    th1, f1, V, res = lrt(s, BASE, BASE + ["h_lag1"])
    jh = int(np.flatnonzero(f1 == 1 + XCOLS.index("h_lag1"))[0])
    # 以玩家為單位的 bootstrap：一步估計在虛無模型下的 score 與 information，逐玩家加總後重抽玩家
    X, xr, xd, y = args_of(s)
    th0s, *_ = lk.fit(ll, T0, free_of(BASE), (X, xr, xd, y), s["player"].to_numpy(), s["opponent"].to_numpy())
    eta = th0s[0] + X @ th0s[1 : NX + 1]
    w = np.exp(th0s[P_DRAW] + th0s[P_DRAW + 1] * xr + th0s[P_DRAW + 2] * xr**2)
    hi, nl = 1 / (1 + np.exp(-(eta - w))), 1 / (1 + np.exp(-(eta + w)))
    sc = np.where(y == 2, 1 - hi, np.where(y == 0, -nl, 1 - hi - nl))
    vv = np.where(y == 2, hi * (1 - hi), np.where(y == 0, nl * (1 - nl), hi * (1 - hi) + nl * (1 - nl)))
    hs = s["h_lag1"].to_numpy()
    pc = np.unique(s["player"].to_numpy(), return_inverse=True)[1]
    P = pc.max() + 1
    U, I = np.bincount(pc, hs * sc, P), np.bincount(pc, hs * hs * vv, P)
    cnt = np.vstack([rng.multinomial(P, np.full(P, 1 / P), size=111) for _ in range(B_BOOT // 111)]).astype(float)
    bb = (cnt @ U) / (cnt @ I)
    share = np.bincount(pc) / len(pc)
    game_rows.append({
        "speed": spd, "n": s.height, "players": P, "effective_players": 1 / (share**2).sum(),
        "beta": th1[1 + XCOLS.index("h_lag1")], "se_dyadic": np.sqrt(V[jh, jh]), "one_step": U.sum() / I.sum(),
        "boot_lo": np.quantile(bb, 0.025), "boot_hi": np.quantile(bb, 0.975), "boot_se": bb.std(),
        "elo_per_0_1": th1[1 + XCOLS.index("h_lag1")] * 0.1 / s["slope_elo"].mean(), **{f"lrt_{a}": b for a, b in res.items()},
    })

    # 設定檢定：在含 h_lag1 的模型上逐一加入一項
    full = BASE + ["h_lag1"]
    for name, cols1, ppo in [("elo_cubic", full + ["elo_c3"], False), ("ebar_x_diff", full + ["ebar_xd"], False), ("h_x_ebar", full + ["h_ebar"], False), ("partial_po", full, True)]:
        _, _, _, res = lrt(s, full, cols1, ppo)
        spec_rows.append({"speed": spd, "check": name, **res})
    # 依序加入 lag 的概似比檢定，在三局之前都有、且上一局屬於兩局以上的批的列上
    s3 = s.filter(pl.col("h_lag3").is_not_null())
    for name, c0, c1 in [("add_lag1", BASE, BASE + ["h_lag1"]), ("add_lag2", BASE + ["h_lag1"], BASE + ["h_lag1", "h_lag2"]), ("add_lag3", BASE + ["h_lag1", "h_lag2"], BASE + ["h_lag1", "h_lag2", "h_lag3"])]:
        _, _, _, res = lrt(s3, c0, c1)
        spec_rows.append({"speed": spd, "check": name, **res})
    print(spd, "models", flush=True)

    # 劑量反應：上一局 h 的五等分下的平均殘差，以玩家加權（每位玩家在每一等分內的平均，再對玩家平均）
    s = s.with_columns(q=pl.Series(np.searchsorted(cuts[j], s["h_lag1"].to_numpy()) + 1))
    for eff in ["direct", "total"]:
        col = "r_dir" if eff == "direct" else "r_tot"
        pq = s.group_by("q", "player").agg(r=pl.col(col).mean(), h=pl.col("h_lag1").mean(), n=pl.len())
        z = pq.group_by("q").agg(
            players=pl.len(), h_mean=pl.col("h").mean(), resid_mean=pl.col("r").mean(), resid_se=pl.col("r").std() / pl.len().sqrt(),
            resid_mean_game=(pl.col("r") * pl.col("n")).sum() / pl.col("n").sum(),
        ).sort("q")
        dose_rows.append(z.with_columns(speed=pl.lit(spd), effect=pl.lit(eff)))

    # 輸棋本身的效果（直接效果）：以玩家加權的 cov(h, r_dir) / cov(h, score_lag1)，delta method；
    # 影響函數的 meat 以「玩家 × 上一局的對手」與「玩家 × 本局的對手」兩種 dyad 各算，取較大者
    wv = 1 / np.bincount(pc)[pc]
    hc = hs - (wv * hs).sum() / wv.sum()
    rd_, dl = s["r_dir"].to_numpy(), s["score_lag1"].to_numpy()
    den = (wv * hc * dl).sum()
    c = (wv * hc * rd_).sum() / den
    phi = (wv * hc * (rd_ - c * dl) / den)[:, None]
    se = max(np.sqrt(lk.dyad_meat(phi, s["player"].to_numpy(), s["opp_lag1"].to_numpy())[0, 0]), np.sqrt(lk.dyad_meat(phi, s["player"].to_numpy(), s["opponent"].to_numpy())[0, 0]))
    ratio_rows.append({"speed": spd, "ratio": c, "se_delta": se, "lo": c - 1.96 * se, "hi": c + 1.96 * se})

con = pl.DataFrame(con_rows)
# 直接效果的主要假設：Bullet 與 Blitz 的第一至三局（兩局以上的批），Holm 法
msk = con.select((pl.col("effect") == "direct") & (pl.col("sample") == "multi") & pl.col("speed").is_in(["Bullet", "Blitz"])).to_series().to_numpy()
ph = np.full(con.height, np.nan)
ph[msk] = holm(con["p"].to_numpy()[msk])
con = con.with_columns(p_holm=pl.Series(ph))
with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250):
    print(con)
con.write_csv("outputs/tables/arena_crossover_contrast.csv")
pl.DataFrame(slope_rows).write_csv("outputs/tables/arena_crossover_player_slope.csv")
game = pl.DataFrame(game_rows)
game.write_csv("outputs/tables/arena_crossover_game_weighted.csv")
spec = pl.DataFrame(spec_rows)
# 設定檢定為一族，以 Holm 法校正；依序加入 lag 是效果的檢定，不在族內
diag = spec["check"].is_in(["elo_cubic", "ebar_x_diff", "h_x_ebar", "partial_po"]).to_numpy()
pd_ = np.full(spec.height, np.nan)
pd_[diag] = holm(spec["p"].to_numpy()[diag])
spec = spec.with_columns(p_holm_diag=pl.Series(pd_))
spec.write_csv("outputs/tables/arena_crossover_spec.csv")
pl.DataFrame(ratio_rows).write_csv("outputs/tables/arena_crossover_ratio_direct.csv")
dose = pl.concat(dose_rows)
dose.write_csv("outputs/tables/arena_crossover_dose.csv")
with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250):
    print(game)
    print(spec)
    print(pl.DataFrame(ratio_rows))

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
for ax, spd in zip(axes, ["Bullet", "Blitz"]):
    for eff, mk in [("direct", "o"), ("total", "s")]:
        z = dose.filter(pl.col("speed") == spd, pl.col("effect") == eff)
        ax.errorbar(z["h_mean"], z["resid_mean"], yerr=1.96 * z["resid_se"], marker=mk, capsize=3, label=f"{eff} effect")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title(spd)
    ax.set_xlabel("previous opponent strength h (expected score units)")
axes[0].set_ylabel("next-game score minus expected (player-weighted)")
axes[0].legend()
fig.tight_layout()
fig.savefig("outputs/figures/arena_crossover_dose.png", dpi=150)

# 對照：以上一局的批分層的 CMH 相關統計量與 Mantel–Haenszel 共同 odds ratio（下一局輸 × 上一局對手比期望強）；
# 只用上一局屬於兩局以上的批的列，批內至少兩列。反應取主基準的殘差與原始得分（0、0.5、1，即有序 CMH，含下一局的管道），
# MH 的「輸」是原始結果。以局加權
ctrl_rows = []
bid1 = t["bid_lag1"].to_numpy()
for gi, gname in enumerate(GROUPS):
    k = r1 if gname == "all" else r1[g1 == gi]
    st = np.unique(bid1[k], return_inverse=True)[1]
    n = np.bincount(st)
    ok = n[st] >= 2
    k, st = k[ok], np.unique(st[ok], return_inverse=True)[1]
    n = np.bincount(st)
    x = h1[k]
    xc = x - (np.bincount(st, x) / n)[st]
    for outcome, yv in [("resid", resid_t[k]), ("raw_score", sc_t[k])]:
        yc = yv - (np.bincount(st, yv) / n)[st]
        T = (xc * yc).sum()
        var = (np.bincount(st, xc * xc) * np.bincount(st, yc * yc) / (n - 1)).sum()
        ctrl_rows.append({"speed": gname, "method": f"stratified_correlation_{outcome}", "stat": T**2 / var, "est": T / np.bincount(st, xc * xc).sum(), "p": stats.chi2.sf(T**2 / var, 1), "n": len(k), "strata": len(n)})
    lo, st_ = (sc_t[k] == 0).astype(int), (x > 0).astype(int)
    tab = np.zeros((2, 2, len(n)))
    np.add.at(tab, (st_, lo, st), 1)
    keep = (tab.sum((0, 1)) >= 2) & (tab.sum(1) > 0).all(0) & (tab.sum(0) > 0).all(0)
    stt = StratifiedTable(tab[:, :, keep])
    ci = stt.oddsratio_pooled_confint()
    tn = stt.test_null_odds()
    ctrl_rows.append({"speed": gname, "method": "mh_odds_ratio_loss_stronger", "stat": tn.statistic, "est": stt.oddsratio_pooled, "est_lo": ci[0], "est_hi": ci[1], "p": tn.pvalue, "n": int(tab[:, :, keep].sum()), "strata": int(keep.sum())})
ctrl = pl.DataFrame(ctrl_rows, infer_schema_length=None)
with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250):
    print(ctrl)
ctrl.write_csv("outputs/tables/arena_crossover_controls.csv")
