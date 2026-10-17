# 先彙總到玩家再用課內方法，能否取代 sandwich（規格見 docs/player-aggregation.md）。
# 玩家之間近似獨立時，以局加權的平均之差 = 玩家總和的比值，標準誤由「玩家為獨立單位、比值的 delta method」得到，
# 代數上等於以玩家為群的 sandwich 乘 sqrt(m / (m - 1))；以玩家等權時即一樣本或配對 t 檢定。
# 兩者都忽略「同一局的兩位玩家」的相依，以 dyadic sandwich 核對。比較只在估計目標相同的列上算標準誤比值

import sys

sys.path.insert(0, "scripts")

import numpy as np
import polars as pl
from scipy import stats

import likelihood as lk
from arena_batches import STOP_RULES, continuation_sample, player_sample_rd, player_slopes, rating_filter_sample, stop_rule_sample

SPEEDS = ["Bullet", "Blitz", "Classical"]


def holm(p):
    o = np.argsort(p)
    out = np.empty(len(p))
    out[o] = np.minimum(1, np.maximum.accumulate(p[o] * (len(p) - np.arange(len(p)))))
    return out


# c == 1 的平均減 c == 0 的平均；沒有 c == 0 的列時為一樣本的平均。回傳點估計、課內方法、玩家分群與 dyadic 的標準誤。
# 每列的影響函數 phi 依玩家加總即玩家層級的偏差，平方和是玩家分群的 sandwich；dyad_meat 另計共有對局的玩家之間的共變數
def compare(y, c, player, opp, weighting):
    cells = [k for k in (1, 0) if (c == k).any()]
    sign = {1: 1.0, 0: -1.0}
    pid, inv = np.unique(player, return_inverse=True)
    M = len(pid)
    S = {k: np.bincount(inv, y * (c == k), M) for k in cells}
    N = {k: np.bincount(inv, c == k, M).astype(float) for k in cells}
    if weighting == "game":
        R = {k: S[k].sum() / N[k].sum() for k in cells}
        est = sum(sign[k] * R[k] for k in cells)
        phi = sum(np.where(c == k, sign[k] * (y - R[k]) / N[k].sum(), 0.0) for k in cells)
        keep = np.ones(len(y), bool)
        # 課內方法：玩家的 (S_k, N_k) 為獨立同分布的向量，平均的比值以 delta method 求變異數
        used = np.ones(M, bool)
        Z = np.column_stack([v for k in cells for v in (S[k], N[k])])
        g = np.concatenate([[sign[k] / N[k].mean(), -sign[k] * R[k] / N[k].mean()] for k in cells])
        se_course = np.sqrt(g @ np.cov(Z, rowvar=False) @ g / M)
        df = M - 1
    else:
        used = np.all([N[k] > 0 for k in cells], 0)
        d_i = sum(sign[k] * S[k][used] / N[k][used] for k in cells)
        est = d_i.mean()
        m = used.sum()
        keep = used[inv]
        # 玩家的偏差 d_i - D 分成兩部分攤到各列：列對自己格內平均的離差，與平均攤到每列的 d_i - D。
        # 只攤未置中的 y 時，依對手加總的 dyadic 項會累積各列的水準而膨脹
        n_tot = np.maximum(sum(N[k] for k in cells), 1)
        dev = np.zeros(M)
        dev[used] = d_i - est
        within = sum(np.where(c == k, sign[k] * (y - (S[k] / np.maximum(N[k], 1))[inv]) / np.maximum(N[k][inv], 1), 0.0) for k in cells)
        phi = np.where(keep, (within + (dev / n_tot)[inv]) / m, 0.0)
        se_course = d_i.std(ddof=1) / np.sqrt(m)
        df = m - 1
    z = np.bincount(inv[keep], phi[keep], M)
    se_cluster = np.sqrt((z**2).sum())
    se_dyadic = np.sqrt(lk.dyad_meat(phi[keep][:, None], player[keep], opp[keep])[0, 0])
    share = np.bincount(inv[keep], None, M)[used] / keep.sum()
    return {
        "weighting": weighting, "n": int(keep.sum()), "players": int(used.sum()),
        "n_eff": 1 / (share**2).sum() if weighting == "game" else float(used.sum()),
        "est": est, "se_course": se_course, "se_cluster": se_cluster, "se_dyadic": se_dyadic,
        "p_course": 2 * stats.t.sf(abs(est / se_course), df), "p_dyadic": 2 * stats.norm.sf(abs(est / se_dyadic)),
    }


rows = []

# 停止規則的回測：刪去減保留。現行做法為以局加權、玩家 block bootstrap
d, del_days = stop_rule_sample()
cur = pl.read_csv("outputs/tables/folk_stop_rules_backtest.csv")
for r in STOP_RULES:
    for s in SPEEDS:
        for grp in ["all", "novice", "established"]:
            x = d.filter(pl.col("speed") == s)
            if grp != "all":
                x = x.filter(pl.col("novice_rd") == (grp == "novice"))
            cr = cur.filter(pl.col("rule") == r, pl.col("speed") == s, pl.col("group") == grp).row(0, named=True)
            for wt in ["game", "player"]:
                o = compare(x["resid"].to_numpy(), x[f"del_{r}"].to_numpy().astype(int), x["player"].to_numpy(), x["opponent"].to_numpy(), wt)
                rows.append({"analysis": "stop_rule_backtest", "cell": f"{r}|{s}|{grp}", "speed": s,
                             "family": grp == "all" and s in ("Bullet", "Blitz"), **o,
                             "est_current": cr["diff"], "se_current": cr["se"], "p_current": cr["p"], "p_holm_current": cr["p_holm"]})

# 隔天重置：隔天第一局減刪去的局，現行做法同上
cur = pl.read_csv("outputs/tables/folk_stop_rules_reset.csv")
nd = del_days.filter(pl.col("next_resid").is_not_null())
for r in STOP_RULES:
    for s in SPEEDS:
        a = d.filter(pl.col(f"del_{r}"), pl.col("speed") == s)
        b = nd.filter(pl.col(f"any_{r}"), pl.col("next_speed") == s)
        y = np.r_[b["next_resid"].to_numpy(), a["resid"].to_numpy()]
        c = np.r_[np.ones(b.height, int), np.zeros(a.height, int)]
        pl_ = np.r_[b["player"].to_numpy(), a["player"].to_numpy()]
        op = np.r_[b["next_opponent"].to_numpy(), a["opponent"].to_numpy()]
        cr = cur.filter(pl.col("rule") == r, pl.col("speed") == s).row(0, named=True)
        for wt in ["game", "player"]:
            rows.append({"analysis": "stop_rule_reset", "cell": f"{r}|{s}", "speed": s, "family": False, **compare(y, c, pl_, op, wt),
                         "est_current": cr["diff"], "se_current": cr["se"], "p_current": cr["p"], "p_holm_current": None})

# 剛輸之後的分層校準：每格的平均殘差是否為 0。現行做法檢驗 Cox 的 alpha = 0、beta = 1，只是部分對應，沒有可比的標準誤
t = rating_filter_sample()
chk = t.group_by("speed", "rdq", "prev_result").agg(n=pl.len(), resid_mean=pl.col("resid").mean()).with_columns(pl.col("rdq") + 1)
ref = pl.read_csv("outputs/tables/rating_filter_error_sample.csv").with_columns(pl.col("rdq") + 1)
mm = chk.join(ref, on=["speed", "rdq", "prev_result"])
assert mm.height == ref.height and (mm["n"] == mm["n_right"]).all() and np.allclose(mm["resid_mean"], mm["resid_mean_right"])
cur = pl.read_csv("outputs/tables/rating_filter_error_cells.csv")
for row in cur.iter_rows(named=True):
    x = t.filter(pl.col("speed") == row["speed"], pl.col("rdq") == row["rdq"] - 1, pl.col("prev_result") == row["prev_result"])
    for wt in ["game", "player"]:
        o = compare(x["resid"].to_numpy(), np.ones(x.height, int), x["player"].to_numpy(), x["opponent"].to_numpy(), wt)
        rows.append({"analysis": "rating_filter_cell", "cell": f"{row['speed']}|rdq{row['rdq']}|{row['prev_result']}", "speed": row["speed"],
                     "family": row["speed"] in ("Bullet", "Blitz"), **o,
                     "est_current": row["alpha"], "se_current": None, "p_current": row["p"], "p_holm_current": row["p_holm"]})

# 續下的總效果：h > 0 與 h <= 0 之後的續下率差。現行做法為 logit 的係數，尺度不同，只比結論
cs = continuation_sample([1])
ref = pl.read_csv("outputs/tables/continuation_selection_sample.csv").filter(pl.col("lag") == 1)
chk = cs.filter(pl.col("cont1").is_not_null()).group_by("speed", "layer").agg(rows=pl.len(), cont_rate=pl.col("cont1").mean())
mm = chk.join(ref, on=["speed", "layer"])
assert mm.height == ref.height and (mm["rows"] == mm["rows_right"]).all() and np.allclose(mm["cont_rate"], mm["cont_rate_right"])
cur = pl.read_csv("outputs/tables/continuation_selection_tests.csv").filter(pl.col("model") == "total")
for row in cur.iter_rows(named=True):
    x = cs.filter(pl.col("speed") == row["speed"], pl.col("layer") == row["layer"], pl.col("cont1").is_not_null())
    for wt in ["game", "player"]:
        o = compare(x["cont1"].cast(pl.Float64).to_numpy(), (x["h"] > 0).cast(pl.Int64).to_numpy(), x["player"].to_numpy(), x["opponent"].to_numpy(), wt)
        rows.append({"analysis": "continuation_total", "cell": f"{row['speed']}|{row['layer']}", "speed": row["speed"],
                     "family": row["speed"] in ("Bullet", "Blitz"), **o,
                     "est_current": None, "se_current": None, "p_current": row["p"], "p_holm_current": row["p_holm"]})

# crossover 的直接效果與總效果：現行即是玩家等權的配對 t 檢定，se_link 為共有對局的玩家對的共變數修正，取兩者較大者
cur = pl.read_csv("outputs/tables/arena_crossover_contrast.csv").filter(pl.col("sample") == "multi")
for row in cur.iter_rows(named=True):
    m = row["players"]
    est, se_t = row["diff"], row["se_t"]
    se_cur = max(se_t, row["se_link"])
    rows.append({"analysis": f"crossover_{row['effect']}", "cell": f"{row['speed']}|lag{row['lag']}", "speed": row["speed"],
                 "family": row["effect"] == "direct" and row["speed"] in ("Bullet", "Blitz"),
                 "weighting": "player", "n": None, "players": m, "n_eff": float(m),
                 "est": est, "se_course": se_t, "se_cluster": se_t * np.sqrt((m - 1) / m), "se_dyadic": row["se_link"],
                 "p_course": 2 * stats.t.sf(abs(est / se_t), m - 1), "p_dyadic": 2 * stats.norm.sf(abs(est / row["se_link"])),
                 "est_current": est, "se_current": se_cur, "p_current": row["p"], "p_holm_current": row["p_holm"]})

res = pl.DataFrame(rows, infer_schema_length=None).with_columns(
    ratio_course=pl.col("se_course") / pl.col("se_dyadic"),
    ratio_cluster=pl.col("se_cluster") / pl.col("se_dyadic"),
    ratio_current=pl.col("se_current") / pl.col("se_dyadic"),
)
# Holm 的族沿用現行做法的族；彙總做法依加權各自校正
out = []
for (an, wt), g in res.group_by("analysis", "weighting", maintain_order=True):
    ph = np.full(g.height, np.nan)
    f = g["family"].to_numpy()
    if f.any():
        ph[f] = holm(g["p_course"].to_numpy()[f])
    out.append(g.with_columns(p_holm_course=pl.Series(ph)))
res = pl.concat(out).with_columns(
    reject_course=pl.col("p_holm_course") < 0.05,
    reject_current=pl.col("p_holm_current") < 0.05,
).with_columns(conclusion_changed=pl.col("reject_course") != pl.col("reject_current"))
with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250, float_precision=4):
    print(res.select("analysis", "cell", "weighting", "players", "n_eff", "est", "est_current", "ratio_course", "ratio_cluster", "ratio_current", "p_holm_course", "p_holm_current", "conclusion_changed"))
res.write_csv("outputs/tables/player_aggregation_compare.csv")

# crossover 中殘差對 h 的玩家斜率（總效果、上一局屬於兩局以上的批）：局數門檻改變時結論是否穩定
grp = ["player", "tournament_id"]
x = (
    player_sample_rd()
    .sort("player", "tournament_id", "start", "game_id")
    .with_columns(h_lag1=pl.col("h").shift(1).over(grp), layer_lag1=pl.col("layer").shift(1).over(grp))
    .filter(pl.col("h_lag1").is_not_null(), pl.col("layer_lag1") == "multi")
)
ref = pl.read_csv("outputs/tables/arena_crossover_player_slope.csv").filter(pl.col("lag") == 1, pl.col("sample") == "multi", pl.col("effect") == "total")
slope_rows = []
for s in SPEEDS:
    xs = x.filter(pl.col("speed") == s)
    for k in [5, 10, 20, 40, 80]:
        slope_rows.append({"speed": s, "min_rows": k, **player_slopes(xs["resid"].to_numpy(), xs["h_lag1"].to_numpy(), xs["player"].to_numpy(), k)})
slope = pl.DataFrame(slope_rows)
mm = slope.filter(pl.col("min_rows") == 10).join(ref, on="speed")
assert (mm["players"] == mm["players_right"]).all() and np.allclose(mm["mean_slope"], mm["mean_slope_right"])
print(slope)
slope.write_csv("outputs/tables/player_aggregation_slope.csv")
