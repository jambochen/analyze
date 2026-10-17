# 隨機配對下，匹配品質 q = |e - 0.5| 對離開的效果。2014 年 arena 的玩家局，識別分兩層：
# 兩局以上的配對批次是設計上的隨機，以批次分層的 conditional logit 估計，並在批內依 lila 規則重新配對作 randomization inference；
# 一局一批的配對只是近似隨機（假設下完的時間與等級分無關），以「錦標賽 × 5 分鐘」固定效應的 logistic 迴歸作模型化分析。
# 兩層都控制期望匹配品質 q_bar（定義見 arena_batches.player_sample）：隨機配對保證的是 q 在給定 q_bar 下與玩家無關，
# 不是 q 本身與玩家無關。

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pyfixest as pf

sys.path.insert(0, "scripts")
import pairing_null
from arena_batches import player_sample
from clogit import fit

B = 2000
RESP = ["tour_left", "leave_session", "leave_8h", "leave_session_half", "leave_session_double"]
MAIN = ["tour_left", "leave_session", "leave_8h"]
PRE = ["elo_c", "elo_c2", "log_games_in_tour", "novice"]

d = (
    player_sample()
    .with_columns(
        elo_c2=pl.col("elo_c") ** 2,
        leave_session=~pl.col("continue_session"),
        leave_session_half=~pl.col("continue_half"),
        leave_session_double=~pl.col("continue_double"),
        tour_win1=pl.concat_str("tournament_id", pl.col("start").dt.truncate("1m").cast(pl.String)),
        tour_win15=pl.concat_str("tournament_id", pl.col("start").dt.truncate("15m").cast(pl.String)),
    )
    .with_columns(pl.col(RESP).cast(pl.Float64))
)

sample = (
    d.group_by("layer", "rule")
    .agg(
        player_games=pl.len(),
        batches=pl.col("bid").n_unique(),
        tournaments=pl.col("tournament_id").n_unique(),
        players=pl.col("player").n_unique(),
        q_bar_missing=pl.col("q_bar").is_null().sum(),
        *[pl.col(r).is_not_null().sum().alias(f"{r}_n") for r in RESP],
        *[pl.col(r).mean().alias(f"{r}_rate") for r in RESP],
    )
    .sort("layer", "rule")
)
with pl.Config(tbl_cols=-1, tbl_width_chars=250):
    print(sample)
sample.write_csv("outputs/tables/match_quality_leave_sample.csv")

m, ns = pairing_null.structure(d)
nrow = m.height
bi, nb = ns["bi"], ns["nb"]
e_act = ns["e_act"]
assert np.abs(e_act - m["e"].to_numpy()).max() < 1e-9
rule = m["rule"].to_numpy()
after = m["after_1130"].to_numpy()
q_bar = m["q_bar"].to_numpy()
e_bar = m["e_bar"].to_numpy()


# 檢定統計量：y 先對批次固定效應與配對前就決定的變數（自身等級分、本場已下局數、新手）殘差化，再取 q - q_bar 的批內斜率。
# 本局結果與執子顏色隨配對而變，sharp null 下無法固定，不進統計量；因此 RI 檢驗的是匹配品質的總效果。
# 未減 q_bar 的版本（q_raw）並列：它的虛無分布不以 0 為中心，顯示等級分位置的混入
Xpre = m.select(PRE).to_numpy()
subgroups = {"all": np.zeros(nrow, int), "rule": (rule == "smart").astype(int), "period": after.astype(int)}
sub_names = {"all": ["all"], "rule": ["random", "smart"], "period": ["before_1130", "after_1130"]}
resid = {}
for r in RESP:
    y = m[r].to_numpy()
    ok = ~np.isnan(y)
    w = ok.astype(float)
    cnt = np.bincount(bi, w, nb)
    yc = np.where(ok, y - (np.bincount(bi, np.where(ok, y, 0), nb) / np.maximum(cnt, 1))[bi], 0)
    Xc = np.column_stack([np.where(ok, x - (np.bincount(bi, np.where(ok, x, 0), nb) / np.maximum(cnt, 1))[bi], 0) for x in Xpre.T])
    beta = np.linalg.lstsq(Xc[ok], yc[ok], rcond=None)[0]
    resid[r] = (np.where(ok, yc - Xc @ beta, 0), w, cnt)


def slopes(x):
    out = {}
    for r, (ec, w, cnt) in resid.items():
        xc = (x - (np.bincount(bi, w * x, nb) / np.maximum(cnt, 1))[bi]) * w
        for s, lab in subgroups.items():
            num = np.bincount(lab, xc * ec)
            den = np.bincount(lab, xc * xc)
            for j, nm in enumerate(sub_names[s]):
                out[(r, s, nm)] = num[j] / den[j]
    return out


def stats(e):
    q = np.abs(e - 0.5)
    return {"q": slopes(q - q_bar), "e": slopes(e - e_bar), "q_raw": slopes(q)}


act = stats(e_act)
ri_rows = []
null_keep = {}
draws_by_null = pairing_null.run(ns, stats, B)
for null in pairing_null.NULLS:
    draws = draws_by_null[null]
    for var in act:
        for key, a in act[var].items():
            nd = np.array([dr[var][key] for dr in draws])
            r, s, nm = key
            sel = (subgroups[s] == sub_names[s].index(nm)) & (resid[r][1] > 0)
            ri_rows.append(
                dict(
                    null=null, var=var, response=r, subgroup=s, level=nm, n=int(sel.sum()),
                    batches=len(np.unique(bi[sel])), slope=a, null_mean=nd.mean(), null_sd=nd.std(),
                    null_lo=np.quantile(nd, 0.025), null_hi=np.quantile(nd, 0.975),
                    p=(1 + (np.abs(nd - nd.mean()) >= abs(a - nd.mean())).sum()) / (B + 1),
                )
            )
            if null == "main" and s == "all":
                null_keep[(var, r)] = (a, nd)
ri = pl.DataFrame(ri_rows)
# 主要假設事先指定為三個主反應變數的 q 總效果（主虛無分布、全部批次），以 Holm 法控制族錯誤率
main_sel = (pl.col("null") == "main") & (pl.col("var") == "q") & (pl.col("subgroup") == "all") & pl.col("response").is_in(MAIN)
pm = ri.filter(main_sel).sort("p")
holm = np.minimum(1, np.maximum.accumulate(pm["p"].to_numpy() * (len(pm) - np.arange(len(pm)))))
ri = ri.join(pm.select("response").with_columns(p_holm=holm, null=pl.lit("main"), var=pl.lit("q"), subgroup=pl.lit("all")), on=["null", "var", "response", "subgroup"], how="left")
with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250):
    print(ri.filter(pl.col("subgroup") == "all"))
ri.write_csv("outputs/tables/match_quality_leave_ri.csv")

# XLA 的 CPU 執行緒數不受 OMP 等環境變數控制，以 affinity 限制在 8 核，與其他並行工作共用機器。
# 放在平行池之後：fork 出的工作行程會繼承 affinity
os.sched_setaffinity(0, range(8))

# 批次分層的 conditional logit。total 只放配對前的變數，對應 RI 所檢驗的總效果；with_result 加入本局結果與顏色，是控制中介後的係數，
# 解讀見 eomm_assumption.py。標準誤以錦標賽聚類：批次巢套在錦標賽中，條件概似的分數只能逐批計算，無法再拆到玩家
cl_rows = []
specs = {
    "total": ["q", "q_bar", *PRE],
    "with_result": ["q", "q_bar", "win", "draw", "white", *PRE],
    "total_signed": ["e", "e_bar", *PRE],
    "with_result_signed": ["e", "e_bar", "win", "draw", "white", *PRE],
    "total_no_qbar": ["q", *PRE],
}
for r in RESP:
    s = m.filter(pl.col(r).is_not_null())
    for spec, cols in specs.items():
        res = fit(s[r].to_numpy(), s.select(cols).to_numpy(), s["bid"].to_numpy(), s["tournament_id"].to_numpy(), cols)
        se = np.sqrt(np.diag(res["cov"]))
        se_n = np.sqrt(np.diag(res["cov_naive"]))
        for j, nm in enumerate(cols):
            cl_rows.append(dict(response=r, spec=spec, term=nm, coef=res["coef"][j], se_cluster_tournament=se[j], se_model=se_n[j], z=res["coef"][j] / se[j], n=res["n_used"], batches=res["groups_used"]))
clog = pl.DataFrame(cl_rows)
with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250):
    print(clog.filter(pl.col("term").is_in(["q", "e", "win", "draw"])))
clog.write_csv("outputs/tables/match_quality_leave_clogit.csv")

# 一局一批：錦標賽 × 5 分鐘的固定效應吸收錦標賽的組成與賽程，q 的變異來自同一時段內誰恰好同時下完。玩家聚類標準誤。
# 只留需要的欄位、字串鍵轉成整數代碼（玩家本來就是代碼）：整份資料框轉 pandas 後在 feglm 內複製，會超過記憶體上限
s1 = (
    d.filter(pl.col("layer") == "single", pl.col("q_bar").is_not_null())
    .select(
        *RESP, "q", "e", "q_bar", "e_bar", "win", "draw", "white", "elo_c", "elo_c2", "log_games_in_tour", "log_remaining", "novice", "early_half", "after_1130",
        *[pl.col(c).cast(pl.Categorical).to_physical() for c in ["tour_win1", "tour_win5", "tour_win15"]], "player",
    )
    .to_pandas()
)
del m
CTRL = "white + elo_c + elo_c2 + log_games_in_tour + log_remaining + novice"
fe_rows = []


def fe_fit(r, x, extra, fe, sub, label):
    df = s1[s1[r].notna()] if sub is None else s1[s1[r].notna() & sub]
    res = pf.feglm(f"{r} ~ {x} {extra} + {CTRL} | {fe}", df, family="logit", vcov={"CRV1": "player"}, lean=True, store_data=False, copy_data=False)
    t = res.tidy()
    for term in t.index:
        if term in ["q", "e", "q_bar", "e_bar", "win", "draw", "white"]:
            fe_rows.append(dict(response=r, spec=label, fe=fe, term=term, coef=t.loc[term, "Estimate"], se_cluster_player=t.loc[term, "Std. Error"], p=t.loc[term, "Pr(>|t|)"], n=res._N))


for r in RESP:
    fe_fit(r, "q + q_bar", "", "tour_win5", None, "total")
    fe_fit(r, "q + q_bar", "+ win + draw", "tour_win5", None, "with_result")
    fe_fit(r, "e + e_bar", "", "tour_win5", None, "total_signed")
    fe_fit(r, "e + e_bar", "+ win + draw", "tour_win5", None, "with_result_signed")
for r in MAIN:
    fe_fit(r, "q", "", "tour_win5", None, "total_no_qbar")
    fe_fit(r, "q + q_bar", "", "tour_win15", None, "total_window_15m")
    fe_fit(r, "q + q_bar", "", "tour_win5", s1["early_half"], "total_early_half")
    fe_fit(r, "q + q_bar", "", "tour_win5", s1["novice"] == 0, "total_veteran")
    fe_fit(r, "q + q_bar", "", "tour_win5", ~s1["after_1130"], "total_before_1130")
    fe_fit(r, "q + q_bar", "", "tour_win5", s1["after_1130"], "total_after_1130")
fe_tab = pl.DataFrame(fe_rows)
with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250):
    print(fe_tab)
fe_tab.write_csv("outputs/tables/match_quality_leave_fe.csv")

# 描述：依有號期望得分分組的離開率，分本局結果。期望得分低於 0.5 表示對手較強
desc = (
    d.filter(pl.col("layer") != "first_single")
    .with_columns(e_bin=(pl.col("e") * 10).floor().clip(0, 9) / 10 + 0.05)
    .group_by("layer", "result", "e_bin")
    .agg(n=pl.len(), *[pl.col(r).mean().alias(r) for r in MAIN], *[pl.col(r).is_not_null().sum().alias(f"{r}_n") for r in MAIN])
    .sort("layer", "result", "e_bin")
)
desc.write_csv("outputs/tables/match_quality_leave_desc.csv")

fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True)
for i, layer in enumerate(["multi", "single"]):
    for j, r in enumerate(MAIN):
        ax = axes[i, j]
        for res_, c in [("loss", "C3"), ("draw", "C7"), ("win", "C0")]:
            x = desc.filter(pl.col("layer") == layer, pl.col("result") == res_, pl.col(f"{r}_n") >= 200)
            ax.plot(x["e_bin"], x[r], "o-", color=c, label=res_, ms=3)
        ax.set_title(f"{r}, {'batches with 2+ games' if layer == 'multi' else 'one-game batches'}", fontsize=9)
        if i == 1:
            ax.set_xlabel("calibrated expected score (no color), binned")
        if j == 0:
            ax.set_ylabel("share leaving")
axes[0, 0].legend(fontsize=8)
fig.tight_layout()
fig.savefig("outputs/figures/match_quality_leave_desc.png", dpi=150)

fig, axes = plt.subplots(3, 3, figsize=(11, 8))
titles = {"q": "q - q_bar", "q_raw": "q, not centered", "e": "e - e_bar"}
for i, var in enumerate(["q", "q_raw", "e"]):
    for j, r in enumerate(MAIN):
        a, nd = null_keep[(var, r)]
        ax = axes[i, j]
        ax.hist(nd, bins=40, color="0.7")
        ax.axvline(a, color="C3")
        ax.axvline(0, color="k", lw=0.6, ls=":")
        ax.set_title(f"{r}, slope on {titles[var]}", fontsize=9)
    axes[i, 0].set_ylabel("re-pairings")
fig.suptitle("Actual (red) vs within-batch re-pairing null, batches with 2+ games, 2014 arena", fontsize=10)
fig.tight_layout()
fig.savefig("outputs/figures/match_quality_leave_ri.png", dpi=150)
