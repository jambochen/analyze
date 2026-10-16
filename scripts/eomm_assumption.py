# EOMM（arXiv 1702.06820）的關鍵假設：給定本局結果，對手（分差）不再影響自己是否離開。
# 在 2014 年 arena 的隨機配對下，於 EOMM 的流失模型（最近 10 局的勝和負、與前一局的間隔、近期局數、本局結果）
# 加入 q = |e - 0.5| 與「結果 × q」，以 Wald 檢定 H0：q 與交互作用的係數皆為 0；並估 EOMM 的理論判準
# Δ = c(勝) + c(負) - 2c(和)。主分析限兩局以上的配對批次（設計上的隨機），一局一批作補充。

import resource
import sys

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pyfixest as pf
from scipy.stats import chi2

sys.path.insert(0, "scripts")
from arena_batches import player_sample
from calibrated import probs
from clogit import fit

NB = 16
RESP = ["leave_8h", "tour_left", "leave_session"]
WINDOWS = {"n_8h": 8 * 3600, "n_1d": 86400, "n_7d": 7 * 86400, "n_30d": 30 * 86400}

d = player_sample().with_columns(
    elo_c2=pl.col("elo_c") ** 2,
    leave_session=~pl.col("continue_session"),
    bullet=(pl.col("speed") == "Bullet").cast(pl.Float64),
    classical=(pl.col("speed") == "Classical").cast(pl.Float64),
).with_columns(pl.col(RESP).cast(pl.Float64)).filter(pl.col("layer") != "first_single", pl.col("q_bar").is_not_null())

# 歷史特徵須含一般對局與 2013 年的對局（player_games 從 2014 年起），故由對局檔重建每位玩家的完整序列。
# 序列的定義同 sessions.py：不含無鐘對局（含通信棋）與自己對自己的局，三個速度類別合併。2013 年 1 月以前的對局史不在資料中：
# 2014 年的對局，近 30 天的局數都完整可見；最近 10 局則只在 2013 年 1 月起累計不到 10 局時以「無」補足，
# 這時「無」混合了真正的新帳號與左截斷，與 player_game_no 的口徑相同
raw = pl.scan_parquet("data/processed/lichess_games.parquet").filter(
    pl.col("base_seconds").is_not_null(), pl.col("white_id") != pl.col("black_id"), pl.col("start") < pl.datetime(2015, 1, 1, time_zone="UTC")
)
score_white = pl.col("result").replace_strict({"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}, default=None)
cols = ["game_id", "start", "speed"]
long = pl.concat([
    raw.select(*cols, player=pl.col("white_id"), color=pl.lit("white"), elo=pl.col("white_elo"), opp_elo=pl.col("black_elo"), score=score_white),
    raw.select(*cols, player=pl.col("black_id"), color=pl.lit("black"), elo=pl.col("black_elo"), opp_elo=pl.col("white_elo"), score=1 - score_white),
])
players = d["player"].unique().implode()
keys = d.select("game_id", "player")
hist = []
for b in range(NB):
    h = (
        long.filter(pl.col("player") % NB == b, pl.col("player").is_in(players))
        .collect()
        .sort("player", "start", "game_id")
    )
    h = probs(h, color=True).with_columns(resid=(pl.col("score") - pl.col("e_cal")).fill_null(0.0), pid=pl.col("player").rle_id())
    key = (h["pid"].to_numpy().astype(np.int64) << 34) + h["start"].dt.epoch("s").to_numpy()
    cs = np.r_[0, np.cumsum(h["resid"].to_numpy())]
    idx = np.arange(h.height)
    extra = {}
    for nm, w in WINDOWS.items():
        lo = np.searchsorted(key, key - w, side="left")
        extra[nm] = idx - lo
        if nm in ["n_8h", "n_7d"]:
            extra["resid_" + nm[2:]] = cs[idx] - cs[lo]
    h = h.with_columns(
        **{nm: pl.Series(v) for nm, v in extra.items()},
        **{f"lag{k}": pl.col("score").shift(k).over("pid") for k in range(1, 11)},
    )
    hist.append(h.join(keys, on=["game_id", "player"], how="semi").drop("pid", "e_cal", "p_win", "p_draw", "p_loss", "elo", "opp_elo", "score", "color", "speed", "start", "resid"))
hist = pl.concat(hist)
d = d.join(hist, on=["game_id", "player"])
print(d.height, hist.height, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1000, "MB", flush=True)

LAGS = []
for k in range(1, 11):
    v = pl.col(f"lag{k}")
    d = d.with_columns(
        **{f"lag{k}_win": (v == 1).fill_null(False).cast(pl.Float64), f"lag{k}_draw": (v == 0.5).fill_null(False).cast(pl.Float64), f"lag{k}_none": v.is_null().cast(pl.Float64)}
    )
    LAGS += [f"lag{k}_win", f"lag{k}_draw", f"lag{k}_none"]
d = d.with_columns(
    log_prev_gap=pl.col("prev_gap").fill_null(0).log1p(),
    **{f"log_{nm}": pl.col(nm).log1p() for nm in WINDOWS},
    # 近期表現殘差（實際得分減校準期望得分）的平均：當日狀態的可觀測代理，用於中介變數的敏感度分析
    form_8h=pl.when(pl.col("n_8h") > 0).then(pl.col("resid_8h") / pl.col("n_8h")).otherwise(0.0),
    form_7d=pl.when(pl.col("n_7d") > 0).then(pl.col("resid_7d") / pl.col("n_7d")).otherwise(0.0),
    q_win=pl.col("q") * pl.col("win"),
    q_draw=pl.col("q") * pl.col("draw"),
    e_win=pl.col("e") * pl.col("win"),
    e_draw=pl.col("e") * pl.col("draw"),
)
EOMM = LAGS + ["log_prev_gap"] + [f"log_{nm}" for nm in WINDOWS]
# EOMM 原模型沒有的控制：q 的分布依自身等級分與候選對手的組成而異。隨機配對保證的是給定期望匹配品質 q_bar（或 e_bar）後
# q 與玩家無關，故 q 的模型一律控制 q_bar，並控制自身等級分與錦標賽的時程
CTRL = ["elo_c", "elo_c2", "white", "log_games_in_tour", "log_remaining", "novice", "bullet", "classical"]
RES = ["win", "draw"]
QT = ["q", "q_win", "q_draw"]
ET = ["e", "e_win", "e_draw"]
FORM = ["form_8h", "form_7d"]
models = {
    "eomm": EOMM + RES + CTRL,
    "eomm_q": EOMM + RES + QT + ["q_bar"] + CTRL,
    "eomm_e": EOMM + RES + ET + ["e_bar"] + CTRL,
    "eomm_q_form": EOMM + RES + QT + ["q_bar"] + CTRL + FORM,
}
d = d.select(*dict.fromkeys([*RESP, "layer", "player", "bid", "tournament_id", "tour_win5", *[c for cols in models.values() for c in cols], *ET]))


def design(df, cols, result=None, q=None):
    """結果或 q 設定為指定值時的設計矩陣；交互作用項隨之重算。"""
    over = {}
    if result is not None:
        over |= {"win": pl.lit(float(result == "win")), "draw": pl.lit(float(result == "draw"))}
    if q is not None:
        over |= {"q": pl.lit(q)}
    x = df.select("q", "e", "win", "draw", *[c for c in cols if c not in QT + ET + RES])
    x = x.with_columns(**over) if over else x
    x = x.with_columns(q_win=pl.col("q") * pl.col("win"), q_draw=pl.col("q") * pl.col("draw"), e_win=pl.col("e") * pl.col("win"), e_draw=pl.col("e") * pl.col("draw"))
    return np.column_stack([np.ones(df.height), x.select(cols).to_numpy()])



def cluster_cov(X, u, bread, g):
    """玩家聚類的 sandwich 共變數；u 為每列的分數權重（logit 為 y - p，OLS 為殘差）。"""
    o = np.argsort(g, kind="stable")
    st = np.r_[0, np.flatnonzero(np.diff(g[o])) + 1]
    S = np.add.reduceat(X[o] * u[o, None], st, axis=0)
    G = len(st)
    return bread @ (S.T @ S) @ bread * G / (G - 1)


# statsmodels 的 Logit 在一局一批（百萬列以上）時記憶體超過上限，故以 Newton 法自行擬合
def logit_fit(y, X, w):
    b = np.zeros(X.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-X @ b))
        H = (X * (w * p * (1 - p))[:, None]).T @ X
        step = np.linalg.solve(H, X.T @ (w * (y - p)))
        b += step
        if np.abs(step).max() < 1e-9:
            break
    return b


def logit_cluster(y, X, g):
    b = logit_fit(y, X, np.ones(len(y)))
    p = 1 / (1 + np.exp(-X @ b))
    Hinv = np.linalg.inv((X * (p * (1 - p))[:, None]).T @ X)
    return b, cluster_cov(X, y - p, Hinv, g)


def ols_cluster(y, X, g):
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ (X.T @ y)
    return b, cluster_cov(X, y - X @ b, XtXi, g)


coef_rows, wald_rows, c_rows, curve_rows = [], [], [], []
for layer in ["multi", "single"]:
    for r in RESP:
        s = d.filter(pl.col("layer") == layer, pl.col(r).is_not_null())
        groups = s["player"].to_numpy()
        for mname, cols in models.items():
            X = design(s, cols)
            b, V = logit_cluster(s[r].to_numpy(), X, groups)
            del X
            names = ["const"] + cols
            for j, nm in enumerate(names):
                if nm in QT + ET + RES + FORM + ["const"]:
                    coef_rows.append(dict(layer=layer, response=r, model=mname, term=nm, coef=b[j], se_cluster_player=np.sqrt(V[j, j]), n=s.height))
            for test_name, terms in [("q_and_interactions", QT), ("e_and_interactions", ET), ("form", FORM)]:
                if all(t in cols for t in terms):
                    R = np.zeros((len(terms), len(names)))
                    for i, t in enumerate(terms):
                        R[i, names.index(t)] = 1
                    rb = R @ b
                    stat = float(rb @ np.linalg.solve(R @ V @ R.T, rb))
                    wald_rows.append(dict(layer=layer, response=r, model=mname, estimator="logit", test=test_name, df=len(terms), stat=stat, p=float(chi2.sf(stat, len(terms))), n=s.height))
            # c(o)：把本局結果設為 o、其他變數取觀察值的平均預測離開機率；Δ 與各 c 的標準誤以 delta method 由玩家聚類的共變數求得
            cs_, gs_ = {}, {}
            for o in ["win", "draw", "loss"]:
                Xo = design(s, cols, result=o)
                p = 1 / (1 + np.exp(-Xo @ b))
                cs_[o] = p.mean()
                gs_[o] = (p * (1 - p)) @ Xo / len(p)
            for o in ["win", "draw", "loss"]:
                c_rows.append(dict(layer=layer, response=r, model=mname, quantity=f"c_{o}", est=cs_[o], se=np.sqrt(gs_[o] @ V @ gs_[o]), n=s.height))
            gd = gs_["win"] + gs_["loss"] - 2 * gs_["draw"]
            c_rows.append(dict(layer=layer, response=r, model=mname, quantity="delta", est=cs_["win"] + cs_["loss"] - 2 * cs_["draw"], se=np.sqrt(gd @ V @ gd), n=s.height))
            if mname == "eomm_q":
                for qv in np.linspace(0, 0.45, 10):
                    for o in ["win", "draw", "loss"]:
                        Xo = design(s, cols, result=o, q=qv)
                        p = 1 / (1 + np.exp(-Xo @ b))
                        g_ = (p * (1 - p)) @ Xo / len(p)
                        curve_rows.append(dict(layer=layer, response=r, result=o, q=qv, pred=p.mean(), se=np.sqrt(g_ @ V @ g_)))
        print(layer, r, "done", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1000, "MB", flush=True)

        # 穩健性：批次分層的 conditional logit（兩局以上的批）或錦標賽 × 5 分鐘固定效應的 logistic（一局一批），
        # 把 q 的比較限制在同一次配對或同一時段之內，不依賴自身等級分的函數形式
        cols = models["eomm_q"]
        if layer == "multi":
            # 速度類別與錦標賽剩餘時間在同一批內不變，條件概似中無法識別
            cols = [c for c in cols if c not in ["bullet", "classical", "log_remaining"]]
            fr = fit(s[r].to_numpy(), s.select(cols).to_numpy(), s["bid"].to_numpy(), s["tournament_id"].to_numpy(), cols)
            ix = [cols.index(t) for t in QT]
            bq = fr["coef"][ix]
            Vq = fr["cov"][np.ix_(ix, ix)]
            stat = float(bq @ np.linalg.solve(Vq, bq))
            wald_rows.append(dict(layer=layer, response=r, model="eomm_q", estimator="clogit_batch_cluster_tournament", test="q_and_interactions", df=3, stat=stat, p=float(chi2.sf(stat, 3)), n=fr["n_used"]))
            for j, t in zip(ix, QT):
                coef_rows.append(dict(layer=layer, response=r, model="eomm_q_clogit", term=t, coef=fr["coef"][j], se_cluster_player=np.nan, se_cluster_tournament=np.sqrt(fr["cov"][j, j]), n=fr["n_used"]))
        else:
            ctrl = [c for c in cols if c not in ["bullet", "classical"]]
            fe = pf.feglm(f"{r} ~ {' + '.join(ctrl)} | tour_win5", s.select(r, *ctrl, pl.col("tour_win5").cast(pl.Categorical).to_physical(), "player").to_pandas(), family="logit", vcov={"CRV1": "player"}, lean=True, store_data=False, copy_data=False)
            t = fe.tidy()
            bq = t.loc[QT, "Estimate"].to_numpy()
            Vq = fe._vcov[np.ix_([list(t.index).index(x) for x in QT], [list(t.index).index(x) for x in QT])]
            stat = float(bq @ np.linalg.solve(Vq, bq))
            wald_rows.append(dict(layer=layer, response=r, model="eomm_q", estimator="fe_logit_tour5m_cluster_player", test="q_and_interactions", df=3, stat=stat, p=float(chi2.sf(stat, 3)), n=fe._N))
            for x in QT:
                coef_rows.append(dict(layer=layer, response=r, model="eomm_q_fe", term=x, coef=t.loc[x, "Estimate"], se_cluster_player=t.loc[x, "Std. Error"], n=fe._N))

coef = pl.DataFrame(coef_rows)
wald = pl.DataFrame(wald_rows)
ctab = pl.DataFrame(c_rows)
curve = pl.DataFrame(curve_rows)
with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250):
    print(wald)
    print(ctab.filter(pl.col("quantity") == "delta"))
    print(coef.filter(pl.col("term").is_in(QT + ET + FORM)))
coef.write_csv("outputs/tables/eomm_assumption_coef.csv")
wald.write_csv("outputs/tables/eomm_assumption_wald.csv")
ctab.write_csv("outputs/tables/eomm_assumption_c.csv")
curve.write_csv("outputs/tables/eomm_assumption_curve.csv")

# 驗證 delta method：兩局以上的批，以玩家為單位的 block bootstrap 重估 Δ 少量次數，比較標準差與 delta method 的標準誤
rng = np.random.default_rng(0)
boot_rows = []
for r in ["leave_8h", "tour_left"]:
    s = d.filter(pl.col("layer") == "multi", pl.col(r).is_not_null())
    cols = models["eomm_q"]
    X = design(s, cols)
    Xo = {o: design(s, cols, result=o) for o in ["win", "draw", "loss"]}
    y = s[r].to_numpy()
    # bootstrap 以代碼為 bincount 的索引，須為連續代碼
    pc = (s["player"].rank("dense") - 1).to_numpy()
    npl = pc.max() + 1
    for i in range(50):
        w = np.bincount(rng.integers(0, npl, npl), minlength=npl)[pc].astype(float)
        b = logit_fit(y, X, w)
        c = {o: np.average(1 / (1 + np.exp(-Xo[o] @ b)), weights=w) for o in Xo}
        boot_rows.append(dict(response=r, rep=i, delta=c["win"] + c["loss"] - 2 * c["draw"]))
boot = pl.DataFrame(boot_rows).group_by("response").agg(reps=pl.len(), boot_mean=pl.col("delta").mean(), boot_sd=pl.col("delta").std())
boot = boot.join(ctab.filter(pl.col("layer") == "multi", pl.col("model") == "eomm_q", pl.col("quantity") == "delta").select("response", delta=pl.col("est"), delta_method_se=pl.col("se")), on="response")
print(boot)
boot.write_csv("outputs/tables/eomm_assumption_bootstrap.csv")

# 中介變數的偏誤公式（線性機率模型，omitted variable bias 在 OLS 下是恆等式）：控制結果後，未觀測的當日狀態 U
# 若同時影響結果與離開，q 的係數偏誤為 γ_U × δ_U（γ_U：U 對離開的係數；δ_U：U 對 q 等項的係數）。
# 以可觀測的近期表現殘差為基準：假設 U 在兩條路徑上都是它的 k 倍強，求 k 多大時偏誤足以抵銷 q 的平均直接效果，或使其信賴區間碰到 0
ovb_rows = []
for layer in ["multi", "single"]:
    for r in RESP:
        s = d.filter(pl.col("layer") == layer, pl.col(r).is_not_null())
        groups = s["player"].to_numpy()
        cols = models["eomm_q"]
        X = design(s, cols)
        y = s[r].to_numpy()
        names = ["const"] + cols
        share = {"win": s["win"].mean(), "draw": s["draw"].mean()}
        a = np.zeros(len(names))
        a[names.index("q")] = 1
        a[names.index("q_win")] = share["win"]
        a[names.index("q_draw")] = share["draw"]
        bo, Vo = ols_cluster(y, X, groups)
        ame = a @ bo
        se = np.sqrt(a @ Vo @ a)
        for u in FORM:
            uz = s[u].to_numpy()
            uz = (uz - uz.mean()) / uz.std()
            Xu = X.T @ uz
            A = np.block([[X.T @ X, Xu[:, None]], [Xu[None, :], np.array([[uz @ uz]])]])
            gamma_u = np.linalg.solve(A, np.r_[X.T @ y, uz @ y])[-1]
            delta_u = a @ np.linalg.solve(X.T @ X, Xu)
            bias1 = gamma_u * delta_u
            row = dict(layer=layer, response=r, benchmark=u, ame_q=ame, ame_q_se=se, gamma_u=gamma_u, delta_u=delta_u, bias_at_k1=bias1, n=s.height)
            # 偏誤與估計同號時，夠強的 U 才能把估計推向 0；異號時 U 只會使真實效果離 0 更遠
            same = np.sign(bias1) == np.sign(ame)
            row["k_explain_away"] = np.sqrt(ame / bias1) if same and bias1 != 0 else np.inf
            bound = ame - np.sign(ame) * 1.96 * se
            row["k_ci_to_zero"] = np.sqrt(bound / bias1) if same and bias1 != 0 and np.sign(bound) == np.sign(ame) else (0.0 if np.sign(bound) != np.sign(ame) else np.inf)
            ovb_rows.append(row)
ovb = pl.DataFrame(ovb_rows)
with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250):
    print(ovb)
ovb.write_csv("outputs/tables/eomm_assumption_ovb.csv")

fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True)
for i, layer in enumerate(["multi", "single"]):
    for j, r in enumerate(RESP):
        ax = axes[i, j]
        for o, c in [("loss", "C3"), ("draw", "C7"), ("win", "C0")]:
            x = curve.filter(pl.col("layer") == layer, pl.col("response") == r, pl.col("result") == o)
            ax.plot(x["q"], x["pred"], color=c, label=o)
            ax.fill_between(x["q"], x["pred"] - 1.96 * x["se"], x["pred"] + 1.96 * x["se"], color=c, alpha=0.2)
        ax.set_title(f"{r}, {'batches with 2+ games' if layer == 'multi' else 'one-game batches'}", fontsize=9)
        if i == 1:
            ax.set_xlabel("match quality q = |e - 0.5|")
        if j == 0:
            ax.set_ylabel("mean predicted P(leave)")
axes[0, 0].legend(fontsize=8)
fig.suptitle("EOMM churn model with q and result x q: average prediction with result fixed", fontsize=10)
fig.tight_layout()
fig.savefig("outputs/figures/eomm_assumption_curve.png", dpi=150)

fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
dl = ctab.filter(pl.col("quantity") == "delta")
for j, r in enumerate(RESP):
    ax = axes[j]
    x = dl.filter(pl.col("response") == r)
    labels = [f"{a}\n{b}" for a, b in zip(x["layer"], x["model"])]
    ax.errorbar(np.arange(x.height), x["est"], yerr=1.96 * x["se"], fmt="o")
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xticks(np.arange(x.height), labels, fontsize=6, rotation=90)
    ax.set_title(f"{r}: c(win) + c(loss) - 2 c(draw)", fontsize=9)
fig.tight_layout()
fig.savefig("outputs/figures/eomm_assumption_delta.png", dpi=150)
