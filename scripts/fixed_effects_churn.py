# Heliyon 2024 的模型在 Lichess 上的複製：玩家日的 14 天流失對當日的分差、勝率與連勝連敗。
# 同一模型分別估計大廳（玩家自選對手）與 arena（系統隨機配對）的玩家日，比較分差的係數。
# 估計量：Heliyon 的雙向固定效應 logit（主要）、雙向固定效應線性機率模型；以玩家為條件的 conditional logit 在抽樣玩家上對照 incidental parameters 的偏誤。

import gc
import sys

import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pyfixest as pf
from scipy import stats

sys.path.insert(0, "scripts")
import likelihood as lk

B = 100
N_SUB = 20_000
BIG = -1e30
rng = np.random.default_rng(0)

EXP_RAW = ["gap_mean", "gap_sd", "win_rate", "win_streak_rate", "loss_streak_rate"]
EXP_ECAL = ["ecal_mean", "ecal_sd", "win_rate", "win_streak_rate", "loss_streak_rate"]
EXP_SCORE = ["gap_mean", "gap_sd", "score_rate", "win_streak_rate", "loss_streak_rate"]
STAGES = ["s1", "s2", "s3", "s4", "s5"]
SPECS = {
    "raw": EXP_RAW + ["n"],
    "ecal": EXP_ECAL + ["n"],
    "raw_score": EXP_SCORE + ["n"],
    # 不含表現變數：分差經由勝負的部分不被控制，對應隨機配對分析的總效果
    "raw_total": ["gap_mean", "gap_sd", "n"],
    "ecal_total": ["ecal_mean", "ecal_sd", "n"],
    "q_total": ["q_mean", "n"],
    "q": ["q_mean", "win_rate", "win_streak_rate", "loss_streak_rate", "n"],
    # 隨機配對下分差的期望仍依自身等級分在對手群中的位置而定；玩家固定效應吸收不了當日等級分的變動，故另以自身等級分控制
    "raw_elo": EXP_RAW + ["n", "elo_c", "elo_c2"],
    "raw_total_elo": ["gap_mean", "gap_sd", "n", "elo_c", "elo_c2"],
    "q_total_elo": ["q_mean", "n", "elo_c", "elo_c2"],
    # 分差等五個變數在各階段各有一個係數（等價於原文的主效果加交互作用），階段本身以虛擬變數進入
    "raw_stage": [f"{x}_x_{s}" for s in STAGES for x in EXP_RAW] + ["n"] + STAGES[1:],
    "ecal_stage": [f"{x}_x_{s}" for s in STAGES for x in EXP_ECAL] + ["n"] + STAGES[1:],
}

pdays = pl.read_parquet("data/processed/player_days.parquet").with_columns(
    # 分差以 100 等級分為單位；期望得分以百分點為單位，與勝率同尺度。對手較強時 gap 為正、ecal 較低
    gap_mean=pl.col("gap_mean") / 100,
    gap_sd=pl.col("gap_sd") / 100,
    ecal_mean=pl.col("ecal_mean") * 100,
    ecal_sd=pl.col("ecal_sd") * 100,
    q_mean=pl.col("q_mean") * 100,
    elo_c=(pl.col("elo_mean") - 1500) / 100,
    elo_c2=((pl.col("elo_mean") - 1500) / 100) ** 2,
    churn=pl.col("churn").cast(pl.Float64),
    n=pl.col("n").cast(pl.Float64),
    # 遊戲階段以當日最後一局的累計局數分段；第一段的上限 30 局即校準所用的新手期門檻
    stage=pl.col("game_no").cut([30, 100, 300, 1000], labels=["s1", "s2", "s3", "s4", "s5"], left_closed=False),
)
new_players = pdays.filter(pl.col("first_game_no") == 1)["player"].unique().implode()
pdays = pdays.with_columns(new_player=pl.col("player").is_in(new_players))


def heliyon_sample(d):
    # 原文只留至少兩個活躍日、且期間內流失狀態有變動的玩家；conditional logit 本來就不使用其他玩家。
    # 某日全無流失（或全流失）時日期的係數發散，該日的觀察對其他係數沒有資訊，與 fixest 的做法相同一併排除，
    # 排除後玩家的條件可能不再成立，故交替篩到不變為止
    while True:
        h = d.height
        d = (
            d.with_columns(nd=pl.len().over("player"), k=pl.col("churn").sum().over("player"))
            .filter(pl.col("nd") >= 2, pl.col("k") > 0, pl.col("k") < pl.col("nd"))
            .with_columns(nd=pl.len().over("day"), k=pl.col("churn").sum().over("day"))
            .filter(pl.col("k") > 0, pl.col("k") < pl.col("nd"))
            .drop("nd", "k")
        )
        if d.height == h:
            break
    return (
        d.sort("player", "day")
        .with_columns([(pl.col("stage") == s).cast(pl.Float64).alias(s) for s in STAGES])
        .with_columns([(pl.col(x) * pl.col(s)).alias(f"{x}_x_{s}") for s in STAGES for x in set(EXP_RAW + EXP_ECAL)])
    )


def cl_data(d, xcols):
    # d 已依玩家排序，dense rank 與排序一致，同一玩家的列連續且編號遞增
    seg = (d["player"].rank("dense").cast(pl.Int64) - 1).to_numpy()
    G = int(seg.max()) + 1
    y = d["churn"].to_numpy()
    nn = np.bincount(seg, minlength=G)
    kk = np.bincount(seg, weights=y, minlength=G).astype(np.int64)
    # 流失日多於一半的玩家改以「未流失」為事件、分數取負，條件概似不變，動態規劃的階數取 min(k, n-k)
    flip = kk > nn - kk
    m = np.where(flip, nn - kk, kk)
    sgn = np.where(flip[seg], -1.0, 1.0)
    ystar = np.where(flip[seg], 1 - y, y)
    starts = np.concatenate([[0], np.cumsum(nn)[:-1]])
    classes = []
    # 依 (m, n 的 2 的冪次區間) 分組補齊長度，補齊造成的浪費不超過一倍；動態規劃的反向傳遞要存每一步的狀態，記憶體與補齊後的格數成正比
    nbin = np.ceil(np.log2(np.maximum(nn, 2))).astype(np.int64)
    for mm, nb in sorted(set(zip(m[m >= 2].tolist(), nbin[m >= 2].tolist()))):
        gid = np.flatnonzero((m == mm) & (nbin == nb))
        nmax = nn[gid].max()
        pos = np.arange(nmax)
        idx = starts[gid][:, None] + pos[None, :]
        mask = pos[None, :] < nn[gid][:, None]
        idx = np.where(mask, idx, 0)
        e0 = np.full((len(gid), mm + 1), BIG)
        e0[:, 0] = 0.0
        classes.append({"gid": jnp.asarray(gid), "idx": jnp.asarray(idx.T), "mask": jnp.asarray(mask.T), "e0": jnp.asarray(e0)})
    days = d["day"].rank("dense").cast(pl.Int64).to_numpy() - 1
    return {
        "X": jnp.asarray(d.select(xcols).to_numpy()),
        "day": jnp.asarray(days),
        "seg": jnp.asarray(seg),
        "sgn": jnp.asarray(sgn),
        "ystar": jnp.asarray(ystar),
        "m1": jnp.asarray(m == 1),
        # 只用其長度：segment_sum 的群數須為編譯期常數
        "G": jnp.zeros(G),
        "classes": classes,
    }, G, int(days.max()) + 1, seg, days


def row_scores(theta, data):
    p = data["X"].shape[1]
    gam = jnp.concatenate([jnp.zeros(1), theta[p:]])
    return data["X"] @ theta[:p] + gam[data["day"]]


def group_ll(s, data, G):
    s = data["sgn"] * s
    num = jax.ops.segment_sum(data["ystar"] * s, data["seg"], G)
    mx = jax.lax.stop_gradient(jax.ops.segment_max(s, data["seg"], G))
    lse = mx + jnp.log(jax.ops.segment_sum(jnp.exp(s - mx[data["seg"]]), data["seg"], G))
    den = jnp.where(data["m1"], lse, 0.0)
    for c in data["classes"]:
        a = s[c["idx"]]

        def step(e, inp):
            aj, mj = inp
            new = jnp.concatenate([e[:, :1], jnp.logaddexp(e[:, 1:], aj[:, None] + e[:, :-1])], axis=1)
            return jnp.where(mj[:, None], new, e), None

        e, _ = jax.lax.scan(step, c["e0"], (a, c["mask"]))
        den = den.at[c["gid"]].set(e[:, -1])
    return num - den


@jax.jit
def cl_ll(th, w, data):
    return jnp.sum(w * group_ll(row_scores(th, data), data, data["G"].shape[0]))


cl_grad = jax.jit(jax.grad(cl_ll))
cl_hvp = jax.jit(jax.vmap(lambda th, w, v, data: jax.jvp(lambda t: jax.grad(cl_ll)(t, w, data), (th,), (v,))[1], in_axes=(None, None, 0, None)))


def cl_fit(d, xcols):
    data, G, D, seg, days = cl_data(d, xcols)
    p = len(xcols)
    k = p + D - 1
    ll = lambda th: cl_ll(th, w1, data)
    w1 = jnp.ones(G)

    def hess(th):
        eye = np.eye(k)
        return np.vstack([np.asarray(cl_hvp(th, w1, jnp.asarray(eye[i : i + 8]), data)) for i in range(0, k, 8)])

    th = jnp.zeros(k)
    cur = float(ll(th))
    for it in range(30):
        g = np.asarray(cl_grad(th, w1, data))
        step = np.linalg.solve(-hess(th), g)
        t = 1.0
        while float(ll(th + t * step)) < cur - 1e-9 and t > 1e-4:
            t /= 2
        th = th + t * step
        new = float(ll(th))
        done = abs(new - cur) < 1e-8 * max(1, abs(cur)) and np.max(np.abs(t * step)) < 1e-7
        cur = new
        if done:
            break
    Hinv = np.linalg.inv(-hess(th))
    srow = np.asarray(jax.grad(lambda s: jnp.sum(group_ll(s, data, G)))(row_scores(th, data)))
    X = np.asarray(data["X"])
    S = np.zeros((G, k))
    np.add.at(S[:, :p], seg, srow[:, None] * X)
    dmask = days > 0
    np.add.at(S, (seg[dmask], p + days[dmask] - 1), srow[dmask])
    V = Hinv @ (S.T @ S) @ Hinv
    print("cl iter", it, flush=True)
    return {"theta": np.asarray(th), "V": V, "V_model": Hinv, "iter": it}


def ybar_var(d):
    # 流失率的玩家 cluster 變異數，供平均半彈性的 delta method
    g = d.group_by("player").agg(y=pl.col("churn").sum(), n=pl.len())
    yb = g["y"].sum() / g["n"].sum()
    G = g.height
    return yb, float(((g["y"] - yb * g["n"]) ** 2).sum()) / g["n"].sum() ** 2 * G / (G - 1)


def tidy(est, se, terms, d, estimator, info):
    est, se = np.asarray(est), np.asarray(se)
    yb, vy = ybar_var(d)
    # 平均半彈性：logit 為 β(1 - ȳ)（原文的 ASE），線性機率模型為 β / ȳ。delta method 含 ȳ 的抽樣變異，β 與 ȳ 的共變視為 0，由 bootstrap 驗證
    if estimator == "lpm_fe":
        ase, ase_se = est / yb, np.sqrt(se**2 / yb**2 + est**2 * vy / yb**4)
    else:
        ase, ase_se = est * (1 - yb), np.sqrt(se**2 * (1 - yb) ** 2 + est**2 * vy)
    return pl.DataFrame({
        "term": terms, "est": est, "se": se, "p": 2 * stats.norm.sf(np.abs(est / se)), "ase": ase, "ase_se": ase_se,
    }).with_columns(estimator=pl.lit(estimator), ybar=pl.lit(yb), **{k: pl.lit(v) for k, v in info.items()})


def fe_fit(d, xcols, family):
    pdf = d.select("player", "day", "churn", *xcols).with_columns(pl.col("day").cast(pl.String)).to_pandas()
    fml = "churn ~ " + " + ".join(xcols) + " | player + day"
    if family == "logit":
        return pf.feglm(fml, data=pdf, family="logit", vcov={"CRV1": "player"}, demeaner_backend="rust")
    return pf.feols(fml, data=pdf, vcov={"CRV1": "player"}, demeaner_backend="rust")


def fit_all(d, spec, info, estimators=("logit_fe", "lpm_fe")):
    xcols = SPECS[spec]
    info = {**info, "spec": spec, "player_days": d.height, "players": d["player"].n_unique()}
    rows, models = [], {}
    for est in estimators:
        if est == "cl":
            r = cl_fit(d, xcols)
            p = len(xcols)
            rows.append(tidy(r["theta"][:p], np.sqrt(np.diag(r["V"])[:p]), xcols, d, "cl", info).with_columns(se_model=np.sqrt(np.diag(r["V_model"])[:p])))
            models[est] = r
        else:
            m = fe_fit(d, xcols, "logit" if est == "logit_fe" else "ols")
            rows.append(tidy(m.coef().to_numpy(), m.se().to_numpy(), list(m.coef().index), d, est, info))
            models[est] = m
    print(info, flush=True)
    return rows, models


def stacked(dl, da, xcols, family):
    # 兩樣本疊起來的完全交互作用模型：玩家與日期的固定效應依樣本分開，係數為「大廳」與「arena 減大廳」。
    # 概似可分離，點估計與分開估計相同；同一玩家在兩樣本的 score 進同一個 cluster，差距的標準誤因此含兩估計值的共變
    st = pl.concat([dl.with_columns(arena=pl.lit(0.0)), da.with_columns(arena=pl.lit(1.0))], how="diagonal").with_columns(
        ps=pl.col("player").cast(pl.String) + "_" + pl.col("arena").cast(pl.String), ds=pl.col("day").cast(pl.String) + "_" + pl.col("arena").cast(pl.String)
    )
    st = st.with_columns([(pl.col(x) * pl.col("arena")).alias(f"{x}__arena") for x in xcols])
    terms = xcols + [f"{x}__arena" for x in xcols]
    pdf = st.select("player", "ps", "ds", "churn", *terms).to_pandas()
    del st
    fml = "churn ~ " + " + ".join(terms) + " | ps + ds"
    # 只用係數與 CRV1 共變異，不保留資料與設計矩陣，否則兩樣本疊起來的模型超過記憶體上限
    lean = dict(lean=True, store_data=False, copy_data=False)
    if family == "logit":
        m = pf.feglm(fml, data=pdf, family="logit", vcov={"CRV1": "player"}, demeaner_backend="rust", **lean)
    else:
        m = pf.feols(fml, data=pdf, vcov={"CRV1": "player"}, demeaner_backend="rust", **lean)
    names = list(m.coef().index)
    return m.coef().to_numpy(), m._vcov, names


def diff_rows(dl, da, spec, family, label):
    xcols = SPECS[spec]
    b, V, names = stacked(dl, da, xcols, family)
    ybl, vyl = ybar_var(dl)
    yba, vya = ybar_var(da)
    out = []
    for x in xcols[:2]:
        i, j = names.index(x), names.index(f"{x}__arena")
        bl, dlt = b[i], b[j]
        ba = bl + dlt
        se_b = np.sqrt(V[j, j])
        if family == "logit":
            cl_, ca_ = 1 - ybl, 1 - yba
            dvy = bl**2 * vyl + ba**2 * vya
        else:
            cl_, ca_ = 1 / ybl, 1 / yba
            dvy = bl**2 * vyl / ybl**4 + ba**2 * vya / yba**4
        # ASE 差 = β_L c_L - (β_L + δ) c_A，對 (β_L, δ) 的梯度
        gr = np.array([cl_ - ca_, -ca_])
        Vs = V[np.ix_([i, j], [i, j])]
        out.append({
            "label": label, "spec": spec, "estimator": "logit_fe" if family == "logit" else "lpm_fe", "term": x,
            "beta_lobby": bl, "beta_arena": ba, "se_lobby": np.sqrt(V[i, i]), "se_arena": np.sqrt(V[i, i] + V[j, j] + 2 * V[i, j]),
            "diff_lobby_minus_arena": -dlt, "diff_se": se_b,
            "ase_lobby": bl * cl_, "ase_arena": ba * ca_, "ase_diff": bl * cl_ - ba * ca_, "ase_diff_se": np.sqrt(gr @ Vs @ gr + dvy),
        })
    return out


base = pdays.filter(pl.col("cut") == "utc", pl.col("day_pure"))
samples = {a: heliyon_sample(base.filter(pl.col("arena") == a)) for a in [False, True]}
SAMPLE_NAME = {False: "lobby", True: "arena"}

desc = []
for cut in ["utc", "shift"]:
    for pure in [True, False]:
        for a in [False, True]:
            d = pdays.filter(pl.col("cut") == cut, pl.col("day_pure") | (not pure), pl.col("arena") == a)
            h = heliyon_sample(d)
            desc.append({
                "cut": cut, "day_pure": pure, "sample": SAMPLE_NAME[a],
                "player_days": d.height, "players": d["player"].n_unique(), "churn_rate": d["churn"].mean(),
                "player_days_kept": h.height, "players_kept": h["player"].n_unique(), "days_kept": h["day"].n_unique(),
                "days_per_player_kept_mean": h.height / h["player"].n_unique(),
                "days_per_player_kept_median": h.group_by("player").len()["len"].median(),
                "players_single_day": d.group_by("player").len().filter(pl.col("len") == 1).height,
                "players_never_churn": d.group_by("player").agg(pl.col("churn").max()).filter(pl.col("churn") == 0).height,
                "churn_rate_kept": h["churn"].mean(),
            })
desc = pl.DataFrame(desc)
print(desc)
desc.write_csv("outputs/tables/fixed_effects_churn_sample.csv")

vars_ = ["churn", "gap_mean", "gap_sd", "ecal_mean", "ecal_sd", "q_mean", "elo_mean", "win_rate", "score_rate", "draw_rate", "win_streak_rate", "loss_streak_rate", "n", "game_no", "bullet", "classical"]
describe = pl.concat([
    d.select(vars_).unpivot().group_by("variable").agg(
        mean=pl.col("value").mean(), sd=pl.col("value").std(), min=pl.col("value").min(), max=pl.col("value").max()
    ).with_columns(sample=pl.lit(SAMPLE_NAME[a]))
    for a, d in samples.items()
])
# 固定效應只用玩家內的變動；分差的玩家內標準差決定了兩個樣本的估計精確度
within = pl.concat([
    d.select("player", *vars_).with_columns([(pl.col(v) - pl.col(v).mean().over("player")) for v in vars_])
    .select(vars_).unpivot().group_by("variable").agg(within_sd=pl.col("value").std()).with_columns(sample=pl.lit(SAMPLE_NAME[a]))
    for a, d in samples.items()
])
describe = describe.join(within, on=["sample", "variable"]).sort("sample", "variable")
print(describe)
describe.write_csv("outputs/tables/fixed_effects_churn_describe.csv")

coefs = []
for spec in ["raw", "ecal", "raw_score", "raw_total", "ecal_total", "q_total", "q", "raw_elo", "raw_total_elo", "q_total_elo"]:
    for a, d in samples.items():
        coefs += fit_all(d, spec, {"sample": SAMPLE_NAME[a], "cut": "utc", "day_pure": True, "subset": "all"})[0]

# 各變數的係數是否隨遊戲階段而異：五個階段係數相等（4 個限制）的概似比檢定，受限模型以共同係數取代五個交互作用項，
# 虛無分布以玩家 cluster 的 sandwich 修正；Wald 統計量並列作對照
stage_tests = []
for spec in ["raw_stage", "ecal_stage"]:
    for a, d in samples.items():
        rows, models = fit_all(d, spec, {"sample": SAMPLE_NAME[a], "cut": "utc", "day_pure": True, "subset": "all"}, estimators=("logit_fe",))
        coefs += rows
        m = models["logit_fe"]
        names = list(m.coef().index)
        Vc = m._vcov.copy()
        Hi = m.vcov("iid")._vcov
        for x in (EXP_RAW if spec == "raw_stage" else EXP_ECAL):
            ii = [names.index(f"{x}_x_{g}") for g in STAGES]
            R = np.diff(np.eye(len(names))[ii], axis=0)
            w, _, pw = lk.wald(m.coef().to_numpy(), Vc, R, 0)
            m0 = fe_fit(d, [c for c in SPECS[spec] if c not in [f"{x}_x_{g}" for g in STAGES]] + [x], "logit")
            # binary logit 的 deviance 即 -2 倍對數概似；兩模型的固定效應相同
            r = lk.adjusted_lrt(m0.deviance - m.deviance, lk.lr_weights_linear(Hi, Vc, R))
            stage_tests.append({"spec": spec, "sample": SAMPLE_NAME[a], "variable": x, **r, "wald": w, "p_wald": pw, "n_full": m._N, "n_restricted": m0._N})
stage_tests = pl.DataFrame(stage_tests)
print(stage_tests)
stage_tests.write_csv("outputs/tables/fixed_effects_churn_stage_tests.csv")

for cut, pure in [("shift", True), ("utc", False)]:
    for a in [False, True]:
        d = heliyon_sample(pdays.filter(pl.col("cut") == cut, pl.col("day_pure") | (not pure), pl.col("arena") == a))
        for spec in ["raw", "ecal"]:
            coefs += fit_all(d, spec, {"sample": SAMPLE_NAME[a], "cut": cut, "day_pure": pure, "subset": "all"})[0]

for subset, cond in [
    ("new_player", pl.col("new_player")),
    ("existing_player", ~pl.col("new_player")),
    ("jan_jul", pl.col("day") < pl.date(2014, 8, 1)),
    ("aug_dec", pl.col("day") >= pl.date(2014, 8, 1)),
]:
    for a in [False, True]:
        d = heliyon_sample(base.filter(pl.col("arena") == a, cond))
        coefs += fit_all(d, "raw", {"sample": SAMPLE_NAME[a], "cut": "utc", "day_pure": True, "subset": subset})[0]

# 大廳與 arena 的差距：堆疊模型。另以分開估計的點估計核對堆疊模型
diff = []
for spec in ["raw", "ecal", "raw_total", "ecal_total", "q_total", "q", "raw_elo", "raw_total_elo", "q_total_elo"]:
    for family in ["logit", "ols"]:
        diff += diff_rows(samples[False], samples[True], spec, family, "full")
for cut, pure in [("shift", True), ("utc", False)]:
    dl = heliyon_sample(pdays.filter(pl.col("cut") == cut, pl.col("day_pure") | (not pure), ~pl.col("arena")))
    da = heliyon_sample(pdays.filter(pl.col("cut") == cut, pl.col("day_pure") | (not pure), pl.col("arena")))
    for spec in ["raw", "ecal"]:
        diff += diff_rows(dl, da, spec, "logit", f"{cut}_{'pure' if pure else 'any'}")
diff = pl.DataFrame(diff)
sep = pl.concat(coefs, how="diagonal").filter(pl.col("cut") == "utc", pl.col("day_pure"), pl.col("subset") == "all").select("sample", "spec", "estimator", "term", "est")
sep = sep.pivot(on="sample", index=["spec", "estimator", "term"], values="est").rename({"lobby": "sep_lobby", "arena": "sep_arena"})
diff = diff.join(sep, on=["spec", "estimator", "term"], how="left").with_columns(
    max_abs_dev_from_separate=pl.when(pl.col("label") == "full").then(
        pl.max_horizontal((pl.col("beta_lobby") - pl.col("sep_lobby")).abs(), (pl.col("beta_arena") - pl.col("sep_arena")).abs())
    )
).drop("sep_lobby", "sep_arena")
print(diff)
diff.write_csv("outputs/tables/fixed_effects_churn_diff.csv")

# Hausman 檢定以 Mundlak 形式做：加入解釋變數的玩家平均，檢驗其係數全為 0。
# 在玩家 cluster 的穩健共變異下仍有效，傳統的 Hausman 統計量須假設隨機效應估計有效率，且 logit 沒有現成的隨機效應對照
haus = []
for spec in ["raw", "ecal"]:
    xs = SPECS[spec]
    for a, d in samples.items():
        dm = d.with_columns([pl.col(x).mean().over("player").alias(f"{x}_bar") for x in xs])
        pdf = dm.select("player", "day", "churn", *xs, *[f"{x}_bar" for x in xs]).with_columns(pl.col("day").cast(pl.String)).to_pandas()
        fml = "churn ~ " + " + ".join(xs + [f"{x}_bar" for x in xs]) + " | day"
        # 線性機率模型只是原文設定的對照，維持 Wald；logit 以修正的概似比檢定作決策，受限模型去掉玩家平均項。
        # 模型依序建立、用完即釋放：兩個模型同時保留資料會超過記憶體上限
        for est in ["lpm_mundlak", "logit_mundlak"]:
            if est == "lpm_mundlak":
                m = pf.feols(fml, data=pdf, vcov={"CRV1": "player"})
            else:
                m = pf.feglm(fml, data=pdf, family="logit", vcov={"CRV1": "player"})
            names = list(m.coef().index)
            ii = [names.index(f"{x}_bar") for x in xs]
            Vc = m._vcov.copy()
            w, _, pw = lk.wald(m.coef().to_numpy(), Vc, np.eye(len(names))[ii], 0)
            row = {"spec": spec, "sample": SAMPLE_NAME[a], "test": est, "df": len(xs), "wald": w, "p_wald": pw}
            if est == "logit_mundlak":
                Hi = m.vcov("iid")._vcov
                dev1, n1 = m.deviance, m._N
                del m
                m0 = pf.feglm("churn ~ " + " + ".join(xs) + " | day", data=pdf, family="logit", vcov="iid", lean=True, store_data=False, copy_data=False)
                row |= lk.adjusted_lrt(m0.deviance - dev1, lk.lr_weights(Hi, Vc, ii))
                row |= {"n_full": n1, "n_restricted": m0._N}
                del m0
            else:
                del m
            haus.append(row)
            gc.collect()
        del pdf, dm
haus = pl.DataFrame(haus, infer_schema_length=None)
print(haus)
haus.write_csv("outputs/tables/fixed_effects_churn_hausman.csv")

# 抽樣玩家上的兩項檢查：conditional logit 對固定效應 logit（incidental parameters），與 bootstrap 對 cluster sandwich。
# 大廳隨機抽 N_SUB 位玩家，arena 玩家數本來就少，全數保留
lob_players = samples[False]["player"].unique().sort().sample(N_SUB, seed=1)
sub = {False: heliyon_sample(samples[False].filter(pl.col("player").is_in(lob_players.implode()))), True: samples[True]}
for spec in ["raw", "ecal"]:
    for a, d in sub.items():
        coefs += fit_all(d, spec, {"sample": SAMPLE_NAME[a], "cut": "utc", "day_pure": True, "subset": "player_subsample"}, estimators=("cl", "logit_fe"))[0]

coefs = pl.concat(coefs, how="diagonal")
print(coefs.filter(pl.col("term").is_in(["gap_mean", "ecal_mean"])).select("sample", "spec", "cut", "day_pure", "subset", "estimator", "term", "est", "se", "ase", "ase_se", "p"))
coefs.write_csv("outputs/tables/fixed_effects_churn_coef.csv")

union = pl.concat([sub[False]["player"], sub[True]["player"]]).unique().sort()
boot = []
for rep in range(B):
    cnt = pl.DataFrame({"player": union, "copies": rng.multinomial(len(union), np.full(len(union), 1 / len(union)))}).filter(pl.col("copies") > 0)
    # 重複抽到的玩家各為一個新玩家（各自的固定效應與 cluster），以複本編號區分
    cnt = cnt.with_columns(copy=pl.int_ranges(pl.col("copies"))).explode("copy")
    db = {a: d.join(cnt.select("player", "copy"), on="player").with_columns(player=pl.col("player").cast(pl.String) + "#" + pl.col("copy").cast(pl.String)).drop("copy") for a, d in sub.items()}
    for r in diff_rows(db[False], db[True], "raw", "logit", "boot"):
        boot.append({**r, "rep": rep})
    print("boot", rep, flush=True)
boot = pl.DataFrame(boot)
boot.write_csv("outputs/tables/fixed_effects_churn_bootstrap_draws.csv")
ref = pl.DataFrame(diff_rows(sub[False], sub[True], "raw", "logit", "subsample"))
qs = ["beta_lobby", "beta_arena", "diff_lobby_minus_arena", "ase_diff"]
ses = {"beta_lobby": "se_lobby", "beta_arena": "se_arena", "diff_lobby_minus_arena": "diff_se", "ase_diff": "ase_diff_se"}
check = pl.concat([
    pl.DataFrame({"term": [t] * len(qs), "quantity": qs,
                  "estimate": [ref.filter(pl.col("term") == t)[q][0] for q in qs],
                  "sandwich_se": [ref.filter(pl.col("term") == t)[ses[q]][0] for q in qs],
                  "bootstrap_se": [boot.filter(pl.col("term") == t)[q].std() for q in qs],
                  "B": [B] * len(qs)})
    for t in ["gap_mean", "gap_sd"]
]).with_columns(ratio=pl.col("bootstrap_se") / pl.col("sandwich_se"))
print(check)
check.write_csv("outputs/tables/fixed_effects_churn_bootstrap_check.csv")

main = coefs.filter(pl.col("cut") == "utc", pl.col("day_pure"), pl.col("spec").is_in(["raw", "ecal"]), pl.col("term").is_in(["gap_mean", "ecal_mean"]))
rowsdef = [("all", "logit_fe", "Two-way FE logit"), ("all", "lpm_fe", "Two-way FE LPM"), ("player_subsample", "logit_fe", "FE logit (player subsample)"), ("player_subsample", "cl", "Conditional logit (player subsample)")]
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, (spec, term, lab) in zip(axes, [("raw", "gap_mean", "Mean rating gap, opponent - self\n(per 100 points)"), ("ecal", "ecal_mean", "Mean calibrated expected score\n(per percentage point)")]):
    for k, (subset, est, _) in enumerate(rowsdef):
        for sgi, (s, col) in enumerate([("lobby", "C0"), ("arena", "C1")]):
            r = main.filter(pl.col("spec") == spec, pl.col("term") == term, pl.col("estimator") == est, pl.col("sample") == s, pl.col("subset") == subset)
            ax.errorbar(r["ase"][0], k + (sgi - 0.5) * 0.3, xerr=1.96 * r["ase_se"][0], fmt="o", color=col, label=s if k == 0 else None)
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_yticks(range(len(rowsdef)), [x[2] for x in rowsdef], fontsize=8)
    ax.invert_yaxis()
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.set_xlabel("Average semi-elasticity of 14-day churn (95% CI)", fontsize=9)
    ax.set_title(lab, fontsize=9)
axes[0].legend(frameon=False)
fig.tight_layout()
fig.savefig("outputs/figures/fixed_effects_churn_gap.png", dpi=150)

st = coefs.filter(pl.col("spec") == "raw_stage", pl.col("estimator") == "logit_fe")
fig, axes = plt.subplots(1, 5, figsize=(14, 3.2))
for ax, x in zip(axes, EXP_RAW):
    for s, col, off in [("lobby", "C0", -0.1), ("arena", "C1", 0.1)]:
        r = st.filter(pl.col("sample") == s)
        est = [r.filter(pl.col("term") == f"{x}_x_{g}")["est"][0] for g in STAGES]
        se = [r.filter(pl.col("term") == f"{x}_x_{g}")["se"][0] for g in STAGES]
        ax.errorbar(np.arange(5) + off, est, yerr=1.96 * np.array(se), fmt="o-", color=col, label=s, ms=3)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xticks(range(5), ["1-30", "31-100", "101-300", "301-1000", ">1000"], rotation=45, fontsize=7)
    ax.set_title(x, fontsize=9)
axes[0].set_ylabel("FE logit coefficient (95% CI)")
axes[0].legend(frameon=False, fontsize=7)
fig.supxlabel("Game stage (cumulative games since 2013-01)", fontsize=9)
fig.tight_layout()
fig.savefig("outputs/figures/fixed_effects_churn_stage.png", dpi=150)
