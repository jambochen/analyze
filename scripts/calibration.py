# 期望得分的校準：Elo 形式的 E 是否校準，和局機率與白方優勢是否隨等級變化。
# 樣本為 2014 年 Bullet、Blitz、Classical 的對局；2013 年起的對局只用來算累計局數。

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import optimize, stats

jax.config.update("jax_enable_x64", True)

UTC = "UTC"
CLASSES = ["Bullet", "Blitz", "Classical"]
# 累計局數的門檻是暫定的新手期界線；calibration_experience.png 用來檢查這條界線是否落在校準斜率趨穩之後
N_EST = 30
ELO = np.log(10) / 400
CHUNK = 250_000
# 有序模型與 proportional odds 檢驗在每個速度類別抽 N_FIT 局的簡單隨機樣本上估計
N_FIT = 1_000_000

# 只讀需要的欄位。2014 年 12 月以前的無鐘對局計入
# Classical 的等級分，但沒有時限，與有鐘的對局不是同一種對局，與通信棋一同排除
g = (
    pl.scan_parquet("data/processed/lichess_games.parquet")
    .filter(pl.col("speed") != "Correspondence", pl.col("base_seconds").is_not_null(), pl.col("start") < pl.datetime(2015, 1, 1, time_zone=UTC))
    .select(
        "start", "speed", "white_elo", "black_elo",
        arena=pl.col("tournament_id").is_not_null(),
        y=pl.col("result").replace_strict({"0-1": 0, "1/2-1/2": 1, "1-0": 2}, default=None, return_dtype=pl.Int8),
        w=pl.col("white_id"),
        b=pl.col("black_id"),
    )
    .with_row_index("i")
    .collect(engine="streaming")
)
# 帳號代碼在子樣本中不連續，壓成連續代碼，後面 bincount 以代碼為群
code = pl.concat([g["w"], g["b"]]).rank("dense").cast(pl.UInt32) - 1
g = g.with_columns(w=code[: g.height], b=code[g.height :])
P = int(code.max()) + 1
del code
# 資料從 2013-01 開始，此月就出現的帳號此前可能已有對局，累計局數低估
censored = (
    pl.concat([g.select("start", p="w"), g.select("start", p="b")])
    .group_by("p")
    .agg(pl.col("start").min())
    .filter(pl.col("start") < pl.datetime(2013, 2, 1, time_zone=UTC))
    .select("p", c=pl.lit(True))
)
# 累計局數含 "*" 與缺等級分的對局，因為它們同樣是玩家的經驗；依速度類別分開算以壓低峰值記憶體
nprev = pl.concat([
    pl.concat([g.filter(pl.col("speed") == tc).select("i", "start", p="w"), g.filter(pl.col("speed") == tc).select("i", "start", p="b")])
    .sort("i")
    .with_columns(n_prev=pl.int_range(pl.len(), dtype=pl.UInt32).over("p"))
    .filter(pl.col("start") >= pl.datetime(2014, 1, 1, time_zone=UTC))
    .select("i", "p", "n_prev")
    for tc in CLASSES
])
d = (
    g.lazy()
    .filter(
        pl.col("start") >= pl.datetime(2014, 1, 1, time_zone=UTC),
        pl.col("y").is_not_null(),
        pl.col("white_elo").is_not_null(),
        pl.col("black_elo").is_not_null(),
    )
    .join(nprev.lazy().rename({"p": "w", "n_prev": "nw"}), on=["i", "w"])
    .join(nprev.lazy().rename({"p": "b", "n_prev": "nb"}), on=["i", "b"])
    .join(censored.lazy().rename({"p": "w", "c": "cw"}), on="w", how="left")
    .join(censored.lazy().rename({"p": "b", "c": "cb"}), on="b", how="left")
    .sort("i")
    .with_columns(
        diff=(pl.col("white_elo") - pl.col("black_elo")).cast(pl.Float64),
        rbar=(pl.col("white_elo") + pl.col("black_elo")) / 2,
        n_min=pl.min_horizontal("nw", "nb"),
        source=pl.when(pl.col("arena")).then(pl.lit("arena")).otherwise(pl.lit("lobby")),
        period=pl.when(pl.col("start") < pl.datetime(2014, 8, 1, time_zone=UTC)).then(pl.lit("Jan-Jul")).otherwise(pl.lit("Aug-Dec")),
        train=pl.col("start") < pl.datetime(2014, 7, 1, time_zone=UTC),
        censored=pl.col("cw").fill_null(False) | pl.col("cb").fill_null(False),
    )
    .with_columns(
        score=pl.col("y") / 2,
        E=1 / (1 + 10 ** (-pl.col("diff") / 400)),
        xr=(pl.col("rbar") - 1500) / 100,
        xd=pl.col("diff") / 100,
        est=pl.col("n_min") >= N_EST,
        band=pl.col("rbar").cut([1200, 1400, 1600, 1800, 2000], labels=["<1200", "1200-1400", "1400-1600", "1600-1800", "1800-2000", ">=2000"]).cast(pl.String),
        exp_bin=pl.col("n_min").cut([0, 2, 5, 10, 20, 50, 100, 200], labels=["0", "1-2", "3-5", "6-10", "11-20", "21-50", "51-100", "101-200", ">200"], left_closed=False).cast(pl.String),
        exp_group=pl.when(pl.col("n_min") < 10).then(pl.lit("<10")).when(pl.col("n_min") < N_EST).then(pl.lit(f"10-{N_EST - 1}")).otherwise(pl.lit(f">={N_EST}")),
    )
    # 字串欄在千萬列上很佔記憶體，分組標籤一律存成 Categorical
    .with_columns(pl.col("speed", "source", "period", "band", "exp_bin", "exp_group").cast(pl.Categorical))
    .select("speed", "w", "b", "y", "score", "diff", "E", "xr", "xd", "rbar", "n_min", "est", "censored", "source", "period", "train", "band", "exp_bin", "exp_group")
    .collect(engine="streaming")
)
del g, nprev, censored

sample = d.group_by("speed", "source", "est", "censored").agg(n=pl.len(), draw_rate=(pl.col("y") == 1).mean()).sort(pl.all())
sample.write_csv("outputs/tables/calibration_sample.csv")
print(sample)
print(d.group_by("speed").agg(n=pl.len(), est=pl.col("est").mean(), censored=pl.col("censored").mean(), censored_novice=(pl.col("censored") & ~pl.col("est")).mean()))

# 校準曲線：依 E 分組的平均得分減平均 E，分層看
strata = {"all": pl.lit("all"), "source": pl.col("source"), "band": pl.col("band"), "experience": pl.col("exp_group"), "period": pl.col("period")}
curve = pl.concat([
    d.filter(pl.col("est") | pl.lit(k == "experience"))
    .with_columns(stratum=v.cast(pl.String), e_bin=(pl.col("E") * 20).floor().clip(0, 19) / 20 + 0.025)
    .group_by("speed", "stratum", "e_bin")
    .agg(
        n=pl.len(),
        e_mean=pl.col("E").mean(),
        score_mean=pl.col("score").mean(),
        resid_mean=(pl.col("score") - pl.col("E")).mean(),
        resid_se=(pl.col("score") - pl.col("E")).std() / pl.len().sqrt(),
        p_win=(pl.col("y") == 2).mean(),
        p_draw=(pl.col("y") == 1).mean(),
        p_loss=(pl.col("y") == 0).mean(),
    )
    .with_columns(stratum_type=pl.lit(k))
    for k, v in strata.items()
]).sort("stratum_type", "speed", "stratum", "e_bin")
curve.write_csv("outputs/tables/calibration_curve.csv")

# 勢均力敵的對局（|d| <= 25）看和局率與白方得分隨平均等級分的變化，不經模型
level = (
    d.filter(pl.col("est"), pl.col("diff").abs() <= 25)
    .with_columns(r_bin=(pl.col("rbar") / 100).floor() * 100 + 50)
    .group_by("speed", "r_bin")
    .agg(n=pl.len(), draw_rate=(pl.col("y") == 1).mean(), white_score=pl.col("score").mean(), e_mean=pl.col("E").mean())
    .filter(pl.col("n") >= 500)
    .sort("speed", "r_bin")
)
level.write_csv("outputs/tables/calibration_level.csv")


# 有序模型：白方的潛在優勢 eta 隨分差與平均等級分變化，和局帶寬 w 隨平均等級分變化。
# P(勝) = F(eta - w)、P(不輸) = F(eta + w)。w 的等級分項讓兩個門檻各自移動，
# 所以「和局率是否隨等級變化」與「白方優勢是否隨等級變化」可分開檢驗
def ll_ord(t, xr, xd, y):
    eta = t[0] + t[1] * xr + t[2] * xr**2 + (t[3] + t[4] * xr) * xd
    w = jnp.exp(t[5] + t[6] * xr + t[7] * xr**2)
    u, v = eta + w, eta - w
    # F(u) - F(v) = F(u) F(-v) (1 - e^{-(u-v)})，避免和局率很低時相減失去精度
    ldraw = jax.nn.log_sigmoid(u) + jax.nn.log_sigmoid(-v) + jnp.log(-jnp.expm1(-2 * w))
    return jnp.where(y == 2, jax.nn.log_sigmoid(v), jnp.where(y == 0, jax.nn.log_sigmoid(-u), ldraw))


# 平均得分的 quasi-likelihood（fractional logit）：只對得分的平均建模，和局的 0.5 照實際值代入，
# 不把結果當二元。logit(E) = d ln10/400，故截距 0、斜率 1 即 E 校準
def ll_frac(t, xr, L, s):
    z = t[0] + t[1] * xr + (t[2] + t[3] * xr) * L
    return s * jax.nn.log_sigmoid(z) + (1 - s) * jax.nn.log_sigmoid(-z)


def ll_bin(t, X, z):
    eta = X @ t
    return z * jax.nn.log_sigmoid(eta) + (1 - z) * jax.nn.log_sigmoid(-eta)


# 概似、梯度與 Hessian 逐塊加總，避免 jax 在數百萬列上一次展開中間陣列
def fit(ll, t0, free, args, w, b):
    free = np.asarray(free)
    t0 = jnp.asarray(t0, dtype=jnp.float64)
    full = lambda x: t0.at[free].set(x)
    nll = lambda x, *a: -jnp.sum(ll(full(x), *a))
    cs = [tuple(a[k : k + CHUNK] for a in args) for k in range(0, len(w), CHUNK)]
    f_, g_, h_ = jax.jit(nll), jax.jit(jax.grad(nll)), jax.jit(jax.hessian(nll))
    sc = jax.jit(jax.jacfwd(lambda x, *a: ll(full(x), *a)))
    r = optimize.minimize(
        lambda x: float(sum(f_(x, *c) for c in cs)), np.asarray(t0[free]),
        jac=lambda x: np.asarray(sum(g_(x, *c) for c in cs)),
        hess=lambda x: np.asarray(sum(h_(x, *c) for c in cs)),
        method="trust-exact",
    )
    H = np.asarray(sum(h_(r.x, *c) for c in cs))
    S = np.concatenate([np.asarray(sc(r.x, *c)) for c in cs])
    Hi = np.linalg.inv(H)
    return np.asarray(full(r.x)), -r.fun, Hi @ dyad_meat(S, w, b) @ Hi, Hi, S, H


# 同一玩家的多局不獨立，且玩家同時以白方與黑方出現：任兩局只要有一位共同玩家，
# 分數的乘積就計入 meat（dyadic cluster）。sum_p S_p S_p' 對同一對手組合的對局重複計了兩次，
# 減去以無序對手組合為群的一次
def dyad_meat(S, w, b):
    S = S.reshape(len(w), -1)
    Sp = np.stack([np.bincount(w, S[:, j], P) + np.bincount(b, S[:, j], P) for j in range(S.shape[1])], 1)
    pair = np.unique(np.minimum(w, b).astype(np.int64) * P + np.maximum(w, b), return_inverse=True)[1]
    Sq = np.stack([np.bincount(pair, S[:, j]) for j in range(S.shape[1])], 1)
    return Sp.T @ Sp - Sq.T @ Sq


def wald(theta, V, R, r):
    R = np.atleast_2d(R)
    diff = R @ theta - r
    stat = float(diff @ np.linalg.solve(R @ V @ R.T, diff))
    return stat, len(r), stats.chi2.sf(stat, len(r))


ORD = ["white_adv", "white_adv_x_rating", "white_adv_x_rating2", "slope_diff", "slope_diff_x_rating", "log_draw_width", "log_draw_width_x_rating", "log_draw_width_x_rating2"]
T0 = [0.1, 0, 0, 0.6, 0, -2.5, 0, 0]
IDX = np.arange(8)

coef_rows, test_rows, implied_rows, cox_rows, pred_rows = [], [], [], [], []
fits = {}
for tc in CLASSES:
    m = d.filter(pl.col("speed") == tc, pl.col("est")).sample(N_FIT, seed=1)
    w, b = m["w"].to_numpy(), m["b"].to_numpy()
    xr, xd, y = (m[c].to_numpy().astype(np.float64) for c in ["xr", "xd", "y"])
    th, ll1, V, Hi, S, H = fit(ll_ord, T0, IDX, (xr, xd, y), w, b)
    fits[tc] = th
    for k in range(8):
        coef_rows.append({"speed": tc, "model": "ordinal", "param": ORD[k], "est": th[k], "se_dyadic": np.sqrt(V[k, k]), "se_iid": np.sqrt(Hi[k, k])})

    # 各受限模型：(名稱, 固定的參數, 固定值)。naive_elo_slope 把累積 logit 的斜率固定為 ln10/400，
    # 但有和局時，與 Elo 期望得分斜率相符的累積 logit 斜率大於 ln10/400，此檢定只作對照
    for name, idx, val in [
        ("white_adv_constant_in_rating", [1, 2], [0, 0]),
        ("draw_width_constant_in_rating", [6, 7], [0, 0]),
        ("slope_constant_in_rating", [4], [0]),
        ("all_rating_terms_zero", [1, 2, 4, 6, 7], [0, 0, 0, 0, 0]),
        ("naive_elo_slope", [3], [100 * ELO]),
    ]:
        t0 = np.array(T0, dtype=float)
        t0[idx] = val
        _, ll0, *_ = fit(ll_ord, t0, np.setdiff1d(IDX, idx), (xr, xd, y), w, b)
        R = np.eye(8)[idx]
        ws, df, wp = wald(th, V, R, np.array(val, dtype=float))
        lr = 2 * (ll1 - ll0)
        test_rows.append({"speed": tc, "model": "ordinal", "test": name, "df": df, "wald_dyadic": ws, "p_dyadic": wp, "lr_iid": lr, "p_lr_iid": stats.chi2.sf(lr, df)})

    # proportional odds：兩個累積事件各自做 logit，以同一組共變數，比較分差項是否相同。
    # 兩組分數一起進 dyadic meat，以取得兩組係數的共變異
    X = np.stack([np.ones_like(xr), xr, xr**2, xd, xd * xr], 1)
    tw, _, _, Hw, Sw, _ = fit(ll_bin, np.zeros(5), np.arange(5), (X, (y == 2).astype(np.float64)), w, b)
    tn, _, _, Hn, Sn, _ = fit(ll_bin, np.zeros(5), np.arange(5), (X, (y >= 1).astype(np.float64)), w, b)
    Hi2 = np.zeros((10, 10))
    Hi2[:5, :5], Hi2[5:, 5:] = Hw, Hn
    V2 = Hi2 @ dyad_meat(np.hstack([Sw, Sn]), w, b) @ Hi2
    R = np.zeros((2, 10))
    R[0, 3], R[0, 8], R[1, 4], R[1, 9] = 1, -1, 1, -1
    ws, df, wp = wald(np.concatenate([tw, tn]), V2, R, np.zeros(2))
    wi, _, pi = wald(np.concatenate([tw, tn]), Hi2, R, np.zeros(2))
    test_rows.append({"speed": tc, "model": "binary_pair", "test": "proportional_odds_in_diff", "df": df, "wald_dyadic": ws, "p_dyadic": wp, "wald_iid": wi, "p_wald_iid": pi})
    for k, nm in enumerate(["intercept", "rating", "rating2", "diff", "diff_x_rating"]):
        coef_rows.append({"speed": tc, "model": "logit_win", "param": nm, "est": tw[k], "se_dyadic": np.sqrt(V2[k, k]), "se_iid": np.sqrt(Hw[k, k])})
        coef_rows.append({"speed": tc, "model": "logit_not_loss", "param": nm, "est": tn[k], "se_dyadic": np.sqrt(V2[5 + k, 5 + k]), "se_iid": np.sqrt(Hn[k, k])})

    # 得分平均的校準（Cox 校準截距與斜率），E 是否在平均上校準；參數少，用該類別全部的老手對局
    mf = d.filter(pl.col("speed") == tc, pl.col("est"))
    L = np.log(mf["E"].to_numpy() / (1 - mf["E"].to_numpy()))
    tf, llf, Vf, Hf, *_ = fit(ll_frac, [0, 0, 1, 0], np.arange(4), (mf["xr"].to_numpy(), L, mf["score"].to_numpy()), mf["w"].to_numpy(), mf["b"].to_numpy())
    del mf
    for k, nm in enumerate(["intercept", "intercept_x_rating", "slope_logitE", "slope_logitE_x_rating"]):
        coef_rows.append({"speed": tc, "model": "frac_logit", "param": nm, "est": tf[k], "se_dyadic": np.sqrt(Vf[k, k]), "se_iid": np.sqrt(Hf[k, k])})
    for name, idx, val in [
        ("elo_mean_calibrated", [0, 1, 2, 3], [0, 0, 1, 0]),
        ("no_white_adv_in_mean", [0, 1], [0, 0]),
        ("elo_slope_in_mean", [2, 3], [1, 0]),
    ]:
        ws, df, wp = wald(tf, Vf, np.eye(4)[idx], np.array(val, dtype=float))
        wi, _, pi = wald(tf, Hf, np.eye(4)[idx], np.array(val, dtype=float))
        test_rows.append({"speed": tc, "model": "frac_logit", "test": name, "df": df, "wald_dyadic": ws, "p_dyadic": wp, "wald_iid": wi, "p_wald_iid": pi})

    # 模型隱含的量，在分差 0 處隨平均等級分的變化
    for r in np.arange(1000, 2301, 50):
        z = (r - 1500) / 100
        eta = th[0] + th[1] * z + th[2] * z**2
        slope = th[3] + th[4] * z
        wd = np.exp(th[5] + th[6] * z + th[7] * z**2)
        F = lambda u: 1 / (1 + np.exp(-u))
        dS = 0.5 * slope / 100 * (F(eta + wd) * (1 - F(eta + wd)) + F(eta - wd) * (1 - F(eta - wd)))
        implied_rows.append({
            "speed": tc, "rbar": r,
            "p_draw_d0": F(eta + wd) - F(eta - wd),
            "white_score_d0": 0.5 * (F(eta + wd) + F(eta - wd)),
            "white_adv_rating_points": 100 * eta / slope,
            "score_slope_ratio_to_elo": dS / (ELO / 4),
        })

    # 分層的 Cox 校準：只估截距與斜率，看哪些子樣本偏離 E
    mt = d.filter(pl.col("speed") == tc)
    for k, col in [("experience", "exp_bin"), ("source", "source"), ("band", "band"), ("period", "period"), ("censored", "censored")]:
        for v in mt[col].unique().sort().to_list():
            ms = mt.filter(pl.col(col) == v, pl.col("est") | pl.lit(k == "experience"))
            if ms.height < 2000:
                continue
            Ls = np.log(ms["E"].to_numpy() / (1 - ms["E"].to_numpy()))
            tc_, _, Vc, *_ = fit(ll_frac, [0, 0, 1, 0], np.array([0, 2]), (np.zeros(ms.height), Ls, ms["score"].to_numpy()), ms["w"].to_numpy(), ms["b"].to_numpy())
            cox_rows.append({"speed": tc, "stratum_type": k, "stratum": str(v), "n": ms.height, "intercept": tc_[0], "intercept_se": np.sqrt(Vc[0, 0]), "slope": tc_[2], "slope_se": np.sqrt(Vc[1, 1])})

    # 時間切分的預測評估：1-6 月估計，7-12 月評估
    tr = d.filter(pl.col("speed") == tc, pl.col("est"), pl.col("train")).sample(N_FIT, seed=2)
    te = d.filter(pl.col("speed") == tc, ~pl.col("train"))
    a = [tr[c].to_numpy().astype(np.float64) for c in ["xr", "xd", "y"]]
    th_tr, *_ = fit(ll_ord, T0, IDX, a, tr["w"].to_numpy(), tr["b"].to_numpy())
    t0 = np.array(T0, dtype=float)
    th_base, *_ = fit(ll_ord, t0, np.array([0, 3, 5]), a, tr["w"].to_numpy(), tr["b"].to_numpy())
    # Elo 形式的三類機率：P(和) = nu * 4E(1-E)，勝負各減去一半，得分的期望恰為 E；nu 由訓練期估計
    Etr, ytr = tr["E"].to_numpy(), tr["y"].to_numpy()
    q = 4 * Etr * (1 - Etr)
    nu = optimize.minimize_scalar(
        lambda v: -np.sum(np.where(ytr == 1, np.log(v * q), np.where(ytr == 2, np.log(Etr - v * q / 2), np.log(1 - Etr - v * q / 2)))),
        bounds=(1e-6, 0.45), method="bounded",
    ).x
    Ete, yte = te["E"].to_numpy(), te["y"].to_numpy()
    xr_te, xd_te = te["xr"].to_numpy(), te["xd"].to_numpy()
    probs = {}
    pd_ = nu * 4 * Ete * (1 - Ete)
    probs["elo_draw_split"] = np.stack([1 - Ete - pd_ / 2, pd_, Ete - pd_ / 2], 1)
    for nm, t in [("ordinal_rating", th_tr), ("ordinal_constant", th_base)]:
        eta = t[0] + t[1] * xr_te + t[2] * xr_te**2 + (t[3] + t[4] * xr_te) * xd_te
        wd = np.exp(t[5] + t[6] * xr_te + t[7] * xr_te**2)
        pw, pn = 1 / (1 + np.exp(-(eta - wd))), 1 / (1 + np.exp(-(eta + wd)))
        probs[nm] = np.stack([1 - pn, pn - pw, pw], 1)
    O = np.eye(3)[yte]
    loss = {}
    for nm, p in probs.items():
        C, Co = np.cumsum(p, 1)[:, :2], np.cumsum(O, 1)[:, :2]
        loss[nm] = {
            "log_loss": -np.log(p[np.arange(len(yte)), yte]),
            "rps": ((C - Co) ** 2).sum(1) / 2,
            "score_sq_err": (p[:, 2] + p[:, 1] / 2 - yte / 2) ** 2,
        }
    for grp in ["all", "<10", f"10-{N_EST - 1}", f">={N_EST}"]:
        msk = np.ones(len(yte), bool) if grp == "all" else (te["exp_group"] == grp).to_numpy()
        wg, bg = te["w"].to_numpy()[msk], te["b"].to_numpy()[msk]
        for nm in probs:
            for metric in ["log_loss", "rps", "score_sq_err"]:
                x = loss[nm][metric][msk]
                dlt = x - loss["elo_draw_split"][metric][msk]
                se = np.sqrt(dyad_meat((dlt - dlt.mean())[:, None], wg, bg)[0, 0]) / len(dlt)
                pred_rows.append({"speed": tc, "exp_group": grp, "model": nm, "metric": metric, "n": int(msk.sum()), "mean": x.mean(), "diff_vs_elo": dlt.mean(), "diff_se_dyadic": se, "nu_train": nu})
    print(tc, "done")

coef = pl.DataFrame(coef_rows)
coef.write_csv("outputs/tables/calibration_coef.csv")
tests = pl.DataFrame(test_rows, infer_schema_length=None)
# Holm 法對全部 dyadic Wald 檢定控制族錯誤率
order = np.argsort(tests["p_dyadic"].to_numpy())
pm = tests["p_dyadic"].to_numpy()[order] * (len(order) - np.arange(len(order)))
adj = np.empty(len(order))
adj[order] = np.minimum(np.maximum.accumulate(pm), 1)
tests = tests.with_columns(p_dyadic_holm=adj)
tests.write_csv("outputs/tables/calibration_tests.csv")
implied = pl.DataFrame(implied_rows)
implied.write_csv("outputs/tables/calibration_implied.csv")
cox = pl.DataFrame(cox_rows)
cox.write_csv("outputs/tables/calibration_cox.csv")
pred = pl.DataFrame(pred_rows)
pred.write_csv("outputs/tables/calibration_prediction.csv")
print(coef)
print(tests)
print(implied.filter(pl.col("rbar") % 200 == 0))
print(cox)
print(pred.filter(pl.col("metric") != "score_sq_err"))

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
for k, fname, title in [("source", "calibration_curve_source.png", "source"), ("band", "calibration_curve_rating.png", "mean rating"), ("experience", "calibration_curve_experience.png", "min prior games of the two players")]:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, tc in zip(axes, CLASSES):
        c = curve.filter(pl.col("stratum_type") == k, pl.col("speed") == tc, pl.col("n") >= 200)
        for j, v in enumerate(c["stratum"].unique().sort().to_list()):
            cc = c.filter(pl.col("stratum") == v)
            ax.errorbar(cc["e_mean"], cc["resid_mean"], yerr=1.96 * cc["resid_se"], label=v, color=colors[j], marker="o", ms=3, lw=1, capsize=2)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_title(tc)
        ax.set_xlabel("Elo expected score E (white)")
    axes[0].set_ylabel("mean score - mean E")
    axes[-1].legend(title=title, fontsize=8)
    fig.suptitle(f"Calibration residual of Elo expected score by {title} (bins of E, width 0.05; iid 95% CI)")
    fig.tight_layout()
    fig.savefig(f"outputs/figures/{fname}", dpi=150)

fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
for j, tc in enumerate(CLASSES):
    lv = level.filter(pl.col("speed") == tc)
    im = implied.filter(pl.col("speed") == tc)
    axes[0, j].plot(lv["r_bin"], lv["draw_rate"], "o", ms=4, label="observed, |d| <= 25")
    axes[0, j].plot(im["rbar"], im["p_draw_d0"], "-", label="ordinal model, d = 0")
    axes[1, j].plot(lv["r_bin"], lv["white_score"], "o", ms=4, label="observed, |d| <= 25")
    axes[1, j].plot(im["rbar"], im["white_score_d0"], "-", label="ordinal model, d = 0")
    axes[1, j].axhline(0.5, color="gray", lw=0.8)
    axes[0, j].set_title(tc)
    axes[1, j].set_xlabel("mean rating of the two players")
axes[0, 0].set_ylabel("draw probability")
axes[1, 0].set_ylabel("white expected score")
axes[0, 0].legend(fontsize=8)
fig.suptitle(f"Draw rate and white advantage by rating level (both players >= {N_EST} prior games)")
fig.tight_layout()
fig.savefig("outputs/figures/calibration_level.png", dpi=150)

order_exp = ["0", "1-2", "3-5", "6-10", "11-20", "21-50", "51-100", "101-200", ">200"]
fig, ax = plt.subplots(figsize=(7, 4))
for j, tc in enumerate(CLASSES):
    c = cox.filter(pl.col("speed") == tc, pl.col("stratum_type") == "experience").with_columns(o=pl.col("stratum").replace_strict({v: i for i, v in enumerate(order_exp)}, return_dtype=pl.Int32)).sort("o")
    ax.errorbar(c["o"] + (j - 1) * 0.1, c["slope"], yerr=1.96 * c["slope_se"], marker="o", ms=4, capsize=2, label=tc, color=colors[j])
ax.axhline(1, color="gray", lw=0.8)
ax.set_xticks(range(len(order_exp)), order_exp)
ax.set_xlabel("min prior games in the speed class (since 2013-01)")
ax.set_ylabel("calibration slope on logit(E)")
ax.set_title("Calibration slope by experience (fractional logit, dyadic 95% CI)")
ax.legend()
fig.tight_layout()
fig.savefig("outputs/figures/calibration_experience.png", dpi=150)
