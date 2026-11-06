# 民間停止規則的回測：刪去與保留的局的平均殘差之差、隔天重置，以玩家 block bootstrap 推論。
# folk_stop_rules.py（2014 年下半年）與 prospective_validation.py（2015 年前瞻驗證）共用。

import numpy as np
import polars as pl
from scipy.stats import norm

import likelihood as lk

STOP_RULES = ["first_loss", "two_losses"]
SPEEDS = ["Bullet", "Blitz", "Classical"]
RULES = STOP_RULES
pd_ = ["player", "day"]


def apply_rules(d, keep):
    """d 為每位玩家每局一列（player、day、resid、result、speed、opponent、novice_rd，依玩家與時間排序），
    加上兩條規則刪去的局 del_<rule>（觸發的那一局本身保留）；keep 為評估段的條件。回傳 (d, del_days)：
    del_days 每天一列，next_resid 為下一個有對局的一天的第一局殘差（next_opponent 為該局的對手），先在整段序列上找，
    再限於評估段。"""
    loss = (pl.col("result") == "loss").cast(pl.Int32)
    d = (
        d.with_columns(run=(pl.col("result") != "loss").cum_sum().over(pd_))
        .with_columns(streak=loss.cum_sum().over(pd_ + ["run"]))
        .with_columns(
            del_first_loss=(loss.cum_sum().over(pd_) - loss) > 0,
            del_two_losses=((pl.col("streak") >= 2).cast(pl.Int32).cum_sum().over(pd_) - (pl.col("streak") >= 2).cast(pl.Int32)) > 0,
        )
    )
    first = d.group_by(pd_, maintain_order=True).agg(pl.col("resid").first(), pl.col("speed").first(), pl.col("opponent").first())
    first = first.with_columns(
        next_resid=pl.col("resid").shift(-1).over("player"), next_speed=pl.col("speed").shift(-1).over("player"), next_opponent=pl.col("opponent").shift(-1).over("player")
    ).select(pd_ + ["next_resid", "next_speed", "next_opponent"])
    d = d.filter(keep).drop("run", "streak")
    del_days = d.group_by(pd_).agg(*[pl.col(f"del_{r}").any().alias(f"any_{r}") for r in STOP_RULES]).join(first, on=pd_, how="left")
    return d, del_days


def backtest(d, del_days, B=2000, speeds=SPEEDS, family=("Bullet", "Blitz")):
    """d、del_days 為 apply_rules 的輸出；speeds 為要報告的速度類別，family 為 Holm 校正的一族（新手分組為 all 的格）。回傳 (res, reset, est, boot, col, rcol)：
    res 為各規則、速度類別與新手分組的刪去減保留，reset 為隔天第一局減刪去的局；
    est、boot 為各格的估計與 bootstrap 值，col、rcol 為格的索引。"""
    # 每位玩家各格的殘差和與局數；bootstrap 以玩家為單位重抽，對這兩張矩陣加權即可
    players = d["player"].unique().sort()
    P = len(players)
    pi = np.searchsorted(players.to_numpy(), d["player"].to_numpy())
    cells = []
    resid = d["resid"].to_numpy()
    spd = d["speed"].to_numpy()
    nov = d["novice_rd"].to_numpy()
    for r in RULES:
        dl = d[f"del_{r}"].to_numpy()
        for s in speeds:
            for g, gm in [("all", np.ones(len(d), bool)), ("novice", nov), ("established", ~nov)]:
                m = (spd == s) & gm
                cells += [(r, s, g, "deleted", m & dl), (r, s, g, "kept", m & ~dl), (r, s, g, "all", m)]
    # 隔天重置：有刪去局的天，其刪去局對下一個 arena 天的第一局
    dd = d.join(del_days, on=pd_)
    dpi = np.searchsorted(players.to_numpy(), dd["player"].to_numpy())
    nd = del_days.filter(pl.col("next_resid").is_not_null())
    npi = np.searchsorted(players.to_numpy(), nd["player"].to_numpy())
    reset_cells = []
    for r in RULES:
        for s in speeds:
            reset_cells.append((r, s, "deleted", dpi, dd["resid"].to_numpy(), (dd[f"del_{r}"] & (dd["speed"] == s)).to_numpy()))
            reset_cells.append((r, s, "next_day_first", npi, nd["next_resid"].to_numpy(), (nd[f"any_{r}"] & (nd["next_speed"] == s)).to_numpy()))

    S = np.zeros((P, len(cells) + len(reset_cells)))
    N = np.zeros_like(S)
    for k, (*_, m) in enumerate(cells):
        S[:, k] = np.bincount(pi[m], resid[m], P)
        N[:, k] = np.bincount(pi[m], None, P)
    for j, (_, _, _, idx, x, m) in enumerate(reset_cells):
        k = len(cells) + j
        S[:, k] = np.bincount(idx[m], x[m], P)
        N[:, k] = np.bincount(idx[m], None, P)

    rng = np.random.default_rng(0)
    boot = []
    for c in range(0, B, 100):
        W = rng.multinomial(P, np.full(P, 1 / P), size=min(100, B - c)).astype(np.float64)
        boot.append((W @ S) / (W @ N))
    boot = np.concatenate(boot)
    est = S.sum(0) / N.sum(0)
    col = {(r, s, g, k): i for i, (r, s, g, k, _) in enumerate(cells)}
    rcol = {(r, s, k): len(cells) + j for j, (r, s, k, *_) in enumerate(reset_cells)}


    def summarize(a, b, extra):
        diff, bd = est[a] - est[b], boot[:, a] - boot[:, b]
        se = np.nanstd(bd, ddof=1)
        return extra | {"diff": diff, "se": se, "lo": np.nanquantile(bd, 0.025), "hi": np.nanquantile(bd, 0.975), "p": 2 * norm.sf(abs(diff) / se)}


    rows = []
    for r in RULES:
        for s in speeds:
            for g in ["all", "novice", "established"]:
                i_d, i_k, i_a = col[(r, s, g, "deleted")], col[(r, s, g, "kept")], col[(r, s, g, "all")]
                row = summarize(i_d, i_k, {"rule": r, "speed": s, "group": g, "n_deleted": int(N[:, i_d].sum()), "n_kept": int(N[:, i_k].sum()), "mean_deleted": est[i_d], "mean_kept": est[i_k]})
                bv = boot[:, i_k] - boot[:, i_a]
                row |= {"kept_minus_never": est[i_k] - est[i_a], "kept_minus_never_se": np.nanstd(bv, ddof=1)}
                rows.append(row)
    res = pl.DataFrame(rows)
    fam = (res["group"] == "all") & res["speed"].is_in(list(family))
    res = res.with_columns(p_holm=lk.holm(res["p"], fam))

    reset = pl.DataFrame([
        summarize(rcol[(r, s, "next_day_first")], rcol[(r, s, "deleted")], {
            "rule": r, "speed": s, "n_deleted": int(N[:, rcol[(r, s, "deleted")]].sum()), "n_next_day_first": int(N[:, rcol[(r, s, "next_day_first")]].sum()),
            "mean_deleted": est[rcol[(r, s, "deleted")]], "mean_next_day_first": est[rcol[(r, s, "next_day_first")]],
        })
        for r in RULES for s in speeds
    ])
    return res, reset, est, boot, col, rcol
