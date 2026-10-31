# arena 的 crossover 樣本與直接效果的有序 logistic，arena_crossover.py 與 player_heterogeneity.py 共用。
import jax.numpy as jnp
import numpy as np
import polars as pl

import likelihood as lk
import ordinal
import pairing_null
from arena_batches import player_sample_rd

SPEEDS = ["Bullet", "Blitz", "Classical"]
LAGS = [1, 2, 3]


def load():
    """有上一局的 arena 玩家局 t，兩局以上的批 m 與其重新配對的結構 ns。"""
    grp = ["player", "tournament_id"]
    d = player_sample_rd().with_columns(
        y=pl.col("result").replace_strict({"loss": 0, "draw": 1, "win": 2}, return_dtype=pl.Int8),
        sp=pl.col("speed").replace_strict({s: i for i, s in enumerate(SPEEDS)}, return_dtype=pl.Int64),
    )

    m, ns = pairing_null.structure(d)
    m = m.with_row_index("midx")


    d = (
        d.join(m.select("game_id", "player", pl.col("midx").cast(pl.Int64)), on=["game_id", "player"], how="left")
        .with_columns(pl.col("midx").fill_null(-1))
        .sort("player", "tournament_id", "start", "game_id")
        .with_columns(
            *[pl.col("h").shift(k).over(grp).alias(f"h_lag{k}") for k in LAGS],
            *[pl.col("midx").shift(k).over(grp).alias(f"midx_lag{k}") for k in LAGS],
            midx_lead1=pl.col("midx").shift(-1).over(grp),
            *[pl.col("layer").shift(k).over(grp).alias(f"layer_lag{k}") for k in LAGS],
            *[pl.col("rule").shift(k).over(grp).alias(f"rule_lag{k}") for k in LAGS],
            bid_lag1=pl.col("bid").shift(1).over(grp),
            score_lag1=pl.col("score").shift(1).over(grp),
            opp_lag1=pl.col("opponent").shift(1).over(grp),
        )
    )
    t = d.filter(pl.col("h_lag1").is_not_null())
    return t, m, ns


def direct_columns(t):
    return t.with_columns(
        sign=pl.when(pl.col("color") == "white").then(1.0).otherwise(-1.0),
        xr=((pl.col("elo") + pl.col("opp_elo")) / 2 - 1500) / 100,
        xd=((pl.col("elo") - pl.col("opp_elo")) / 100).cast(pl.Float64),
    ).with_columns(
        sxr=pl.col("sign") * pl.col("xr"),
        sxr2=pl.col("sign") * pl.col("xr") ** 2,
        xdxr=pl.col("xd") * pl.col("xr"),
        gxd=pl.col("xd") * ordinal.g("rd", "opp_rd"),
        elo_c2=pl.col("elo_c") ** 2,
        elo_c3=pl.col("elo_c") ** 3,
        ebar_xd=pl.col("e_bar") * pl.col("xd"),
        h_ebar=pl.col("h_lag1") * pl.col("e_bar"),
    )


# 直接效果的有序 logistic。t[0] 截距，t[1:NX+1] 各共變數，t[NX+1:NX+4] 和局帶寬的對數（常數、平均等級分、其平方），
# t[NX+4] partial proportional odds：分差對「勝」門檻的斜率位移。未列入 free 的參數固定為 0
# 本場已下局數：12 人以下的批優先配對等待最久者，它與 h 的平衡不通過
BASE = ["sign", "sxr", "sxr2", "xd", "xdxr", "gxd", "elo_c", "elo_c2", "log_games_in_tour", "e_bar"]
EXTRA = ["h_lag1", "h_lag2", "h_lag3", "elo_c3", "ebar_xd", "h_ebar"]
XCOLS = BASE + EXTRA
NX = len(XCOLS)
P_DRAW, P_PPO = NX + 1, NX + 4


def ll(th, X, xr, xd, y):
    eta = th[0] + X @ th[1 : NX + 1]
    return ordinal.ll(eta, jnp.exp(th[P_DRAW] + th[P_DRAW + 1] * xr + th[P_DRAW + 2] * xr**2), y, shift=th[P_PPO] * xd)


def free_of(cols, ppo=False):
    return np.r_[0, [1 + XCOLS.index(c) for c in cols], P_DRAW, P_DRAW + 1, P_DRAW + 2, [P_PPO] if ppo else []].astype(int)


T0 = np.zeros(NX + 5)
T0[P_DRAW] = -2.5


def args_of(s):
    X = s.select(XCOLS).fill_null(0.0).to_numpy().astype(np.float64)
    return (X, s["xr"].to_numpy(), s["xd"].to_numpy(), s["y"].to_numpy().astype(np.float64))


def direct_resid(ts):
    """該速度類別的不含 h 的直接效果模型的殘差（得分減期望得分）。"""
    a_s = args_of(ts)
    # 以該速度類別全部有上一局的列估計
    th0, *_ = lk.fit(ll, T0, free_of(BASE), a_s, ts["player"].to_numpy(), ts["opponent"].to_numpy())
    X, xr, xd, _ = a_s
    eta = th0[0] + X @ th0[1 : NX + 1]
    pw, pn = ordinal.cum(eta, np.exp(th0[P_DRAW] + th0[P_DRAW + 1] * xr + th0[P_DRAW + 2] * xr**2))
    e_dir = (pw + pn) / 2
    return ts["score"].to_numpy() - e_dir
