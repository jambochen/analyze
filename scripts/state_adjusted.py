# 依狀態調整的勝率的樣本、模型的機率與評估量，state_adjusted_probs.py（2014 年估計與評估）與
# prospective_validation.py（2015 年前瞻驗證）共用：兩者的樣本與機率若各寫一份，驗證的就不是同一個模型。

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl

import likelihood as lk
import novice
import ordinal
from arena_batches import player_sample_rd
from runs import runs_before

# 訓練好的模型（有序 logistic 的係數、LightGBM、XGBoost、MLP），由 state_adjusted_probs.py 寫出
MODEL_DIR = "data/processed/state_adjusted_models"
# LightGBM 與 XGBoost 的輸入：主基準的輸入加近況，全為對局前的資訊
GBM_FEATURES = ["elo", "opp_elo", "rd", "opp_rd", "white", "eta", "w", "prev1", "prev2", "prev3", "dr", "opp_prev1", "opp_dr", "games_in_tour"]


def sample(period="2014"):
    """period（arena_batches.PERIODS 的鍵）的 arena 玩家局中本場有上一局者，加上雙方的近況。"""
    grp = ["player", "tournament_id"]
    code = pl.col("result").replace_strict({"loss": 0, "draw": 1, "win": 2}, return_dtype=pl.Float64)
    base = (
        player_sample_rd(period)
        .sort("player", "tournament_id", "start", "game_id")
        .with_columns(
            prev_result=pl.col("result").shift(1).over(grp),
            prev2=code.shift(2).over(grp),
            prev3=code.shift(3).over(grp),
            prev1=code.shift(1).over(grp),
            dr=((pl.col("elo") - pl.col("elo").shift(1).over(grp)) / 100).cast(pl.Float64),
        )
    )
    # 對手的近況取自對手在同一局的那一列；對手在這場沒有上一局時 opp_missing 為 1，近況項記為 0
    opp = base.select("game_id", opponent="player", opp_prev_result="prev_result", opp_prev1="prev1", opp_dr="dr")
    return (
        base.join(opp, on=["game_id", "opponent"], how="left")
        .filter(pl.col("prev_result").is_not_null())
        .with_columns(
            y=(pl.col("score") * 2).cast(pl.Int64),
            prev_draw=(pl.col("prev_result") == "draw").cast(pl.Float64),
            prev_win=(pl.col("prev_result") == "win").cast(pl.Float64),
            opp_missing=pl.col("opp_prev_result").is_null().cast(pl.Float64),
            opp_prev_draw=(pl.col("opp_prev_result") == "draw").fill_null(False).cast(pl.Float64),
            opp_prev_win=(pl.col("opp_prev_result") == "win").fill_null(False).cast(pl.Float64),
            opp_dr=pl.col("opp_dr").fill_null(0.0),
            novice_rd=novice.novice_rd("rd", "opp_rd"),
            white=(pl.col("color") == "white").cast(pl.Float64),
        )
        .with_columns(
            d_prev_draw=pl.col("prev_draw") - pl.col("opp_prev_draw"),
            d_prev_win=pl.col("prev_win") - pl.col("opp_prev_win"),
            d_dr=pl.col("dr") - pl.col("opp_dr"),
        )
    )


# sym_* 乘雙方之差，即兩人共用的係數
PARAMS = ["arena_shift", "arena_log_draw_width", "prev_draw", "prev_win", "dr", "sym_prev_draw", "sym_prev_win", "sym_dr", "opp_missing"]
XCOLS = ["prev_draw", "prev_win", "dr", "d_prev_draw", "d_prev_win", "d_dr", "opp_missing"]


# 主基準的 eta、w 為 offset。t0 平移 eta、t1 縮放和局帶寬：主基準在全站對局上估計，
# arena 的玩家與大廳不同，這兩項吸收 arena 的整體差異
def ll(t, eta, w, X, y):
    return ordinal.ll(eta + t[0] + X @ t[2:], w * jnp.exp(t[1]), y)


def probs(t, eta, w, X):
    return ordinal.probs(eta + t[0] + X @ t[2:], w * np.exp(t[1]))


# MLP 的輸出層給線性預測式與和局帶寬的對數，經與有序 logistic 相同的累積 logit 得到機率
def mlp_out(p, X):
    h = jax.nn.relu(X @ p[0] + p[1])
    h = jax.nn.relu(h @ p[2] + p[3])
    o = h @ p[4] + p[5]
    return o[:, 0], jnp.exp(o[:, 1])


def mlp_x(d):
    cont = d.select("elo", "opp_elo", "rd", "opp_rd", "eta", "w", "dr", "opp_dr", pl.col("games_in_tour").log1p()).to_numpy().astype(np.float32)
    # 前幾局不存在時歸入第四類，不以 0 或平均填補
    cat = np.concatenate([np.eye(4, dtype=np.float32)[d[c].fill_null(3).cast(pl.Int64).to_numpy()] for c in ["prev1", "prev2", "prev3", "opp_prev1"]], 1)
    return cont, np.c_[d["white"].to_numpy().astype(np.float32), cat]


def losses(p, y):
    O = np.eye(3)[y]
    C, Co = np.cumsum(p, 1)[:, :2], np.cumsum(O, 1)[:, :2]
    return {"log_loss": -np.log(p[np.arange(len(y)), y]), "rps": ((C - Co) ** 2).sum(1) / 2}


def diff_row(spd, a, b, la, lb, msk, wg, bg, group):
    rows = []
    for metric in ["log_loss", "rps"]:
        dlt = la[metric][msk] - lb[metric][msk]
        se = lk.dyad_se((dlt - dlt.mean()) / msk.sum(), wg[msk], bg[msk])
        rows.append({"speed": spd, "group": group, "model": a, "vs": b, "metric": metric, "n": int(msk.sum()), "mean": la[metric][msk].mean(), "diff": dlt.mean(), "diff_se_dyadic": se})
    return rows


def eval_rows():
    """2014 下半年 arena 中本場有上一局的玩家局，加上三個預報：e_sys（Elo 形式，表頭等級分）、e_base（主基準）、
    e_adj（依狀態調整的勝率，2014 上半年估計的模型），與本局之前同場內的連敗、連勝局數 loss_run、win_run。
    連串在整場錦標賽的序列上計算，之前的局可以落在上半年。"""
    grp = ["player", "tournament_id"]
    full = runs_before(player_sample_rd().sort(grp + ["start", "game_id"]), grp).select("game_id", "player", "loss_run", "win_run")
    d = sample().filter(pl.col("start") >= pl.datetime(2014, 7, 1, time_zone="UTC"))
    parts = []
    for spd in ["Bullet", "Blitz", "Classical"]:
        te = d.filter(pl.col("speed") == spd)
        th = np.load(f"{MODEL_DIR}/{spd}.npz")["th"]
        p = probs(th, te["eta"].to_numpy(), te["w"].to_numpy(), te.select(XCOLS).to_numpy().astype(np.float64))
        parts.append(te.with_columns(e_adj=pl.Series(p[:, 2] + p[:, 1] / 2)))
    return (
        pl.concat(parts)
        .with_columns(e_sys=1 / (1 + 10 ** (-(pl.col("elo") - pl.col("opp_elo")) / 400)))
        .join(full, on=["game_id", "player"])
        .select("game_id", "player", "opponent", "speed", "score", "e_sys", "e_base", "e_adj", "loss_run", "win_run", novice="novice_rd")
    )
