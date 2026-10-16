# 校準後的期望得分：取 calibration.py 估得的有序模型（分時間類別），供各分析共用。
# 匹配品質與勝和負機率都要用同一個校準模型，複本會漂移，故集中在此。
#
# 模型：x_r = (平均等級分 - 1500) / 100，x_d = (白方 - 黑方) / 100
#   eta = t0 + t1 x_r + t2 x_r^2 + (t3 + t4 x_r) x_d
#   w   = exp(t5 + t6 x_r + t7 x_r^2)
#   P(白勝) = sigmoid(eta - w)，P(白不輸) = sigmoid(eta + w)

import polars as pl

PARAMS = [
    "white_adv", "white_adv_x_rating", "white_adv_x_rating2",
    "slope_diff", "slope_diff_x_rating",
    "log_draw_width", "log_draw_width_x_rating", "log_draw_width_x_rating2",
]

coef = (
    pl.read_csv("outputs/tables/calibration_coef.csv")
    .filter(pl.col("model") == "ordinal", pl.col("param").is_in(PARAMS))
    .pivot(on="param", index="time_class", values="est")
    .select("time_class", *PARAMS)
)


def probs(df, color=True):
    """對 player_games 的列，依自身視角加上 p_win、p_draw、p_loss、e_cal 四欄。

    color=False 時去掉白方優勢（t0..t2 設 0），得到只反映等級分差距的期望得分：
    匹配品質要衡量的是配對本身的懸殊程度，執子顏色另作解釋變數。
    """
    is_white = pl.col("color") == "white"
    w_elo = pl.when(is_white).then(pl.col("elo")).otherwise(pl.col("opp_elo"))
    b_elo = pl.when(is_white).then(pl.col("opp_elo")).otherwise(pl.col("elo"))
    xr = ((w_elo + b_elo) / 2 - 1500) / 100
    xd = (w_elo - b_elo) / 100
    adv = (pl.col("white_adv") + pl.col("white_adv_x_rating") * xr + pl.col("white_adv_x_rating2") * xr**2) if color else pl.lit(0.0)
    eta = adv + (pl.col("slope_diff") + pl.col("slope_diff_x_rating") * xr) * xd
    w = (pl.col("log_draw_width") + pl.col("log_draw_width_x_rating") * xr + pl.col("log_draw_width_x_rating2") * xr**2).exp()
    p_white_win = 1 / (1 + (-(eta - w)).exp())
    p_white_not_loss = 1 / (1 + (-(eta + w)).exp())
    p_draw = p_white_not_loss - p_white_win
    p_win = pl.when(is_white).then(p_white_win).otherwise(1 - p_white_not_loss)
    p_loss = 1 - p_win - p_draw
    return (
        df.join(coef.lazy() if isinstance(df, pl.LazyFrame) else coef, on="time_class", how="left")
        .with_columns(p_win=p_win, p_draw=p_draw, p_loss=p_loss)
        .with_columns(e_cal=pl.col("p_win") + pl.col("p_draw") / 2)
        .drop(PARAMS)
    )
