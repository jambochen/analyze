# 校準後的期望得分：取 calibration.py 估得的有序模型（分速度類別），供各分析共用。
# 匹配品質與勝和負機率都要用同一個校準模型，複本會漂移，故集中在此。
#
# 模型：x_r = (平均等級分 - 1500) / 100，x_d = (白方 - 黑方) / 100
#   eta = t0 + t1 x_r + t2 x_r^2 + (t3 + t4 x_r) x_d
#   w   = exp(t5 + t6 x_r + t7 x_r^2)
#   P(白勝) = sigmoid(eta - w)，P(白不輸) = sigmoid(eta + w)

import polars as pl

import ordinal

PARAMS = [
    "white_adv", "white_adv_x_rating", "white_adv_x_rating2",
    "slope_diff", "slope_diff_x_rating",
    "log_draw_width", "log_draw_width_x_rating", "log_draw_width_x_rating2",
]

coef = (
    pl.read_csv("outputs/tables/calibration_coef.csv")
    .filter(pl.col("model") == "ordinal", pl.col("param").is_in(PARAMS))
    .pivot(on="param", index="speed", values="est")
    .select("speed", *PARAMS)
)


# 主基準：加入雙方 RD 的有序 logistic（expected_score_baselines.py 的 rd_ordinal，全部對局）。
# 在上式的 eta 再加 t8 g(sigma) x_d，sigma 為雙方 RD 的合成
RD_PARAMS = PARAMS + ["slope_g_diff"]
rd_coef = (
    pl.read_csv("outputs/tables/expected_score_baselines_coef.csv")
    .filter(pl.col("model") == "rd_ordinal", pl.col("sample") == "all")
    .pivot(on="param", index="speed", values="est")
    .select("speed", *RD_PARAMS)
)


def _white_view():
    # 列是自身視角，係數是白方視角；兩個係數表的平均等級分與和局帶寬參數同名
    is_white = pl.col("color") == "white"
    w_elo = pl.when(is_white).then(pl.col("elo")).otherwise(pl.col("opp_elo"))
    b_elo = pl.when(is_white).then(pl.col("opp_elo")).otherwise(pl.col("elo"))
    xr = ((w_elo + b_elo) / 2 - 1500) / 100
    xd = (w_elo - b_elo) / 100
    w = (pl.col("log_draw_width") + pl.col("log_draw_width_x_rating") * xr + pl.col("log_draw_width_x_rating2") * xr**2).exp()
    return is_white, xr, xd, w


def rd_baseline(df):
    """對 player_games 的列（須含 rd、opp_rd），依自身視角加上主基準的 eta、w、e_base 與 slope_elo。

    P(勝) = sigmoid(eta - w)，P(不輸) = sigmoid(eta + w)。slope_elo 為自身等級分每多 1 分 eta 增加多少
    （分差斜率，不含平均等級分項），用於把效果換算成等級分。
    """
    is_white, xr, xd, w = _white_view()
    sign = pl.when(is_white).then(1.0).otherwise(-1.0)
    slope = pl.col("slope_diff") + pl.col("slope_diff_x_rating") * xr + pl.col("slope_g_diff") * ordinal.g("rd", "opp_rd")
    eta = pl.col("white_adv") + pl.col("white_adv_x_rating") * xr + pl.col("white_adv_x_rating2") * xr**2 + slope * xd
    return (
        df.join(rd_coef.lazy() if isinstance(df, pl.LazyFrame) else rd_coef, on="speed", how="left")
        .with_columns(eta=sign * eta, w=w, slope_elo=slope / 100)
        .with_columns(e_base=(1 / (1 + (-(pl.col("eta") - pl.col("w"))).exp()) + 1 / (1 + (-(pl.col("eta") + pl.col("w"))).exp())) / 2)
        .drop(RD_PARAMS)
    )


def probs(df, color=True):
    """對 player_games 的列，依自身視角加上 p_win、p_draw、p_loss、e_cal 四欄。

    color=False 時去掉白方優勢（t0..t2 設 0），得到只反映等級分差距的期望得分：
    匹配品質要衡量的是配對本身的懸殊程度，執子顏色另作解釋變數。
    """
    is_white, xr, xd, w = _white_view()
    adv = (pl.col("white_adv") + pl.col("white_adv_x_rating") * xr + pl.col("white_adv_x_rating2") * xr**2) if color else pl.lit(0.0)
    eta = adv + (pl.col("slope_diff") + pl.col("slope_diff_x_rating") * xr) * xd
    p_white_win = 1 / (1 + (-(eta - w)).exp())
    p_white_not_loss = 1 / (1 + (-(eta + w)).exp())
    p_draw = p_white_not_loss - p_white_win
    p_win = pl.when(is_white).then(p_white_win).otherwise(1 - p_white_not_loss)
    p_loss = 1 - p_win - p_draw
    return (
        df.join(coef.lazy() if isinstance(df, pl.LazyFrame) else coef, on="speed", how="left")
        .with_columns(p_win=p_win, p_draw=p_draw, p_loss=p_loss)
        .with_columns(e_cal=pl.col("p_win") + pl.col("p_draw") / 2)
        .drop(PARAMS)
    )
