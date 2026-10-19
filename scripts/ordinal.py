# 勝和負三項結果的有序 logistic：P(勝) = F(eta - w)，P(不輸) = F(eta + w)，w > 0 為和局帶寬的一半。
# eta 的共變數由各分析自行組合；概似、機率、對 eta 的 score 與 Glicko 的 g(sigma) 只寫在這裡，
# 和局項的數值穩定寫法與 PPO 的截斷若各自複製，漂移不會報錯。y 編碼為負 0、和 1、勝 2。

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl

jax.config.update("jax_enable_x64", True)

Q = np.log(10) / 400


def g(rd, opp_rd):
    """Glickman 對一局結果的收縮係數 g(sigma)，polars 運算式；sigma^2 為雙方 RD 的平方和，rd、opp_rd 為欄名。"""
    s2 = pl.col(rd).cast(pl.Float64) ** 2 + pl.col(opp_rd).cast(pl.Float64) ** 2
    return 1 / (1 + 3 * Q**2 * s2 / np.pi**2).sqrt()


def ll(eta, w, y, shift=0.0):
    """逐列的對數概似（jax）。shift 為 partial proportional odds 對「勝」門檻的位移：P(勝) = F(eta - w + shift)。
    位移大到兩門檻交錯時和局機率為負，以極小值截住，概似因此極低，不會被選中。"""
    u, v = eta + w, eta - w + shift
    # F(u) - F(v) = F(u) F(-v) (1 - e^{-(u-v)})，避免和局率很低時相減失去精度
    ldraw = jax.nn.log_sigmoid(u) + jax.nn.log_sigmoid(-v) + jnp.log(-jnp.expm1(-jnp.maximum(2 * w - shift, 1e-12)))
    return jnp.where(y == 2, jax.nn.log_sigmoid(v), jnp.where(y == 0, jax.nn.log_sigmoid(-u), ldraw))


def cum(eta, w):
    """(P(勝), P(不輸))，numpy。"""
    return 1 / (1 + np.exp(-(eta - w))), 1 / (1 + np.exp(-(eta + w)))


def probs(eta, w):
    """每列 [負, 和, 勝] 的機率，numpy。"""
    pw, pn = cum(eta, w)
    return np.stack([1 - pn, pn - pw, pw], 1)


def score_info(eta, w, y):
    """對數概似對 eta 的逐列 score 與 information（封閉形式），numpy。"""
    hi, nl = cum(eta, w)
    s = np.where(y == 2, 1 - hi, np.where(y == 0, -nl, 1 - hi - nl))
    v = np.where(y == 2, hi * (1 - hi), np.where(y == 0, nl * (1 - nl), hi * (1 - hi) + nl * (1 - nl)))
    return s, v
