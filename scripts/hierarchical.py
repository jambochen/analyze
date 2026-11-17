# Gee 等人的相關玩家 intercept/slope；binary 原式與本研究的雙方 ordinal 延伸共用 prior。
import numpy as np
import pymc as pm
import pytensor.tensor as pt


def player_effects(n, slope=True):
    """非中心化的 Normal 玩家偏差；二維 rho 的 Beta(2,2) 與 LKJ(2) 等價。"""
    sd = pm.HalfNormal("sd", 1, shape=2 if slope else 1)
    z = pm.Normal("z", 0, 1, shape=(n, 2 if slope else 1))
    if slope:
        rho = pm.Deterministic("rho", 2*pm.Beta("rho_unit", 2, 2)-1)
        u = pt.stack([sd[0]*z[:, 0], sd[1]*(rho*z[:, 0]+pt.sqrt(1-rho**2)*z[:, 1])], axis=1)
    else:
        u = z*sd
    return pm.Deterministic("u", u)


def gee_model(X, h, player, y):
    """原論文 binary likelihood 與全部 hyperprior，輸入為顏色、原尺度評分差及中心化近況。"""
    with pm.Model() as model:
        sigma = pm.HalfNormal("sigma", 1, shape=3)
        gamma = pm.Normal("gamma", 0, sigma[:2], shape=2)
        mu_beta = pm.Normal("mu_beta", 0, sigma[2])
        u = player_effects(int(player.max())+1)
        eta = pt.dot(X, gamma)+u[player, 0]+(mu_beta+u[player, 1])*h
        pm.Bernoulli("result", logit_p=eta, observed=y)
    return model


def design(d):
    """與簡單近況模型相同固定效果；尚未白化的優勢及 log-width 設計矩陣。"""
    xr, xd, gx, sign = [d[c].to_numpy() for c in ["xr", "xd", "gxd", "sign"]]
    X = np.column_stack([sign, sign*xr, sign*xr**2, xd, xr*xd, gx,
                         *[d[c].to_numpy() for c in ["d_prev_draw", "d_prev_win", "opp_missing", "d_dr"]]])
    Z = np.column_stack([np.ones(d.height), xr, xr**2])
    return X, Z


def whiten(X):
    """僅使用訓練共變數的可逆變換，降低評分差與 RD 收縮評分差的共線性。"""
    _, s, vt = np.linalg.svd(X, full_matrices=False)
    assert s.min() > 1e-8
    T = vt.T/s*np.sqrt(len(X))
    return X@T, T


def ordinal_logp(eta, width, y):
    """與 ordinal.ll 相同的穩定三類 likelihood，供 PyTensor/NUTS 自動微分。"""
    upper, lower = eta+width, eta-width
    ld = -pt.softplus(-upper)-pt.softplus(lower)+pt.log(-pt.expm1(-2*width))
    return pt.where(pt.eq(y, 2), -pt.softplus(-lower), pt.where(pt.eq(y, 0), -pt.softplus(upper), ld))


def ordinal_model(X, Z, a, b, ha, hb, y, slope=True):
    """共同比較的雙方模型；每個真實對局恰一列。"""
    with pm.Model() as model:
        beta = pm.Normal("beta", 0, 2.5, shape=X.shape[1])
        # 白化前的中心是固定和局帶寬 exp(-2)，不以結果設定 prior。
        mean_width = np.linalg.lstsq(Z, np.full(len(Z), -2.), rcond=None)[0]
        width_beta = pm.Normal("width_beta", mean_width, 1., shape=Z.shape[1])
        u = player_effects(int(max(a.max(), b.max()))+1, slope)
        eta = pt.dot(X, beta)+u[a, 0]-u[b, 0]
        if slope:
            eta += u[a, 1]*ha-u[b, 1]*hb
        pm.Potential("result", ordinal_logp(eta, pt.exp(pt.dot(Z, width_beta)), y).sum())
    return model
