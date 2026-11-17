# 各 posterior chain 獨立計算，讓 NumPy 的積分工作可平行，避免啟動額外的配適行程。
import numpy as np
from scipy.special import expit
from numpy.polynomial.hermite import hermgauss


def game_rows(d, X, Z, ha, hb):
    """核對雙方設計的對稱性後，每局只作一次 posterior 積分。"""
    _, first, inverse = np.unique(d["game_id"].to_numpy(), return_index=True, return_inverse=True)
    a, b = d["a_id"].to_numpy(), d["b_id"].to_numpy()
    reverse = a != a[first][inverse]
    assert np.array_equal(a, np.where(reverse, b[first][inverse], a[first][inverse]))
    assert np.array_equal(b, np.where(reverse, a[first][inverse], b[first][inverse]))
    assert np.allclose(X, X[first][inverse]*np.where(reverse, -1, 1)[:, None], rtol=1e-12, atol=1e-12)
    assert np.allclose(Z, Z[first][inverse], rtol=1e-12, atol=1e-12)
    assert np.array_equal(ha, np.where(reverse, hb[first][inverse], ha[first][inverse]))
    assert np.array_equal(hb, np.where(reverse, ha[first][inverse], hb[first][inverse]))
    return first, inverse, reverse


def chain_probabilities(chain, samples, X, Z, ai, bi, ak, bk, ha, hb, kind):
    """固定 posterior 的單鏈預測；未見玩家以 Normal distribution 積分。"""
    beta, wb, u, sd = [samples[k][chain] for k in ["beta", "width_beta", "u", "sd"]]
    rho = samples["rho"][chain] if kind == "slope" else np.zeros(len(beta))
    nodes, weights = hermgauss(15)
    check_nodes, check_weights = hermgauss(31)
    out = np.zeros((len(X), 3))
    error, checked = 0., False
    for lo in range(0, len(X), 256):
        sl = slice(lo, lo+256)
        e = beta@X[sl].T+u[:, ai[sl], 0]*ak[sl]-u[:, bi[sl], 0]*bk[sl]
        width = np.exp(wb@Z[sl].T)
        var = sd[:, 0, None]**2*((~ak[sl]).astype(float)+(~bk[sl]).astype(float))
        if kind == "slope":
            e += u[:, ai[sl], 1]*ha[sl]*ak[sl]-u[:, bi[sl], 1]*hb[sl]*bk[sl]
            var += (sd[:, 1, None]**2*ha[sl]**2+2*(rho*sd[:, 0]*sd[:, 1])[:, None]*ha[sl])*(~ak[sl])
            var += (sd[:, 1, None]**2*hb[sl]**2+2*(rho*sd[:, 0]*sd[:, 1])[:, None]*hb[sl])*(~bk[sl])
        assert var.min() > -1e-12
        spread = np.sqrt(2*np.maximum(var, 0))
        unknown = (~ak[sl]) | (~bk[sl])
        pw, ploss = expit(e-width), expit(-e-width)
        if unknown.any():
            pw[:, unknown], ploss[:, unknown] = 0., 0.
            for node, weight in zip(nodes, weights):
                shifted = e[:, unknown]+node*spread[:, unknown]
                pw[:, unknown] += weight/np.sqrt(np.pi)*expit(shifted-width[:, unknown])
                ploss[:, unknown] += weight/np.sqrt(np.pi)*expit(-shifted-width[:, unknown])
            if not checked:
                check = sum(weight/np.sqrt(np.pi)*expit(e[:, unknown]+node*spread[:, unknown]-width[:, unknown])
                            for node, weight in zip(check_nodes, check_weights))
                error = float(np.abs(check-pw[:, unknown]).max())
                checked = True
        out[sl, 0] = ploss.mean(0)
        out[sl, 2] = pw.mean(0)
        out[sl, 1] = 1-out[sl, 0]-out[sl, 2]
    return out, error


def calibration_rows(y, probabilities, groups):
    """同時輸出自身分箱與基準共同分箱；不以不同分箱的 ECE 直接排名。"""
    rows = []
    for group, mask in groups:
        for target, observed in [("score", y/2), ("loss", y == 0), ("draw", y == 1), ("win", y == 2)]:
            values = {name: prob[:, 2]+prob[:, 1]/2 if target == "score" else prob[:, ["loss", "draw", "win"].index(target)]
                      for name, prob in probabilities.items()}
            for name, pred in values.items():
                for binning in ["own_prediction", "baseline_prediction"]:
                    anchor = pred if binning == "own_prediction" else values["baseline"]
                    bins = np.minimum((10*anchor).astype(int), 9)
                    for b in np.unique(bins[mask]):
                        keep = mask & (bins == b)
                        rows.append({"group": group, "model": name, "target": target, "binning": binning,
                                     "bin": b, "n": int(keep.sum()), "predicted": pred[keep].mean(), "observed": observed[keep].mean()})
    return rows
