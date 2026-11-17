import numpy as np
from nutpie.compiled_pyfunc import from_pyfunc
from likelihood_rs import hierarchical_logp


def compiled_model(X, Z, a, b, ha, hb, y, slope=True, blocks=16):
    """完整 posterior 與 gradient 由 Rust 計算，Python 僅建立 sampler 的資料及輸出介面。"""
    X, Z = [np.ascontiguousarray(v, dtype=np.float64) for v in [X, Z]]
    a, b, y = [np.ascontiguousarray(v, dtype=np.int64) for v in [a, b, y]]
    ha, hb = [np.ascontiguousarray(v, dtype=np.float64) for v in [ha, hb]]
    players = int(max(a.max(), b.max()))+1
    p, q, k = X.shape[1], Z.shape[1], 2 if slope else 1
    mean_width = np.linalg.lstsq(Z, np.full(len(Z), -2.), rcond=None)[0]
    z_start = p+q+k
    ndim = z_start+players*k+int(slope)

    def make_logp():
        def logp(theta):
            return hierarchical_logp(theta, X, Z, a, b, ha, hb, y, mean_width, players, slope, blocks)
        return logp

    def make_expand(seed1, seed2, chain):
        def expand(theta):
            sd = np.exp(theta[p+q:z_start])
            z = theta[z_start:z_start+players*k].reshape(players, k)
            u = z*sd
            out = {"beta": theta[:p].copy(), "width_beta": theta[p:p+q].copy(), "sd": sd, "z": z.copy()}
            if slope:
                t = theta[-1]
                rho = np.tanh(t/2)
                corr_scale = np.exp(-np.logaddexp(0., t)/2-np.logaddexp(0., -t)/2)*2
                u[:, 1] = sd[1]*(rho*z[:, 0]+corr_scale*z[:, 1])
                out["rho"] = np.array(rho)
            out["u"] = np.ascontiguousarray(u)
            return {name: out[name] for name in names}
        return expand

    def initial(seed):
        theta = np.random.default_rng(seed).uniform(-1., 1., ndim)
        theta[p:p+q] += mean_width
        return theta

    names = ["beta", "width_beta", "sd", "z", "u"]+(["rho"] if slope else [])
    shapes = [(p,), (q,), (k,), (players, k), (players, k)]+([()] if slope else [])
    return from_pyfunc(ndim, make_logp, make_expand, [np.dtype('float64')]*len(names), shapes, names,
                       make_initial_point_fn=initial)
