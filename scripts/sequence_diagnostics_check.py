# 用全部可能賽果核對視窗摘要，涵蓋和局、分段及同局雙方相反結果。
from itertools import product

import numpy as np

from sequence_summary import summarize, window_sum

p = np.array([[.2, .1, .7], [.5, .2, .3], [.3, .4, .3]])
# 兩位玩家，各三局；同一實際對局反轉視角。
pp = np.r_[p, p[:, ::-1]]
e = pp[:, 2] + pp[:, 1] / 2
v = pp[:, 2] + pp[:, 1] / 4 - e**2
windows = [(metric, 2, np.array([0, 1, 3, 4]), [np.ones(4, bool)], [np.ones(4)/4]) for metric in ["loss_run", "cumulative_square"]]
mean = np.zeros(4)
for yy in product(range(3), repeat=3):
    yy = np.array(yy)
    y = np.r_[yy, 2-yy] / 2
    probability = np.prod(p[np.arange(3), yy])
    z = summarize(y, e, v, windows)
    brute = []
    for metric, k, valid, _, _ in windows:
        vals = [float((y[i:i+k] == 0).all()) if metric == "loss_run" else ((y[i:i+k]-e[i:i+k]).sum()**2 / v[i:i+k].sum()) for i in valid]
        brute.extend([np.mean(vals)]*2)
    assert np.allclose(z, brute)
    mean += probability * z
loss_reference = np.mean([pp[i, 0]*pp[i+1, 0] for i in [0, 1, 3, 4]])
assert np.allclose(mean, [loss_reference, loss_reference, 1, 1])
assert np.array_equal(window_sum(np.array([0, 1, 0, 0, 1]), 2), [1, 1, 0, 1])
print("enumeration, mirrored outcomes, draw breaks, boundaries, analytic expectations: passed")

# 等玩家權重不讓有更多視窗的人取得更多總權重；空分組保留缺值。
y = np.array([0., 0., 0., 1.])
w = [("loss_run", 1, np.arange(4), [np.ones(4, bool), np.zeros(4, bool)], [np.array([1/6, 1/6, 1/6, 1/2]), np.array([])])]
z = summarize(y, np.full(4, .5), np.full(4, .25), w)
assert np.allclose(z[:2], [.75, .5]) and np.isnan(z[2:]).all()
segment = np.array([1, 1, 1, 2, 2, 3])
assert np.array_equal(np.flatnonzero(segment[1:] == segment[:-1]), [0, 1, 3])
print("unequal player weights, empty groups, discontinuous windows: passed")
