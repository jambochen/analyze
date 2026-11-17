# 事件序列共用於參考實作、調參與正式預測；正值為開局，負值為賽果可用。
import numpy as np
from scipy.special import ndtr


def replay_arrays(d, cutoff_ms):
    """對局轉為密集玩家代碼與事件序列；同時發生時先預測再更新。"""
    n = len(d)
    _, inverse = np.unique(np.r_[d['white_id'].to_numpy(), d['black_id'].to_numpy()], return_inverse=True)
    times = np.r_[d['created_ms'].to_numpy(), d['last_move_ms'].to_numpy()]
    types = np.r_[np.zeros(n, dtype=int), np.ones(n, dtype=int)]
    indices = np.r_[np.arange(1, n+1), -np.arange(1, n+1)]
    order = np.lexsort((np.abs(indices), types, times))
    order = order[times[order] < cutoff_ms]
    return (np.ascontiguousarray(inverse[:n], dtype=np.int64), np.ascontiguousarray(inverse[n:], dtype=np.int64),
            d['y'].to_numpy().astype(np.int64), indices[order].astype(np.int64))


def reference(a, b, y, events, config):
    """Python TrueSkill 套件與 SciPy CDF 作獨立數值參考，不用於大量候選重放。"""
    import trueskill
    sigma, tau, draw, eps = config
    env = trueskill.TrueSkill(mu=0, sigma=sigma, beta=1, tau=tau, draw_probability=draw, backend='scipy')
    state = [env.create_rating() for _ in range(int(max(a.max(), b.max()))+1)]
    p = np.zeros((len(a), 3))
    for event in events:
        i = abs(event)-1
        sa, sb = state[a[i]], state[b[i]]
        if event > 0:
            c = np.sqrt(sa.sigma**2+sb.sigma**2+2*tau**2+2)
            diff = sa.mu-sb.mu
            p[i] = [ndtr((-eps-diff)/c), ndtr((eps-abs(diff))/c)-ndtr((-eps-abs(diff))/c), ndtr((diff-eps)/c)]
        elif y[i] == 0:
            state[b[i]], state[a[i]] = env.rate_1vs1(sb, sa)
        else:
            state[a[i]], state[b[i]] = env.rate_1vs1(sa, sb, drawn=y[i] == 1)
    return p
