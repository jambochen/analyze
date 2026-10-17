# 虛無假設下（每局獨立、殘差 ±0.5 等機率），「一路下到第一次輸」刪去減保留的差，以局加權與玩家等權各是多少。
# 保留的局以觸發的那一局輸棋結束，局數越少，這一局在玩家自己的平均中占比越大；局數本身由結果決定，
# 所以每位玩家的平均之差不以 0 為中心，以局加權的總和之比則受 optional stopping 保護

import numpy as np

rng = np.random.default_rng(0)
P = 20000
S_all, N_all, d = np.zeros(2), np.zeros(2), []
for i in range(P):
    S, N = np.zeros(2), np.zeros(2)
    for _ in range(rng.integers(1, 4)):
        r = rng.integers(0, 2, rng.integers(1, 8)) - 0.5
        deleted = (np.cumsum(r < 0) - (r < 0)) > 0
        for k in (0, 1):
            S[k] += r[deleted == k].sum()
            N[k] += (deleted == k).sum()
    S_all += S
    N_all += N
    if N.all():
        d.append(S[1] / N[1] - S[0] / N[0])
print("game", S_all[1] / N_all[1] - S_all[0] / N_all[0])
print("player", np.mean(d), np.std(d) / np.sqrt(len(d)))
