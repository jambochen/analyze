# 開場用的機率示意：勝率恰為一半、各局獨立時，n 局中至少出現一次長度 ≥ 8 的連勝或連敗的機率。
# 對照玩家常用的「連 8 場的機率是 1/256」：那只是某個固定的 8 局區間全勝的機率。
# 純機率計算，不用資料。

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

K, N = 8, 300

# 狀態為目前同色連串的長度 1..K-1；任一局後連串達 K 即吸收
# 每局以 1/2 延續連串、1/2 換成另一種結果而重設為 1
dist = np.zeros(K)
dist[1] = 1.0
hit = [0.0]
for n in range(2, N + 1):
    new = np.zeros(K)
    new[1] = dist[1:].sum() / 2
    new[2:] = dist[1:-1] / 2
    hit.append(hit[-1] + dist[-1] / 2)
    dist = new
hit = np.array(hit)

out = pl.DataFrame({"games": np.arange(1, N + 1), "p_streak_ge_8": hit})
print(out.filter(pl.col("games").is_in([8, 20, 50, 100, 200, 300])))
out.write_csv("outputs/tables/streak_chance.csv")

fig, ax = plt.subplots(figsize=(6, 3.6))
ax.plot(out["games"], out["p_streak_ge_8"], color="#2a78d6", lw=2)
ax.axhline(1 / 256, color="#eb6834", lw=2, ls="--")
ax.text(N, 1 / 256 + 0.02, "1/256: one fixed run of 8", ha="right", va="bottom", color="0.25", fontsize=9)
ax.text(120, 0.55, "at least one run of ≥ 8\nwins or losses", ha="right", va="bottom", color="0.25", fontsize=9)
ax.set_xlabel("games played at a 50% win chance")
ax.set_ylabel("probability")
ax.set_ylim(0, 1)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("outputs/figures/streak_chance.svg")
