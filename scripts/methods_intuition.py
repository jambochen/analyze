# 方法簡報的直觀圖：全以已知參數模擬，不用 Lichess 資料，圖內標 simulated。

import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams["font.size"] = 12
rng = np.random.default_rng(20141)
q = np.log(10) / 400


# 校準圖：真實斜率 0.8，系統仍以斜率 1 給期望得分
d = rng.normal(0, 250, 200_000)
p_sys = 1 / (1 + np.exp(-q * d))
y = rng.random(d.size) < 1 / (1 + np.exp(-0.8 * q * d))
bins = np.quantile(p_sys, np.linspace(0, 1, 11))
k = np.clip(np.searchsorted(bins, p_sys, side="right") - 1, 0, 9)
mp = np.array([p_sys[k == i].mean() for i in range(10)])
my = np.array([y[k == i].mean() for i in range(10)])
fig, ax = plt.subplots(figsize=(4.6, 4.2))
ax.plot([0, 1], [0, 1], color="0.6", lw=1, ls="--")
ax.plot(mp, my, "o-", color=BLUE, lw=1.5, ms=5)
ax.text(0.97, 0.80, "overrates the stronger side", ha="right", va="center", color="0.25")
ax.text(0.03, 0.20, "underrates the weaker side", ha="left", va="center", color="0.25")
ax.set_xlabel("mean expected score (system)")
ax.set_ylabel("mean actual score")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title("simulated, true slope b = 0.8", loc="left")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("outputs/figures/methods_intuition_calibration.svg")

# 區間涵蓋：玩家局數右偏，每位玩家有自己的偏離，局與局在玩家內相關
m, sims = 300, 20
naive, player = [], []
for _ in range(sims):
    n = np.ceil(rng.pareto(1.2, m) * 3 + 1).astype(int)
    u = rng.normal(0, 0.15, m)
    pid = np.repeat(np.arange(m), n)
    e = u[pid] + rng.normal(0, 0.45, n.sum())
    N, mean = e.size, e.mean()
    se_naive = e.std(ddof=1) / np.sqrt(N)
    tot = np.bincount(pid, e) - n * mean
    se_player = np.sqrt(m / (m - 1) * (tot**2).sum()) / N
    naive.append((mean, se_naive))
    player.append((mean, se_player))
fig, axes = plt.subplots(2, 1, figsize=(6, 4.2), sharex=True)
for ax, rows, lab in zip(axes, (naive, player), ("games treated as independent", "summarized by player")):
    for i, (mu, se) in enumerate(rows):
        lo, hi = mu - 1.96 * se, mu + 1.96 * se
        miss = lo > 0 or hi < 0
        ax.plot([lo, hi], [i, i], color=ORANGE if miss else BLUE, lw=1.6)
    ax.axvline(0, color="0.3", lw=1)
    ax.set_title(lab, loc="left")
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
axes[1].set_xlabel("95% interval for the mean residual (simulated, true value 0)")
fig.tight_layout()
fig.savefig("outputs/figures/methods_intuition_coverage.svg")


# 累積殘差的路徑：每局對手的期望得分隨機，結果只分勝負
def walk(T, stop_after=None):
    p = rng.uniform(0.3, 0.7, T)
    w = rng.random(T) < p
    e = w - p
    if stop_after is None:
        return np.cumsum(e), T
    lost = 0
    for t in range(T):
        lost = 0 if w[t] else lost + 1
        if lost == stop_after:
            return np.cumsum(e[: t + 1]), t + 1
    return np.cumsum(e), T


T = 20
fig, ax = plt.subplots(figsize=(6, 3.6))
for _ in range(20):
    path, _ = walk(T)
    ax.plot(np.arange(1, T + 1), path, color=BLUE, lw=1, alpha=0.6)
ax.axhline(0, color="0.3", lw=1)
ax.set_xlabel("game")
ax.set_ylabel("cumulative residual $M_t$")
ax.set_title("simulated, every expected score correct", loc="left")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("outputs/figures/methods_intuition_paths.svg")

fixed = np.array([walk(T)[0][-1] for _ in range(20_000)])
ruled = np.array([walk(T, 2)[0][-1] for _ in range(20_000)])
print("fixed mean", fixed.mean(), "rule mean", ruled.mean())
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), gridspec_kw={"width_ratios": (1.2, 1)})
ax = axes[0]
for _ in range(20):
    path, t = walk(T, 2)
    ax.plot(np.arange(1, t + 1), path, color=BLUE, lw=1, alpha=0.6)
    if t < T:
        ax.plot(t, path[-1], "o", color=ORANGE, ms=4)
ax.axhline(0, color="0.3", lw=1)
ax.set_xlabel("game")
ax.set_ylabel("cumulative residual $M_t$")
ax.set_title("stop after two straight losses (dots)", loc="left")
ax.spines[["top", "right"]].set_visible(False)
ax = axes[1]
edges = np.linspace(-6, 6, 49)
ax.hist(fixed, edges, density=True, histtype="step", color=BLUE, lw=1.6, label=f"play 20 games, mean {fixed.mean():+.2f}")
ax.hist(ruled, edges, density=True, histtype="step", color=ORANGE, lw=1.6, label=f"stopping rule, mean {ruled.mean():+.2f}")
ax.axvline(0, color="0.3", lw=1)
ax.set_xlabel("final $M$ (simulated)")
ax.set_yticks([])
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=1)
ax.spines[["top", "right", "left"]].set_visible(False)
fig.tight_layout()
fig.savefig("outputs/figures/methods_intuition_stopping.svg")

# 以下為方法介紹頁的通用例子，與本題資料無關

# logistic 迴歸：讀書時數與是否通過
hours = rng.uniform(0, 10, 120)
passed = rng.random(hours.size) < 1 / (1 + np.exp(-(hours - 5) * 1.1))
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.plot(hours, passed + rng.uniform(-0.03, 0.03, hours.size), "o", color=BLUE, ms=4, alpha=0.6)
xs = np.linspace(0, 10, 200)
ax.plot(xs, 1 / (1 + np.exp(-(xs - 5) * 1.1)), color=ORANGE, lw=2)
ax.set_xlabel("hours studied")
ax.set_ylabel("passed (1) or not (0)")
ax.set_yticks([0, 0.5, 1])
ax.set_title("simulated: the curve is P(pass | hours)", loc="left")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("outputs/figures/methods_intuition_logistic_example.svg")

# 分層列聯表：腎結石的兩種療法（Charig et al., 1986）。分層看 A 都較好，合併後反而 B 較好
rates = {"small stones": (81 / 87, 234 / 270), "large stones": (192 / 263, 55 / 80), "pooled": (273 / 350, 289 / 350)}
fig, ax = plt.subplots(figsize=(6, 3.8))
for i, (k, (a, b)) in enumerate(rates.items()):
    ax.bar(i - 0.18, a, 0.34, color=BLUE, label="treatment A" if i == 0 else None)
    ax.bar(i + 0.18, b, 0.34, color=ORANGE, label="treatment B" if i == 0 else None)
ax.set_xticks(range(3), list(rates))
ax.set_ylim(0.5, 1)
ax.set_ylabel("success rate")
ax.legend(frameon=False, ncols=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("outputs/figures/methods_intuition_simpson_example.svg")

# 彙總到受試者：每人量多次，兩種條件
m = 8
base = rng.normal(70, 10, m)
fig, ax = plt.subplots(figsize=(6, 3.8))
for i in range(m):
    a = base[i] + rng.normal(0, 2, 4)
    b = base[i] + 3 + rng.normal(0, 2, 4)
    ax.plot(np.zeros(4) + rng.uniform(-0.05, 0.05, 4), a, "o", color=BLUE, ms=3, alpha=0.6)
    ax.plot(np.ones(4) + rng.uniform(-0.05, 0.05, 4), b, "o", color=ORANGE, ms=3, alpha=0.6)
    ax.plot([0, 1], [a.mean(), b.mean()], color="0.4", lw=1)
ax.set_xticks([0, 1], ["condition 1", "condition 2"])
ax.set_xlim(-0.4, 1.4)
ax.set_ylabel("score")
ax.set_title("simulated: 8 people, 4 measurements each", loc="left")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("outputs/figures/methods_intuition_repeated_example.svg")

# 公平賭局的停止策略：贏到 3 元或下滿 30 次就停
final = []
fig, axes = plt.subplots(2, 1, figsize=(6, 5.2))
for k in range(20_000):
    steps = rng.choice([-1, 1], 30)
    path = np.cumsum(steps)
    hit = np.flatnonzero(path >= 3)
    t = hit[0] + 1 if hit.size else 30
    final.append(path[t - 1])
    if k < 15:
        axes[0].plot(np.arange(1, t + 1), path[:t], color=BLUE, lw=1, alpha=0.6)
        if hit.size:
            axes[0].plot(t, 3, "o", color=ORANGE, ms=4)
final = np.array(final)
axes[0].axhline(0, color="0.3", lw=1)
axes[0].set_xlabel("bet")
axes[0].set_ylabel("winnings")
axes[0].set_title("simulated fair coin: quit at +3 (dots)", loc="left")
axes[0].spines[["top", "right"]].set_visible(False)
vals, cnt = np.unique(final, return_counts=True)
axes[1].bar(vals, cnt / cnt.sum(), color=BLUE, width=0.8)
axes[1].set_xlabel(f"final winnings, mean {final.mean():+.2f}")
axes[1].set_yticks([])
axes[1].spines[["top", "right", "left"]].set_visible(False)
fig.tight_layout()
fig.savefig("outputs/figures/methods_intuition_gambler_example.svg")
print("gambler mean", final.mean())
