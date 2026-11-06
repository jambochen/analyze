# 期末報告（期望勝率作為判斷玩家狀態的基準）投影片用的統計圖與衍生表。
# 只讀 outputs/tables 的結果，不重估任何模型。

import matplotlib.pyplot as plt
import polars as pl

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTE, GRID = "#22262b", "#6b7178", "#e6e6e3"
CLASSES = ["Bullet", "Blitz", "Classical"]
COLOR = dict(zip(CLASSES, [BLUE, ORANGE, AQUA]))
MARK = dict(zip(CLASSES, ["o", "s", "^"]))
T = lambda f: pl.read_csv(f"outputs/tables/{f}.csv")
plt.rcParams["svg.fonttype"] = "path"


def style(ax, axis="x"):
    ax.grid(axis=axis, color=GRID, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)


# 修正量：老手對局，雙方上一局結果不同的組合與對手本場第一局；結果相同的組合恰為 0，不畫
shift = T("state_adjusted_shift").filter(~pl.col("novice_rd"))
combos = [("win", "loss"), ("draw", "loss"), ("win", "draw"), ("win", "first"), ("draw", "first"), ("loss", "first")]
labels = [f"own {a}, opponent {b}" if b != "first" else f"own {a}, opponent's first game" for a, b in combos] + ["rating change difference +100"]
fig, ax = plt.subplots(figsize=(6.4, 3.9))
for j, tc in enumerate(CLASSES):
    s = shift.filter(pl.col("speed") == tc)
    for i, (a, b) in enumerate(combos + [(None, None)]):
        r = s.filter(pl.col("contrast") == "dr_per_100") if a is None else s.filter(pl.col("own_prev") == a, pl.col("opp_prev") == b)
        if r.height == 0:
            continue
        r = r.row(0, named=True)
        y = i + (j - 1) * 0.22
        ax.plot([r["elo"] - 1.96 * r["elo_se"], r["elo"] + 1.96 * r["elo_se"]], [y, y], color=COLOR[tc], lw=1.5)
        ax.plot(r["elo"], y, MARK[tc], color=COLOR[tc], ms=5, label=tc if i == 0 else None)
ax.axvline(0, color=INK, lw=0.8, ls="--")
ax.set_yticks(range(len(labels)), labels)
ax.invert_yaxis()
ax.set_xlabel("adjustment to own rating (rating points)")
style(ax)
ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
fig.tight_layout()
fig.savefig("outputs/figures/baseline_slides_shift.svg")

# 續下的選擇：Blitz 與 Bullet 第一局、兩局以上的批、以玩家加權；只用續下者、倒數機率加權、修剪上下界
b = T("continuation_selection_bounds").filter(pl.col("layer") == "multi", pl.col("lag") == 1, pl.col("weighting") == "player", pl.col("monotone"))
methods = [("continuers", "continuers only"), ("ipw", "inverse-probability weighting"), ("lee", "trimming bounds")]
fig, ax = plt.subplots(figsize=(5.6, 2.8))
for j, tc in enumerate(CLASSES[:2]):
    for i, (m, lab) in enumerate(methods):
        r = b.filter(pl.col("speed") == tc, pl.col("method") == m).row(0, named=True)
        y = i + (j - 0.5) * 0.3
        ax.plot([r["ci_lo_elo"], r["ci_hi_elo"]], [y, y], color=COLOR[tc], lw=1.5, label=tc if i == 0 else None)
        if m == "lee":
            ax.plot([r["lo_elo"], r["hi_elo"]], [y, y], "|", color=COLOR[tc], ms=9)
        else:
            ax.plot(r["lo_elo"], y, MARK[tc], color=COLOR[tc], ms=5)
ax.axvline(0, color=INK, lw=0.8, ls="--")
ax.set_yticks(range(len(methods)), [lab for _, lab in methods])
ax.invert_yaxis()
ax.set_xlabel("strongest minus weakest fifth (rating points)")
style(ax)
ax.legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig("outputs/figures/baseline_slides_bounds.svg")

# 連勝連敗：前四局全負或全勝之後的平均殘差，三個基準
st = T("streaks")
BASE = [("e_sys", "system (Elo)", MUTE, "o"), ("e_base", "main baseline", INK, "s"), ("e_adj", "state-adjusted", ORANGE, "^")]
for name, rows in [("lichess", [("lichess", "Bullet"), ("lichess", "Blitz")]), ("aoe2", [("aoe2", "1v1")])]:
    fig, axes = plt.subplots(1, len(rows), figsize=(3.4 * len(rows) + 0.8, 3.2), sharey=True, squeeze=False)
    for ax, (g, sp) in zip(axes[0], rows):
        for i, after in enumerate(["4_losses", "4_wins"]):
            for k, (bs, lab, col, mk) in enumerate(BASE):
                r = st.filter(pl.col("game") == g, pl.col("speed") == sp, pl.col("after") == after, pl.col("baseline") == bs).row(0, named=True)
                x = i + (k - 1) * 0.18
                ax.plot([x, x], [r["lo"], r["hi"]], color=col, lw=1.5)
                ax.plot(x, r["mean_resid"], mk, color=col, ms=5, label=lab if i == 0 else None)
        ax.axhline(0, color=INK, lw=0.8, ls="--")
        ax.set_xticks([0, 1], ["after 4 losses", "after 4 wins"])
        ax.set_title(sp if g == "lichess" else "AoE2 1v1", fontsize=10)
        style(ax, "y")
    axes[0][0].set_ylabel("mean score minus expected")
    axes[0][-1].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"outputs/figures/baseline_slides_streaks_{name}.svg")

# 相依資料下的檢定水準：當獨立的 LRT 與修正的 LRT 在模擬中的拒絕率，名目 5%
size = T("adjusted_lrt_size").filter(pl.col("test") == "one_df").select("n_player", "pareto_shape", "effective_players", "naive_0.05", "adjusted_0.05")
print(size)
size.write_csv("outputs/tables/baseline_slides_lrt_size.csv")

# 配對批次的大小：各批局數的分布，開賽第一批與之後的批分開，以局數加權
bs = T("arena_random_pairing_batch_size").filter(pl.col("period") == "2014").group_by("first", "n_games").agg(pl.col("games").sum())
bs = bs.with_columns(k=pl.col("n_games").clip(upper_bound=6)).group_by("first", "k").agg(pl.col("games").sum()).sort("first", "k")
tot = bs["games"].sum()
fig, ax = plt.subplots(figsize=(5.2, 3.0))
for j, (first, lab, col) in enumerate([(False, "later batches", INK), (True, "first batch of the arena", MUTE)]):
    z = bs.filter(pl.col("first") == first)
    ax.bar(z["k"] + (j - 0.5) * 0.38, z["games"] / tot, width=0.36, color=col, label=lab)
ax.set_xticks(range(1, 7), ["1", "2", "3", "4", "5", "6+"])
ax.set_xlabel("games in the batch")
ax.set_ylabel("share of 2014 arena games")
style(ax, "y")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig("outputs/figures/baseline_slides_batch_size.svg")

# 劑量反應：上一局 h 的五等分，下一局的平均殘差；以玩家加權
dose = T("arena_crossover_dose")
fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9), sharey=True)
for ax, eff, title in zip(axes, ["total", "direct"], ["total effect", "direct effect"]):
    for j, tc in enumerate(CLASSES[:2]):
        z = dose.filter(pl.col("speed") == tc, pl.col("effect") == eff).sort("q")
        x = z["q"] + (j - 0.5) * 0.15
        ax.errorbar(x, z["resid_mean"], yerr=1.96 * z["resid_se"], fmt=MARK[tc] + "-", color=COLOR[tc], ms=5, lw=1.2, capsize=0, label=tc)
    ax.axhline(0, color=INK, lw=0.8, ls="--")
    ax.set_xticks(range(1, 6))
    ax.set_xlabel("quintile of h in previous game")
    ax.set_title(title, fontsize=10)
    style(ax, "y")
axes[0].set_ylabel("mean residual, next game")
axes[0].legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig("outputs/figures/baseline_slides_dose.svg")

# 續下率：ē 格內的 h 五等分，兩局以上的批、第一局、以玩家加權
rates = T("continuation_selection_rates").filter(pl.col("layer") == "multi", pl.col("lag") == 1, pl.col("weighting") == "player")
fig, ax = plt.subplots(figsize=(4.6, 3.0))
for j, tc in enumerate(CLASSES[:2]):
    z = rates.filter(pl.col("speed") == tc).sort("arm")
    ax.errorbar(z["arm"] + (j - 0.5) * 0.12, z["rate"], yerr=1.96 * z["se"], fmt=MARK[tc] + "-", color=COLOR[tc], ms=5, lw=1.2, capsize=0, label=tc)
ax.set_xticks(range(1, 6), ["1\nweakest", "2", "3", "4", "5\nstrongest"])
ax.set_xlabel("opponent strength relative to expectation (quintile of h)")
ax.set_ylabel("share playing another game")
style(ax, "y")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig("outputs/figures/baseline_slides_continue_rate.svg")

# 修剪上下界依之後第幾局
bl = T("continuation_selection_bounds").filter(pl.col("layer") == "multi", pl.col("weighting") == "player", pl.col("monotone"), pl.col("method") == "lee")
fig, ax = plt.subplots(figsize=(4.6, 3.0))
for j, tc in enumerate(CLASSES[:2]):
    z = bl.filter(pl.col("speed") == tc).sort("lag")
    for r in z.iter_rows(named=True):
        x = r["lag"] + (j - 0.5) * 0.2
        ax.plot([x, x], [r["ci_lo_elo"], r["ci_hi_elo"]], color=COLOR[tc], lw=1.5, label=tc if r["lag"] == 1 else None)
        ax.plot([x, x], [r["lo_elo"], r["hi_elo"]], color=COLOR[tc], lw=4, alpha=0.5)
ax.axhline(0, color=INK, lw=0.8, ls="--")
ax.set_xticks([1, 2, 3], ["1st", "2nd", "3rd"])
ax.set_xlabel("next game")
ax.set_ylabel("trimming bounds (rating points)")
style(ax, "y")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig("outputs/figures/baseline_slides_bounds_lag.svg")

# 剛贏、剛輸之後的校準截距，依 RD 五等分
cells = T("rating_filter_error_cells")
RES = [("loss", "previous loss", "v"), ("draw", "previous draw", "o"), ("win", "previous win", "^")]
GRAY = {"loss": INK, "draw": MUTE, "win": ORANGE}
fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.9), sharey=True)
for ax, tc in zip(axes, CLASSES[:2]):
    for k, (res, lab, mk) in enumerate(RES):
        z = cells.filter(pl.col("speed") == tc, pl.col("prev_result") == res).sort("rdq")
        ax.errorbar(z["rdq"] + (k - 1) * 0.12, z["alpha"], yerr=1.96 * z["alpha_se"], fmt=mk, color=GRAY[res], ms=5, capsize=0, lw=1.2, label=lab)
    ax.axhline(0, color=INK, lw=0.8, ls="--")
    ax.set_xticks(range(1, 6), ["1\nlowest", "2", "3", "4", "5\nhighest"])
    ax.set_xlabel("quintile of pre-game RD")
    ax.set_title(tc, fontsize=10)
    style(ax, "y")
axes[0].set_ylabel("calibration intercept")
axes[0].legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig("outputs/figures/baseline_slides_after_result.svg")

# 修正前後的校準：依上一局結果，主基準與依狀態調整的勝率
cv = T("state_adjusted_probs_curve")
fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.4), sharex=True, sharey=True)
for i, (m, lab) in enumerate([("baseline", "main baseline"), ("opp_sym", "state-adjusted")]):
    for j, tc in enumerate(CLASSES[:2]):
        ax = axes[i, j]
        for k, (res, rl, mk) in enumerate(RES):
            z = cv.filter(pl.col("speed") == tc, pl.col("model") == m, pl.col("prev_result") == res).sort("ebin")
            ax.errorbar(z["pred_mean"], z["resid_mean"], yerr=1.96 * z["resid_se"], fmt=mk + "-", color=GRAY[res], ms=4, lw=1, capsize=0, label=rl)
        ax.axhline(0, color=INK, lw=0.8, ls="--")
        ax.set_title(f"{tc}, {lab}", fontsize=9)
        style(ax, "y")
for ax in axes[1]:
    ax.set_xlabel("predicted expected score")
for ax in axes[:, 0]:
    ax.set_ylabel("score minus predicted")
axes[0, 0].legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig("outputs/figures/baseline_slides_adjusted_curve.svg")

# 連串之後弱方的比例：Elo 低估弱方，弱方占多數的連敗就顯得好於期望
fig, ax = plt.subplots(figsize=(4.6, 2.9))
groups = [("lichess", "Bullet", "Bullet"), ("lichess", "Blitz", "Blitz"), ("aoe2", "1v1", "AoE2")]
for i, (g, sp, lab) in enumerate(groups):
    for k, (after, al, col) in enumerate([("4_losses", "after 4 losses", INK), ("4_wins", "after 4 wins", MUTE)]):
        r = st.filter(pl.col("game") == g, pl.col("speed") == sp, pl.col("after") == after, pl.col("baseline") == "e_sys").row(0, named=True)
        ax.bar(i + (k - 0.5) * 0.38, r["share_underdog_sys"], width=0.36, color=col, label=al if i == 0 else None)
ax.axhline(0.5, color=INK, lw=0.8, ls="--")
ax.set_xticks(range(3), [lab for *_, lab in groups])
ax.set_ylim(0, 1)
ax.set_ylabel("share of games as underdog")
style(ax, "y")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig("outputs/figures/baseline_slides_underdog.svg")
