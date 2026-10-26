# 新手期的斷點：Elo 形式期望得分的校準斜率在 RD 110 處是否跳躍。
# 規格見 docs/methods.md「新手期的界定」的「斷點的檢定」。推論把對局當成獨立。

import sys

sys.path.insert(0, "scripts")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import stats

import likelihood as lk
import novice
import ordinal

UTC = "UTC"
CLASSES = ["Bullet", "Blitz", "Classical"]
N_FIT = 1_000_000
J = novice.RD / 100

d = (
    pl.scan_parquet("data/processed/lichess_games.parquet")
    .filter(
        pl.col("speed").is_in(CLASSES), pl.col("base_seconds").is_not_null(),
        pl.col("start") >= pl.datetime(2014, 1, 1, time_zone=UTC), pl.col("start") < pl.datetime(2015, 1, 1, time_zone=UTC),
        pl.col("white_elo").is_not_null(), pl.col("black_elo").is_not_null(), pl.col("result").is_in(["1-0", "0-1", "1/2-1/2"]),
    )
    .select("game_id", "speed", score=pl.col("result").replace_strict({"0-1": 0.0, "1/2-1/2": 0.5, "1-0": 1.0}), L=(pl.col("white_elo") - pl.col("black_elo")) * ordinal.Q)
    .join(pl.scan_parquet("data/processed/glicko_replay.parquet").select("game_id", "perf", "white_rd", "black_rd", "white_left_truncated", "black_left_truncated"), on="game_id")
    .filter(pl.col("perf") == pl.col("speed"), ~pl.col("white_left_truncated"), ~pl.col("black_left_truncated"), pl.col("white_rd").is_not_null(), pl.col("black_rd").is_not_null())
    .select("speed", "score", "L", x=pl.max_horizontal("white_rd", "black_rd").cast(pl.Float64) / 100)
    # 首局的 RD 恰為 350，所有新帳號都從這一點起算
    .filter(pl.col("x") < 3.5)
    .collect(engine="streaming")
)
print(d.group_by("speed").agg(n=pl.len(), x_med=pl.col("x").median(), share_novice=(pl.col("x") >= J).mean()))


# restricted cubic spline（Harrell 的寫法）：x 本身加 k-2 個在兩端之外為線性的三次項
def rcs(x, t):
    k = len(t)
    cols = [x]
    for j in range(k - 2):
        c = np.maximum(x - t[j], 0) ** 3 - np.maximum(x - t[k - 2], 0) ** 3 * (t[k - 1] - t[j]) / (t[k - 1] - t[k - 2]) + np.maximum(x - t[k - 1], 0) ** 3 * (t[k - 2] - t[j]) / (t[k - 1] - t[k - 2])
        cols.append(c / (t[k - 1] - t[0]) ** 2)
    return np.stack(cols, 1)


test_rows, curve_rows = [], []
for tc in CLASSES:
    m = d.filter(pl.col("speed") == tc)
    m = m.sample(min(N_FIT, m.height), seed=1)
    x, L, s = m["x"].to_numpy(), m["L"].to_numpy(), m["score"].to_numpy()
    lo, hi = np.quantile(x, [0.01, 0.99])
    one = np.ones_like(x)
    fits = {}
    for k in [5, 7]:
        t = np.linspace(lo, hi, k)
        B = np.column_stack([one, rcs(x, t)])
        X = np.column_stack([B, B * L[:, None]])
        th, p, H, ll = lk.frac_newton(X, s, [0] * B.shape[1] + [1] + [0] * (B.shape[1] - 1))
        Xj = np.column_stack([X, (x >= J) * L])
        thj, _, Hj, llj = lk.frac_newton(Xj, s, list(th) + [0])
        fits[k] = (t, X.shape[1], ll)
        lr = 2 * (llj - ll)
        test_rows.append({"speed": tc, "test": "jump_at_110", "knots": k, "n": m.height, "lr": lr, "df": 1, "p_naive": stats.chi2.sf(lr, 1), "jump": thj[-1], "jump_se_iid": np.sqrt(np.linalg.inv(Hj)[-1, -1])})
        # b(x) 的曲線與 iid 區間，取不含跳躍的樣條
        xs = np.linspace(lo, hi, 200)
        Bs = np.column_stack([np.ones_like(xs), rcs(xs, t)])
        nb = B.shape[1]
        V = np.linalg.inv(H)[nb:, nb:]
        b = Bs @ th[nb:]
        se = np.sqrt(np.einsum("ij,jk,ik->i", Bs, V, Bs))
        for xv, bv, sv in zip(xs, b, se):
            curve_rows.append({"speed": tc, "knots": k, "rd": xv * 100, "slope": bv, "lo": bv - 1.96 * sv, "hi": bv + 1.96 * sv})
    lr = 2 * (fits[7][2] - fits[5][2])
    df = fits[7][1] - fits[5][1]
    test_rows.append({"speed": tc, "test": "spline_7_vs_5", "knots": 7, "n": m.height, "lr": lr, "df": df, "p_naive": stats.chi2.sf(lr, df)})
    print(pl.DataFrame(test_rows[-3:], strict=False))

tests = pl.DataFrame(test_rows, strict=False)
# Holm：每個 (檢定, 節點數) 以三個速度類別為一族
tests = pl.concat([
    g.sort("p_naive").with_columns(p_holm=(pl.col("p_naive") * (3 - pl.int_range(pl.len()))).cum_max().clip(upper_bound=1))
    for _, g in tests.group_by("test", "knots", maintain_order=True)
], how="diagonal")
tests.write_csv("outputs/tables/novice_breakpoint_tests.csv")
curve = pl.DataFrame(curve_rows)
curve.write_csv("outputs/tables/novice_breakpoint_curve.csv")
print(tests)

cox = pl.read_csv("outputs/tables/novice_period_cox.csv").filter(pl.col("stratum_type") == "rd_max", pl.col("rd_max_median") < 350)
INK, MUTE = "#2b2b2b", "#8a8a85"
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for ax, tc in zip(axes, CLASSES):
    cv = curve.filter(pl.col("speed") == tc, pl.col("knots") == 5)
    ax.fill_between(cv["rd"], cv["lo"], cv["hi"], color="#2a78d6", alpha=0.2, lw=0)
    ax.plot(cv["rd"], cv["slope"], color="#2a78d6", lw=2, label="spline, 5 knots")
    c = cox.filter(pl.col("speed") == tc)
    ax.plot(c["rd_max_median"], c["slope"], "o", ms=5, color=MUTE, label="RD-bin estimates")
    ax.axvline(novice.RD, color=MUTE, lw=0.8, ls=":")
    ax.set_title(tc, color=INK)
    ax.set_xlabel("max pre-game RD of the two players")
    ax.grid(axis="y", color="#e6e6e3", lw=0.6)
axes[0].set_ylabel("calibration slope of Elo E")
axes[0].legend(fontsize=8, frameon=False)
fig.suptitle("Calibration slope vs RD, first games excluded (games treated as independent)")
fig.tight_layout()
fig.savefig("outputs/figures/novice_breakpoint.png", dpi=150)
