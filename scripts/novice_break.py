# 新手門檻（RD 110）處的斷點：以事件時間對齊，檢定 Elo 期望得分的校準截距與斜率在越過門檻時是否跳躍。
# 規格見 docs/methods.md「新手期的界定」的「斷點的檢定」。邊際模型：新手的偏差即帳號實力與等級分之差，
# 玩家隨機效果會把它吸收掉。

import os
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, "scripts")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from statsmodels.stats.multitest import multipletests

import game_source
import likelihood as lk
import novice
import ordinal

UTC = "UTC"
SPEEDS = ["Bullet", "Blitz", "Classical"]
K_LO, K_HI = -5, 4
plt.rcParams["svg.fonttype"] = "path"

rp = pl.scan_parquet("data/processed/glicko_replay.parquet")
side = pl.concat([
    rp.select("game_id", color=pl.lit(c), rd=f"{c}_rd", lt=f"{c}_left_truncated", opp_rd=f"{o}_rd", opp_lt=f"{o}_left_truncated")
    for c, o in [("white", "black"), ("black", "white")]
])
pg = (
    pl.scan_parquet("data/processed/player_games.parquet")
    .filter(pl.col("speed").is_in(SPEEDS), pl.col("start") < pl.datetime(2015, 1, 1, time_zone=UTC))
    .select("game_id", "player", "opponent", "color", "start", "speed", "source", "elo", "opp_elo", "score")
    .join(side, on=["game_id", "color"])
    .sort("player", "speed", "start", "game_id")
    .collect(engine="streaming")
)
# 事件時間在玩家該速度類別的全部 rated 對局上計算：RD 隨每一局更新，不論來源
pg = pg.with_columns(
    idx=pl.int_range(pl.len()).over("player", "speed"),
    first_elo=pl.col("elo").first().over("player", "speed"),
    first_lt=pl.col("lt").first().over("player", "speed"),
)
cross = pg.filter(pl.col("rd") < novice.RD).group_by("player", "speed").agg(cross_idx=pl.col("idx").min())
t = (
    pg.join(cross, on=["player", "speed"])
    .with_columns(k=pl.col("idx") - pl.col("cross_idx"))
    .filter(
        pl.col("first_elo") == 1500, ~pl.col("first_lt"),
        pl.col("k").is_between(K_LO, K_HI),
        pl.col("opp_rd") < novice.RD, ~pl.col("opp_lt"),
        pl.col("start") >= pl.datetime(2014, 1, 1, time_zone=UTC),
        pl.col("elo").is_not_null(), pl.col("opp_elo").is_not_null(), pl.col("score").is_not_null(),
    )
)
t = t.filter(game_source.KEEP)
t = t.with_columns(
    L=(pl.col("elo") - pl.col("opp_elo")).cast(pl.Float64) * ordinal.Q,
    J=(pl.col("k") >= 0).cast(pl.Float64),
    kf=pl.col("k").cast(pl.Float64),
).with_columns(kL=pl.col("kf") * pl.col("L"), JL=pl.col("J") * pl.col("L"))

sample = t.group_by("speed", "k").agg(n=pl.len(), players=pl.col("player").n_unique(), rd_mean=pl.col("rd").mean()).sort("speed", "k")
with pl.Config(tbl_rows=-1):
    print(sample)
sample.write_csv("outputs/tables/novice_break_sample.csv")

XCOLS = ["const", "L", "kf", "kL", "J", "JL"]
free_of = lambda cols: np.array([XCOLS.index(c) for c in cols], dtype=int)
T0 = np.zeros(len(XCOLS))
KS = list(range(K_LO, K_HI + 1))
test_rows, coef_rows, curve_rows = [], [], []
for spd in SPEEDS:
    s = t.filter(pl.col("speed") == spd).with_columns(const=pl.lit(1.0))
    w, b = s["player"].to_numpy(), s["opponent"].to_numpy()
    y = s["score"].to_numpy().astype(np.float64)
    a = (s.select(XCOLS).to_numpy().astype(np.float64), y)
    th, V, r = lk.lrt_player_dyad(lk.logit_ll, T0, free_of(XCOLS[:4]), free_of(XCOLS), a, w, b)
    test_rows.append({"speed": spd, "n": s.height, "players": s["player"].n_unique(), **r})
    for j, c in enumerate(XCOLS):
        coef_rows.append({"speed": spd, "param": c, "est": th[j], "se_dyadic": np.sqrt(V[j, j])})
    # 診斷：每個 k 各估截距與斜率
    D = np.concatenate([(s["k"] == k).to_numpy().astype(float)[:, None] * np.c_[np.ones(s.height), s["L"].to_numpy()] for k in KS], 1)
    th2, _, V2, *_ = lk.fit(lk.logit_ll, np.zeros(D.shape[1]), np.arange(D.shape[1]), (D, y), w, b)
    for i, k in enumerate(KS):
        curve_rows.append({"speed": spd, "k": k, "intercept": th2[2 * i], "intercept_se": np.sqrt(V2[2 * i, 2 * i]), "slope": th2[2 * i + 1], "slope_se": np.sqrt(V2[2 * i + 1, 2 * i + 1])})

tests = pl.DataFrame(test_rows)
tests = tests.with_columns(p_holm=pl.Series(multipletests(tests["p"].to_numpy(), method="holm")[1]))
coef = pl.DataFrame(coef_rows)
curve = pl.DataFrame(curve_rows)
with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250):
    print(tests)
    print(coef)
    print(curve)
tests.write_csv("outputs/tables/novice_break_tests.csv")
coef.write_csv("outputs/tables/novice_break_coef.csv")
curve.write_csv("outputs/tables/novice_break_curve.csv")

COLOR = dict(zip(SPEEDS, ["#2a78d6", "#eb6834", "#1baf7a"]))
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
for ax, (c, lab) in zip(axes, [("intercept", "calibration intercept"), ("slope", "calibration slope")]):
    for j, spd in enumerate(SPEEDS):
        z = curve.filter(pl.col("speed") == spd)
        ax.errorbar(z["k"] + (j - 1) * 0.12, z[c], yerr=1.96 * z[c + "_se"], fmt="o-", color=COLOR[spd], ms=4, lw=1, capsize=0, label=spd)
    ax.axvline(-0.5, color="#6b7178", lw=0.8, ls="--")
    ax.set_xlabel("games relative to first game with RD < 110")
    ax.set_ylabel(lab)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig("outputs/figures/novice_break_curve.svg")
