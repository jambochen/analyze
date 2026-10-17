# 懸殊的配對是否下得較短。表頭沒有結束時間，以 API 的 lastMoveAt 減 createdAt 量對局長度；
# API 只查了每場錦標賽的第一局（data/raw/lichess-tournament-source/），即開賽批的一局，屬純隨機配對；
# 只用開賽批有兩局以上者（全場只有兩人時對手是確定的），並控制期望匹配品質 q_bar。
# 若懸殊對局較短，以開局間隔定義的續下與錦標賽中途離開，會機械性地隨 q 上升而下降：下完得早，下一局開得早，賽程也剩得多。

import sys

import matplotlib.pyplot as plt
import polars as pl
import pyfixest as pf

sys.path.insert(0, "scripts")
from arena_batches import player_sample

api = (
    pl.read_ndjson("data/raw/lichess-tournament-source/games.ndjson")
    .select(game_id="id", status="status", duration=(pl.col("lastMoveAt") - pl.col("createdAt")) / 1000)
)
d = (
    player_sample()
    .filter(pl.col("layer") == "multi")
    .join(api, on="game_id")
    .with_columns(
        elo_c2=pl.col("elo_c") ** 2,
        no_start=(pl.col("status") == "noStart").cast(pl.Float64),
        log_duration=(pl.col("duration") + 1).log(),
        duration_ratio=pl.col("duration") / pl.col("est_duration"),
        leave_session=(~pl.col("continue_session")).cast(pl.Float64),
        tour_left=pl.col("tour_left").cast(pl.Float64),
        leave_8h=pl.col("leave_8h").cast(pl.Float64),
        tc=pl.col("speed").cast(pl.Categorical).to_physical(),
        time_control=pl.col("est_duration"),
        pid=pl.col("player"),
    )
)
# 對局層級只取白方那一列；q 與 color=False 的 e 對雙方對稱
games = d.filter(pl.col("color") == "white")
print(games.height, games["status"].value_counts().sort("count"))

rows = []
for y, sub in [("log_duration", None), ("log_duration", "started"), ("no_start", None), ("duration_ratio", "started")]:
    df = games if sub is None else games.filter(pl.col("no_start") == 0)
    res = pf.feols(f"{y} ~ q + q_bar + elo_c + elo_c2 | time_control", df.select(y, "q", "q_bar", "elo_c", "elo_c2", "time_control").to_pandas(), vcov="hetero")
    t = res.tidy()
    rows.append(dict(response=y, sample=sub or "all", term="q", coef=t.loc["q", "Estimate"], se_hetero=t.loc["q", "Std. Error"], p=t.loc["q", "Pr(>|t|)"], n=res._N))
dur = pl.DataFrame(rows)
print(dur)
dur.write_csv("outputs/tables/match_quality_duration.csv")

binned = (
    games.with_columns(q_bin=(pl.col("q") * 10).floor().clip(0, 4) / 10)
    .group_by("speed", "q_bin")
    .agg(n=pl.len(), no_start=pl.col("no_start").mean(), median_duration_ratio=pl.col("duration_ratio").filter(pl.col("no_start") == 0).median())
    .sort("speed", "q_bin")
)
print(binned)
binned.write_csv("outputs/tables/match_quality_duration_binned.csv")

# 加入對局長度後 q 對離開的係數變化。對局長度在 q 之後、本身不是隨機指派的，這只是描述 q 的效果有多少與長度同行，不是中介效果的估計
med = []
for r in ["tour_left", "leave_session", "leave_8h"]:
    df = d.filter(pl.col(r).is_not_null()).select(r, "q", "q_bar", "elo_c", "elo_c2", "log_games_in_tour", "novice", "log_duration", "no_start", "win", "draw", "tc", "pid").to_pandas()
    for spec, extra in [("total", ""), ("with_duration", "+ log_duration + no_start"), ("with_result", "+ win + draw"), ("with_result_duration", "+ win + draw + log_duration + no_start")]:
        res = pf.feglm(f"{r} ~ q + q_bar + elo_c + elo_c2 + novice {extra} | tc", df, family="logit", vcov={"CRV1": "pid"})
        t = res.tidy()
        med.append(dict(response=r, spec=spec, coef_q=t.loc["q", "Estimate"], se_q=t.loc["q", "Std. Error"], p=t.loc["q", "Pr(>|t|)"], n=res._N))
med = pl.DataFrame(med)
print(med)
med.write_csv("outputs/tables/match_quality_duration_leave.csv")

fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
for tcn, c in [("Bullet", "C0"), ("Blitz", "C1")]:
    x = binned.filter(pl.col("speed") == tcn)
    axes[0].plot(x["q_bin"] + 0.05, x["median_duration_ratio"], "o-", color=c, label=tcn)
    axes[1].plot(x["q_bin"] + 0.05, x["no_start"], "o-", color=c, label=tcn)
axes[0].set_xlabel("match quality q = |e - 0.5|, binned")
axes[0].set_ylabel("median duration / nominal length")
axes[1].set_xlabel("match quality q = |e - 0.5|, binned")
axes[1].set_ylabel("share of games with no start")
axes[0].legend(fontsize=8)
fig.suptitle("First game of each 2014 arena tournament (API timestamps)", fontsize=10)
fig.tight_layout()
fig.savefig("outputs/figures/match_quality_duration.png", dpi=150)
