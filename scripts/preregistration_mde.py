# 前瞻驗證的最小可偵測效果（docs/preregistration.md）。
# 2015 年樣本只數 arena 對局的局數，不讀結果欄；標準誤依 2014 年各分析的標準誤，按局數比的平方根縮放，
# 即假設 2015 年的玩家活躍度集中程度與 2014 年相同。檢定力 0.8。

import sys

sys.path.insert(0, "scripts")

import numpy as np
import polars as pl
from scipy.stats import norm

from arena_batches import player_sample_rd

UTC = "UTC"
POWER = norm.ppf(0.8)
Z1, Z2 = norm.ppf(0.95), norm.ppf(0.975)

g = pl.scan_parquet("data/processed/lichess_games.parquet").filter(pl.col("tournament_system") == "arena", pl.col("speed").is_in(["Bullet", "Blitz"]))
dt = lambda y, m, d: pl.datetime(y, m, d, time_zone=UTC)  # noqa: E731
windows = {"2014": (dt(2014, 1, 1), dt(2015, 1, 1)), "2014h2": (dt(2014, 7, 1), dt(2015, 1, 1)), "2015": (dt(2015, 1, 1), dt(2015, 4, 20))}
n_games = {
    (w, s): v
    for w, (a, b) in windows.items()
    for s, v in g.filter(pl.col("start") >= a, pl.col("start") < b).group_by("speed").agg(pl.len()).collect().iter_rows()
}

spe = dict(player_sample_rd().group_by("speed").agg(pl.col("score_per_elo").mean()).iter_rows())
rows = []

# 1. Blitz 剛被分到強對手之後的總效果：單尾 0.05。換算成等級分用 2014 年的 elo_per_0_1 與估計值之比
ri = pl.read_csv("outputs/tables/arena_crossover_ri.csv").filter(
    pl.col("null") == "main", pl.col("stat") == "beta_lag1", pl.col("weighting") == "player", pl.col("batches") == "all", pl.col("speed") == "Blitz"
).row(0, named=True)
scale = np.sqrt(n_games["2014", "Blitz"] / n_games["2015", "Blitz"])
se = ri["null_sd"] * scale
elo_per_beta = abs(ri["elo_per_0_1"] / ri["est_centered"])
rows.append({"quantity": "blitz_total_effect", "speed": "Blitz", "rule": "", "unit": "elo_per_0.1_h", "se_2014": ri["null_sd"] * elo_per_beta, "scale": scale, "mde": (Z1 + POWER) * se * elo_per_beta})

# 2. 依狀態調整的勝率的對數分數差：雙尾 0.05，單位是每局的 log loss
pr = pl.read_csv("outputs/tables/state_adjusted_probs_prediction.csv").filter(
    pl.col("metric") == "log_loss", pl.col("group") == "all", pl.col("model") == "opp_sym", pl.col("vs") == "baseline", pl.col("speed").is_in(["Bullet", "Blitz"])
)
for r in pr.iter_rows(named=True):
    scale = np.sqrt(n_games["2014h2", r["speed"]] / n_games["2015", r["speed"]])
    rows.append({"quantity": "log_loss_difference", "speed": r["speed"], "rule": "", "unit": "log_loss_per_game", "se_2014": r["diff_se_dyadic"], "scale": scale, "mde": (Z2 + POWER) * r["diff_se_dyadic"] * scale})

# 3. 停止規則回測的刪去減保留之差：雙尾 0.05，以等級分表示
bt = pl.read_csv("outputs/tables/folk_stop_rules_backtest.csv").filter(pl.col("group") == "all", pl.col("speed").is_in(["Bullet", "Blitz"]))
for r in bt.iter_rows(named=True):
    scale = np.sqrt(n_games["2014h2", r["speed"]] / n_games["2015", r["speed"]])
    rows.append({"quantity": "stop_rule_difference", "speed": r["speed"], "rule": r["rule"], "unit": "elo", "se_2014": r["se"] / spe[r["speed"]], "scale": scale, "mde": (Z2 + POWER) * r["se"] / spe[r["speed"]] * scale})

out = pl.DataFrame(rows)
print(out)
out.write_csv("outputs/tables/preregistration_mde.csv")
pl.DataFrame([{"window": w, "speed": s, "arena_games": v} for (w, s), v in n_games.items()]).write_csv("outputs/tables/preregistration_games.csv")
