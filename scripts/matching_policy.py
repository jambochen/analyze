# 固定已配對玩家、開局時間與執棋顏色，探索替換對手的單步空間；不是完整隊列回放。
import os
import time

os.environ["JAX_PLATFORMS"] = "cpu"
import numpy as np
import polars as pl
from scipy.optimize import linear_sum_assignment

import arena_batches as ab
import matching_objectives as mo
import novice
from matching_probabilities import measures

started = time.monotonic()
smoke = os.environ.get("MATCHING_SMOKE") == "1"
coef = pl.read_csv("outputs/tables/temporal_prediction_coef.csv")
state = pl.read_csv("outputs/tables/temporal_prediction_state_coef.csv")
base = ab.player_sample_rd("2014").sort("player", "tournament_id", "start", "game_id").with_columns(
    dr=((pl.col("elo") - pl.col("elo").shift().over("player", "tournament_id")) / 100).fill_null(0.),
    last_result=pl.col("result").shift().over("player", "tournament_id"),
).filter(pl.col("start") >= ab.D(2014, 7, 1)).with_columns(
    fold=pl.when(pl.col("start") < ab.D(2014, 10, 1)).then(pl.lit("Jul-Sep")).otherwise(pl.lit("Oct-Dec")),
    last_draw=(pl.col("last_result") == "draw").fill_null(False).cast(pl.Float64),
    last_win=(pl.col("last_result") == "win").fill_null(False).cast(pl.Float64),
    missing=pl.col("last_result").is_null().cast(pl.Float64),
)
# 每個批次必須完整，不能先篩掉沒有近況或高 RD 的玩家。
assert base.group_by("bid").agg(n=pl.len(), expected=pl.col("n_games").first() * 2).filter(pl.col("n") != pl.col("expected")).is_empty()
print("loaded", base.height, "seconds", time.monotonic() - started, flush=True)




names = ["elo_gap", "base_imbalance", "recent_imbalance", "gap_ge200", "gap_ge400", "base_outside25_75", "recent_outside25_75", "uncertain_pair", "both_history_missing"]
policies = ["observed", "rating", "baseline", "recent", "tail200", "tail400", "minimax", "cap0", "cap50", "cap100", "rating_guard"]
extra_names = ["batch_max_gap", "worse_gap", "better_gap", "positive_gap_change", "worse_base", "worse_recent"]
player_names = ["elo_gap", "base_imbalance", "recent_imbalance", "gap_change", "worse_gap", "better_gap", "positive_gap_change", "worse_base", "worse_recent"]
player_rows = []
rows = []
coverage = []
checks = []
features = ["elo", "rd", "last_draw", "last_win", "missing", "dr"]
for (speed, fold), part in base.partition_by("speed", "fold", as_dict=True).items():
    th = coef.filter(pl.col("game") == "Lichess", pl.col("speed") == speed, pl.col("fold") == fold, pl.col("fit") == "base_refit").sort("param_index")["est"].to_numpy()
    bc = state.filter(pl.col("game") == "Lichess", pl.col("speed") == speed, pl.col("fold") == fold, pl.col("model") == "recent")
    bd = dict(bc.select("param", "est").iter_rows())
    assert set(bd) <= {"baseline_offset", "d_prev_draw", "d_prev_win", "opp_missing", "d_dr"}
    beta = np.array([bd.get(c, 0.) for c in ["d_prev_draw", "d_prev_win", "opp_missing", "d_dr"]])
    if smoke:
        ids = pl.concat([part.filter(pl.col("n_games") > 1)["bid"].unique().sort().head(30),
                         part.filter(pl.col("n_games") == 1)["bid"].unique().sort().head(1000)])
        part = part.filter(pl.col("bid").is_in(ids.implode()))
    totals = {}
    player_totals = {}
    counts = {}
    diagnostics = {"max_objective_violation": 0.}
    singles = part.filter(pl.col("n_games") == 1)
    w = singles.filter(pl.col("white") == 1).sort("game_id")
    b = singles.filter(pl.col("white") == 0).sort("game_id")
    value = measures(w.select(features).to_numpy(), b.select(features).to_numpy(), th, beta)
    for policy in policies:
        totals[("single", policy)] = np.r_[len(value), 0, value.sum(axis=0), value[:, 0].sum(), np.zeros(5)]
        pv = np.column_stack([np.tile(value[:, :3], (2, 1)), np.zeros((2 * len(value), 6))])
        own_rd = np.r_[w["rd"].to_numpy(), b["rd"].to_numpy()]
        for group, keep in [("all", np.ones(len(pv), dtype=bool)), ("high_rd", own_rd >= novice.RD), ("established", own_rd < novice.RD)]:
            player_totals[("single", policy, group)] = np.r_[keep.sum(), pv[keep].sum(axis=0)]
    counts["single"] = [w.height, w.height]
    mult = part.filter(pl.col("n_games") > 1).partition_by("bid")
    for batch in mult:
        w = batch.filter(pl.col("white") == 1).sort("game_id")
        b = batch.filter(pl.col("white") == 0).sort("game_id")
        n = w.height
        assert n == b.height and w["game_id"].equals(b["game_id"])
        wa, ba = w.select(features).to_numpy(), b.select(features).to_numpy()
        v = measures(np.repeat(wa, n, axis=0), np.tile(ba, (n, 1)), th, beta).reshape(n, n, -1)
        mask = np.ones((n, n), dtype=bool)
        wp, bp = w["prev"].fill_null("").to_numpy(), b["prev"].fill_null("").to_numpy()
        repeat = (wp[:, None] == bp[None, :]) & (wp[:, None] != "")
        rule, first = batch["rule"][0], batch["first"][0]
        category = "first" if first else "later_random" if rule == "random" else "later_smart"
        if not first and n == 6:
            category = "ambiguous_rule"
        elif rule == "smart" and np.any(np.diag(repeat)):
            category = "observed_rematch"
        elif rule == "smart":
            mask &= ~repeat
        frozen = category in ["ambiguous_rule", "observed_rematch"]
        ix = np.arange(n)
        # 最大化換對手數，只用來確認有沒有合法替代配法。
        alt = ix if frozen else linear_sum_assignment(np.where(mask, np.eye(n), np.inf))[1]
        if not frozen and np.array_equal(alt, ix):
            category = "no_alternative"
        counts.setdefault(category, [0, 0])
        counts[category][0] += 1
        counts[category][1] += n
        choices = {policy: ix for policy in policies}
        if not frozen and category != "no_alternative":
            choices.update(mo.assignments(v[:, :, 0], mask))
            for policy, ci in [("baseline", 1), ("recent", 2)]:
                choices[policy] = linear_sum_assignment(np.where(mask, v[:, :, ci], np.inf))[1]
        original = v[ix, ix]
        for policy in policies:
            chosen = choices[policy]
            assert len(np.unique(chosen)) == n and mask[ix, chosen].all()
            selected = v[ix, chosen]
            # 白方按列、黑方按被選中的欄與各自原對手比較。
            change = np.vstack([selected[:, :3] - original[:, :3], selected[:, :3] - original[chosen, :3]])
            if policy in ["rating", "baseline", "recent"]:
                ci = ["rating", "baseline", "recent"].index(policy)
                violation = selected[:, ci].sum() - original[:, ci].sum()
                diagnostics["max_objective_violation"] = max(diagnostics["max_objective_violation"], violation)
                assert violation < 1e-8
            elif policy == "rating_guard":
                assert selected[:, 0].sum() == v[ix, choices["rating"], 0].sum()
                assert (change[:, 0] > 0).sum() <= ((v[ix, choices["rating"], 0] > original[:, 0]).sum() + (v[ix, choices["rating"], 0] > original[choices["rating"], 0]).sum())
            elif policy.startswith("cap"):
                assert change[:, 0].max() <= int(policy[3:])
                assert selected[:, 0].sum() <= original[:, 0].sum()
            elif policy.startswith("tail"):
                threshold = int(policy[4:])
                assert (selected[:, 0] >= threshold).sum() <= (original[:, 0] >= threshold).sum()
            elif policy == "minimax":
                assert selected[:, 0].max() <= original[:, 0].max()
            extra = [selected[:, 0].max(), (change[:, 0] > 0).mean(), (change[:, 0] < 0).mean(), np.maximum(change[:, 0], 0).mean(), (change[:, 1] > 1e-12).mean(), (change[:, 2] > 1e-12).mean()]
            vals = np.r_[n, np.sum(chosen != ix), selected.sum(axis=0), np.array(extra) * n]
            totals.setdefault((category, policy), np.zeros(len(vals)))
            totals[(category, policy)] += vals
            pv = np.column_stack([np.tile(selected[:, :3], (2, 1)), change[:, 0], change[:, 0] > 0, change[:, 0] < 0, np.maximum(change[:, 0], 0), change[:, 1] > 1e-12, change[:, 2] > 1e-12])
            own_rd = np.r_[wa[:, 1], ba[chosen, 1]]
            for group, keep in [("all", np.ones(2 * n, dtype=bool)), ("high_rd", own_rd >= novice.RD), ("established", own_rd < novice.RD)]:
                player_totals.setdefault((category, policy, group), np.zeros(1 + len(player_names)))
                player_totals[(category, policy, group)] += np.r_[keep.sum(), pv[keep].sum(axis=0)]
    for (category, policy), vals in totals.items():
        rows.append(dict(speed=speed, fold=fold, category=category, policy=policy, games=int(vals[0]), changed_games=int(vals[1]), **dict(zip(names + extra_names, vals[2:] / vals[0]))))
    for (category, policy, group), vals in player_totals.items():
        if vals[0] > 0:
            player_rows.append(dict(speed=speed, fold=fold, category=category, policy=policy, group=group, player_games=int(vals[0]), **dict(zip(player_names, vals[1:] / vals[0]))))
    for category, (nb, ng) in counts.items():
        coverage.append(dict(speed=speed, fold=fold, category=category, batches=nb, games=ng))
    checks.append(dict(speed=speed, fold=fold, games=part.height // 2, **diagnostics))
    assert sum(v[1] for v in counts.values()) == part.height // 2
    print(speed, fold, "multi_batches", len(mult), "elapsed", time.monotonic() - started, flush=True)

out = "outputs/tables/matching_policy_smoke" if smoke else "outputs/tables/matching_policy"
r = pl.DataFrame(rows)
r.write_csv(out + "_strata.csv")
pl.DataFrame(player_rows).write_csv(out + "_players.csv")
summary = r.group_by("speed", "fold", "policy").agg(
    pl.col("games").sum(), pl.col("changed_games").sum(),
    *[(pl.col(c) * pl.col("games")).sum().truediv(pl.col("games").sum()).alias(c) for c in names + extra_names],
).sort("speed", "fold", "policy")
summary.write_csv(out + "_summary.csv")
pl.DataFrame(coverage).write_csv(out + "_coverage.csv")
pl.DataFrame(checks).write_csv(out + "_checks.csv")
print(summary, flush=True)
print("seconds", time.monotonic() - started, flush=True)
pooled = []
for scope, d in [("all", r), ("modifiable", r.filter(pl.col("category").is_in(["first", "later_smart", "later_random"])) )]:
    pooled.append(d.group_by("policy").agg(
        pl.col("games").sum(), pl.col("changed_games").sum(),
        *[(pl.col(c) * pl.col("games")).sum().truediv(pl.col("games").sum()).alias(c) for c in names + extra_names],
    ).with_columns(scope=pl.lit(scope)))
pl.concat(pooled).write_csv(out + "_pooled.csv")
