# 玩家近期表現提示：只用前局形成提醒，觀察續下者下一局的預測資訊。
import os
import time
os.environ["JAX_PLATFORMS"] = "cpu"
import numpy as np
import polars as pl

import arena_batches as ab
import likelihood as lk
import novice
import ordinal
from runs import runs_before
import temporal_models as tm

started = time.monotonic()
smoke = os.environ.get("PLAYER_ALERT_SMOKE") == "1"
coef = pl.read_csv("outputs/tables/temporal_prediction_coef.csv")
state = pl.read_csv("outputs/tables/temporal_prediction_state_coef.csv")
grp = ["player", "tournament_id"]
base = ab.player_sample_rd("2014").sort(*grp, "start", "game_id").with_columns(
    prev_result=pl.col("result").shift().over(grp),
    dr=((pl.col("elo") - pl.col("elo").shift().over(grp)) / 100).fill_null(0.),
    previous_start=pl.col("start").shift().over(grp),
    time=pl.col("start").dt.epoch("s"),
)
base = runs_before(base, grp)
opp = base.select("game_id", opponent="player", opp_prev_result="prev_result", opp_dr="dr")
base = base.join(opp, on=["game_id", "opponent"], how="left").with_columns(
    d_dr=pl.col("dr") - pl.col("opp_dr").fill_null(0.),
    d_prev_draw=(pl.col("prev_result") == "draw").fill_null(False).cast(pl.Float64) - (pl.col("opp_prev_result") == "draw").fill_null(False).cast(pl.Float64),
    d_prev_win=(pl.col("prev_result") == "win").fill_null(False).cast(pl.Float64) - (pl.col("opp_prev_result") == "win").fill_null(False).cast(pl.Float64),
    opp_missing=pl.col("opp_prev_result").is_null().cast(pl.Float64),
    sign=2 * pl.col("white") - 1,
    xr=((pl.col("elo") + pl.col("opp_elo")) / 2 - 1500) / 100,
    xd=(pl.col("elo") - pl.col("opp_elo")) / 100,
    gxd=ordinal.g("rd", "opp_rd") * (pl.col("elo") - pl.col("opp_elo")) / 100,
).sort(*grp, "start", "game_id")
print("loaded", base.height, time.monotonic() - started, flush=True)
rows, overlap, thresholds, checks, monthly = [], [], [], [], []
folds = [("Jul-Sep", "2014-07-01", "2014-10-01"), ("Oct-Dec", "2014-10-01", "2015-01-01")]
for speed in (["Bullet"] if smoke else ["Bullet", "Blitz"]):
    for fold, start, end in (folds[:1] if smoke else folds):
        lo, hi = [int(np.datetime64(x, "s").astype(np.int64)) for x in [start, end]]
        th = coef.filter(pl.col("game") == "Lichess", pl.col("speed") == speed, pl.col("fold") == fold, pl.col("fit") == "base_refit").sort("param_index")["est"].to_numpy()
        d = tm.offset(base.filter(pl.col("speed") == speed, pl.col("time") < hi), th, "Lichess")
        p = ordinal.probs(d["eta"].to_numpy(), d["width"].to_numpy())
        e = p[:, 2] + .5 * p[:, 1]
        variance = p[:, 2] + .25 * p[:, 1] - e**2
        eta = d["eta"].to_numpy().copy()
        bc = state.filter(pl.col("game") == "Lichess", pl.col("speed") == speed, pl.col("fold") == fold, pl.col("model") == "recent")
        for name, value in bc.select("param", "est").iter_rows():
            if name != "baseline_offset":
                eta += d[name].to_numpy() * value
        pr = ordinal.probs(eta, d["width"].to_numpy())
        scored = d.with_columns(e_base=pl.Series(e), e_recent=pl.Series(pr[:, 2] + .5 * pr[:, 1]), variance=pl.Series(variance), past_error=pl.col("score") - pl.Series(e))
        d = scored.with_columns(
            error5=pl.col("past_error").shift().rolling_sum(5).over(grp),
            variance5=pl.col("variance").shift().rolling_sum(5).over(grp),
            expectation5=pl.col("e_base").shift().rolling_mean(5).over(grp),
        ).filter(pl.col("error5").is_not_null()).with_columns(z5=pl.col("error5") / pl.col("variance5").sqrt())
        assert (d["previous_start"] < d["start"]).all()
        train = d.filter(pl.col("time") < lo)
        rate = (train["loss_run"] >= 3).mean()
        cutoff = float(np.quantile(train["z5"].to_numpy(), rate))
        thresholds.append(dict(speed=speed, fold=fold, train_rows=train.height, train_max_time=train["time"].max(), test_start=lo, loss3_rate=rate, z_cutoff=cutoff, matched_train_rate=(train["z5"] <= cutoff).mean()))
        te = d.filter(pl.col("time") >= lo).with_columns(
            loss3=pl.col("loss_run") >= 3, loss4=pl.col("loss_run") >= 4,
            adjusted_fixed=pl.col("z5") <= -1.645, adjusted_matched=pl.col("z5") <= cutoff,
        )
        assert train["time"].max() < lo <= te["time"].min() and te["time"].max() < hi
        # 抽查 rolling 特徵確為當局之前五局；當局得分不能影響自己的提醒。
        probe = d.select(grp).unique(maintain_order=True).head(12)
        max_history_error = 0.
        for seq in scored.join(probe, on=grp, how="semi").partition_by(grp):
            expected_error = np.array([seq["past_error"].to_numpy()[j-5:j].sum() for j in range(5, seq.height)])
            expected_variance = np.array([seq["variance"].to_numpy()[j-5:j].sum() for j in range(5, seq.height)])
            actual = d.join(seq.select("game_id", "player"), on=["game_id", "player"], how="semi").sort("start", "game_id")
            error = max(np.max(abs(actual["error5"].to_numpy() - expected_error)), np.max(abs(actual["variance5"].to_numpy() - expected_variance)))
            max_history_error = max(max_history_error, error)
        assert max_history_error < 1e-10
        checks.append(dict(speed=speed, fold=fold, eligible_rows=te.height, test_min_time=te["time"].min(), test_max_time=te["time"].max(), max_history_error=max_history_error))
        for group, selected in [("all", pl.lit(True)), ("high_rd", pl.col("rd") >= novice.RD), ("established", pl.col("rd") < novice.RD)]:
            g = te.filter(selected)
            a, b = g["player"].to_numpy(), g["opponent"].to_numpy()
            y = g["score"].to_numpy()
            _, player_index = np.unique(a, return_inverse=True)
            for rule in ["loss3", "loss4", "adjusted_fixed", "adjusted_matched"]:
                alert = g[rule].to_numpy()
                assert alert.any() and (~alert).any()
                burden = g.group_by("player").agg(rate=pl.col(rule).mean(), alerts=pl.col(rule).sum())
                ca = np.bincount(player_index, weights=alert)[player_index]
                cn = np.bincount(player_index, weights=~alert)[player_index]
                wa = np.divide(alert.astype(float), ca, out=np.zeros(len(alert)), where=ca > 0)
                wn = np.divide((~alert).astype(float), cn, out=np.zeros(len(alert)), where=cn > 0)
                for model in ["base", "recent"]:
                    residual = y - g["e_" + model].to_numpy()
                    ma, mn = residual[alert].mean(), residual[~alert].mean()
                    phi_a = np.where(alert, residual - ma, 0) / alert.sum()
                    phi_n = np.where(~alert, residual - mn, 0) / (~alert).sum()
                    se = lk.dyad_se((phi_a - phi_n)[:, None], a, b)
                    sea = lk.dyad_se(phi_a[:, None], a, b)
                    ea, en = (wa * residual).sum() / wa.sum(), (wn * residual).sum() / wn.sum()
                    equal_se = lk.dyad_se((wa * (residual - ea) / wa.sum())[:, None], a, b)
                    equal_diff_se = lk.dyad_se((wa * (residual - ea) / wa.sum() - wn * (residual - en) / wn.sum())[:, None], a, b)
                    rows.append(dict(speed=speed, fold=fold, group=group, rule=rule, model=model, n=g.height, alerted=int(alert.sum()), alert_rate=alert.mean(), players=burden.height,
                                     largest_player_alert_share=burden["alerts"].max() / alert.sum(), top10_alert_share=burden["alerts"].sort(descending=True).head(10).sum() / alert.sum(),
                                     equal_player_residual_alert=ea, equal_player_alert_lo=ea - 1.96 * equal_se, equal_player_alert_hi=ea + 1.96 * equal_se,
                                     equal_player_diff=ea - en, equal_player_diff_lo=ea - en - 1.96 * equal_diff_se, equal_player_diff_hi=ea - en + 1.96 * equal_diff_se,
                                     player_any_rate=(burden["rate"] > 0).mean(), player_rate_median=burden["rate"].median(), player_rate_p90=burden["rate"].quantile(.9),
                                     score_after_alert=y[alert].mean(), loss_after_alert=(y[alert] == 0).mean(), expected_after_alert=g["e_" + model].to_numpy()[alert].mean(), past_expected=g["expectation5"].to_numpy()[alert].mean(),
                                     residual_alert=ma, residual_alert_lo=ma - 1.96 * sea, residual_alert_hi=ma + 1.96 * sea, residual_nonalert=mn, diff=ma - mn, lo=ma - mn - 1.96 * se, hi=ma - mn + 1.96 * se))
        for (month,), md in te.with_columns(month=pl.col("start").dt.month()).partition_by("month", as_dict=True).items():
            for rule in ["loss3", "loss4", "adjusted_fixed", "adjusted_matched"]:
                alerted = md.filter(pl.col(rule))
                monthly.append(dict(speed=speed, fold=fold, month=month, rule=rule, n=md.height, alerted=alerted.height, residual=(alerted["score"] - alerted["e_recent"]).mean()))
        for raw, adjusted, label in [(True, True, "both"), (True, False, "loss_only"), (False, True, "adjusted_only"), (False, False, "neither")]:
            g = te.filter(pl.col("loss3") == raw, pl.col("adjusted_matched") == adjusted)
            residual = (g["score"] - g["e_recent"]).to_numpy()
            mean = residual.mean()
            se = lk.dyad_se(((residual - mean) / len(residual))[:, None], g["player"].to_numpy(), g["opponent"].to_numpy())
            overlap.append(dict(speed=speed, fold=fold, category=label, n=g.height, share=g.height / te.height, past_expected=g["expectation5"].mean(), score=g["score"].mean(), expected=g["e_recent"].mean(), residual=mean, lo=mean - 1.96 * se, hi=mean + 1.96 * se))
        print(speed, fold, te.height, "seconds", time.monotonic() - started, flush=True)

out = "outputs/tables/player_alerts_smoke" if smoke else "outputs/tables/player_alerts"
pl.DataFrame(rows).write_csv(out + "_evaluation.csv")
pl.DataFrame(overlap).write_csv(out + "_overlap.csv")
pl.DataFrame(thresholds).write_csv(out + "_thresholds.csv")
pl.DataFrame(checks).write_csv(out + "_checks.csv")
pl.DataFrame(monthly).write_csv(out + "_monthly.csv")
print("seconds", time.monotonic() - started, flush=True)
