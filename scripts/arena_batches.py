# arena 配對批次的重建，與以批次為單位的 2014 年 arena 玩家局樣本。
# 批次：同一場錦標賽中開局時間相同（到秒）的對局視為同一次配對（依據見 docs/results/random-pairing.md）。

import numpy as np
import polars as pl
from scipy import stats

import novice
import pairing_null
from calibrated import probs, rd_baseline

UTC = "UTC"


def games(lo, hi):
    """[lo, hi) 之間的錦標賽對局，加上批次欄位；回傳 (g, batch, bad)。bad 為含伺服器異常批次的錦標賽，未排除。"""
    g = (
        pl.scan_parquet("data/processed/lichess_games.parquet")
        .filter(pl.col("tournament_id").is_not_null(), pl.col("start") >= lo, pl.col("start") < hi)
        .select("game_id", "start", "tournament_id", "white_elo", "black_elo", w="white_id", b="black_id")
        .collect()
        .sort("tournament_id", "start", "game_id")
    )
    batch = (
        g.group_by("tournament_id", "start", maintain_order=True)
        .agg(n_games=pl.len(), n_players=pl.concat_list("w", "b").list.explode().n_unique())
        .with_columns(
            first=pl.int_range(pl.len()).over("tournament_id") == 0,
            gap=pl.col("start").diff().over("tournament_id").dt.total_seconds(),
        )
        .with_columns(
            # 空閒人數為 2g 或 2g+1（奇數時剩一人不配）。g >= 7 必超過 12 人；g = 6 可能是 12 或 13 人，歸入 12 人以下並在分層表中單列
            rule=pl.when(pl.col("first") | (pl.col("n_games") >= 7)).then(pl.lit("random")).otherwise(pl.lit("smart")),
            size=pl.when(pl.col("n_games") >= 7).then(pl.lit("7+")).when(pl.col("n_games") >= 4).then(pl.lit("4-6")).otherwise(pl.col("n_games").cast(pl.String)),
        )
        .with_row_index("bid")
    )
    g = g.join(batch.select("tournament_id", "start", "bid", "rule", "size", "n_games"), on=["tournament_id", "start"]).sort("bid", "game_id")
    # 同一秒內同一玩家出現在兩局，不是任何配對規則產生得了的，視為伺服器異常
    bad = batch.filter(pl.col("n_players") < 2 * pl.col("n_games"))["tournament_id"].unique().implode()
    return g, batch, bad


def prev_games(g):
    """每位玩家在同一場錦標賽中上一局的 game_id（prev）。兩人的 prev 相同即為上一局剛對過。"""
    return (
        pl.concat([g.select("game_id", "start", "bid", "tournament_id", player="w"), g.select("game_id", "start", "bid", "tournament_id", player="b")])
        .sort("tournament_id", "player", "start", "game_id")
        .with_columns(prev=pl.col("game_id").shift().over("tournament_id", "player"))
    )


def _sample():
    # 從 2013-12-31 讀起，跨年的錦標賽才不會把 2014 年的第一批誤判為開賽批
    g, batch, bad = games(pl.datetime(2013, 12, 31, time_zone=UTC), pl.datetime(2015, 1, 1, time_zone=UTC))
    g = g.filter(~pl.col("tournament_id").is_in(bad))
    batch = batch.filter(~pl.col("tournament_id").is_in(bad))
    prev = prev_games(g).select("game_id", "player", "prev")
    pg = (
        pl.scan_parquet("data/processed/player_games.parquet")
        .filter(pl.col("arena"), pl.col("start") >= pl.datetime(2014, 1, 1, time_zone=UTC), pl.col("start") < pl.datetime(2015, 1, 1, time_zone=UTC))
        .select(
            "game_id", "player", "color", "start", "speed", "tournament_id", "elo", "opp_elo", "score", "result",
            "player_game_no", "prev_gap", "est_duration", "session_threshold", "continue_session", "leave_8h",
            "tour_left", "tour_first", "tour_last", "return_seconds", "return_censored", "to_outage",
        )
        .collect()
        .join(batch.select("tournament_id", "start", "bid", "rule", "size", "n_games", "first"), on=["tournament_id", "start"])
        .join(prev, on=["game_id", "player"])
        .sort("player", "start", "game_id")
        .with_columns(games_in_tour=pl.int_range(pl.len()).over("player", "tournament_id"))
    )
    pg = probs(pg, color=False).rename({"e_cal": "e"}).drop("p_win", "p_draw", "p_loss")
    to_outage = pl.col("to_outage").fill_null(10**12)
    nxt = pl.col("return_seconds")
    return pg.with_columns(
        q=(pl.col("e") - 0.5).abs(),
        white=(pl.col("color") == "white").cast(pl.Float64),
        win=(pl.col("result") == "win").cast(pl.Float64),
        draw=(pl.col("result") == "draw").cast(pl.Float64),
        elo_c=(pl.col("elo") - 1500) / 100,
        log_games_in_tour=pl.col("games_in_tour").log1p(),
        log_remaining=((pl.col("tour_last") - pl.col("start")).dt.total_seconds() / 60).log1p(),
        # 玩家層級的控制變數：自己的累計局數未滿門檻
        novice_games=(pl.col("player_game_no") < novice.GAMES).cast(pl.Float64),
        after_1130=pl.col("start") >= pl.datetime(2014, 11, 30, time_zone=UTC),
        early_half=(pl.col("start") - pl.col("tour_first")) < (pl.col("tour_last") - pl.col("tour_first")) / 2,
        # 時間窗內遇到全站停擺時，沒有下一局不是玩家的選擇，該反應變數設為缺值
        tour_left=pl.when(to_outage > (pl.col("tour_last") - pl.col("start")).dt.total_seconds()).then(pl.col("tour_left")),
        continue_session=pl.when(to_outage > pl.col("session_threshold")).then(pl.col("continue_session")),
        leave_8h=pl.when(to_outage > 8 * 3600).then(pl.col("leave_8h")),
        # 主門檻的一半與兩倍下的續下，2014 年的時間窗都在資料終點之前
        continue_half=pl.when(to_outage > pl.col("session_threshold") / 2).then(~pl.col("return_censored") & (nxt <= pl.col("session_threshold") / 2)),
        continue_double=pl.when(to_outage > pl.col("session_threshold") * 2).then(~pl.col("return_censored") & (nxt <= pl.col("session_threshold") * 2)),
    )


def player_sample():
    """2014 年 arena 的玩家局，含批次、上一局、匹配品質與解釋變數；已排除異常錦標賽。

    layer：multi 為兩局以上的批（設計上的隨機）；single 為開賽之後的一局一批（近似隨機）；
    first_single 為只有一局的開賽批（全場兩人，批內沒有可打亂的配法，兩層都不用）。
    e_bar、q_bar 為期望匹配品質：multi 取主虛無分布下的精確期望，single 取同場同 5 分鐘的候選對手平均。
    q 的期望依玩家在候選者中的等級分位置而定（等級分極端者期望 q 較大），而這個位置本身可能與離開有關，
    比較 q 的效果時須控制它（或以 q - q_bar 為處理變數），否則混入的是位置的效果。
    """
    d = _sample().with_columns(
        layer=pl.when(pl.col("n_games") >= 2).then(pl.lit("multi")).when(~pl.col("first")).then(pl.lit("single")).otherwise(pl.lit("first_single")),
        tour_win5=pl.concat_str("tournament_id", pl.col("start").dt.truncate("5m").cast(pl.String)),
    )
    m, ns = pairing_null.structure(d)
    e_m, q_m = pairing_null.expected(ns)
    e_w, q_w = pairing_null.expected_window(d, ns)
    d = d.with_columns(e_bar_w=e_w, q_bar_w=q_w)
    return (
        d.join(m.select("game_id", "player", e_bar_m=pl.Series(e_m), q_bar_m=pl.Series(q_m)), on=["game_id", "player"], how="left")
        .with_columns(
            e_bar=pl.when(pl.col("layer") == "multi").then("e_bar_m").otherwise("e_bar_w"),
            q_bar=pl.when(pl.col("layer") == "multi").then("q_bar_m").otherwise("q_bar_w"),
        )
        .drop("e_bar_m", "q_bar_m", "e_bar_w", "q_bar_w")
    )


def player_sample_rd():
    """player_sample 加上對手的帳號代碼、重放的雙方 RD 與主基準（calibrated.rd_baseline）。

    h = e_bar - e 為隨機配對產生的對手強弱，正值為對手比依配對機制所期望的強；resid 為得分減主基準的期望得分；
    score_per_elo 為主基準下期望得分對自身等級分的斜率，把得分尺度的效果換算成等級分。
    """
    ids = pl.scan_parquet("data/processed/lichess_games.parquet").select("game_id", "white_id", "black_id")
    rp = pl.scan_parquet("data/processed/glicko_replay.parquet").select("game_id", "white_rd", "black_rd")
    d = (
        player_sample()
        .join(ids.collect(), on="game_id")
        .join(rp.collect(), on="game_id")
        .with_columns(
            rd=pl.when(pl.col("color") == "white").then("white_rd").otherwise("black_rd").cast(pl.Float64),
            opp_rd=pl.when(pl.col("color") == "white").then("black_rd").otherwise("white_rd").cast(pl.Float64),
            opponent=pl.when(pl.col("color") == "white").then("black_id").otherwise("white_id"),
        )
        .drop("white_id", "black_id", "white_rd", "black_rd")
    )
    lo = 1 / (1 + (-(pl.col("eta") - pl.col("w"))).exp())
    hi = 1 / (1 + (-(pl.col("eta") + pl.col("w"))).exp())
    return rd_baseline(d).with_columns(
        h=pl.col("e_bar") - pl.col("e"),
        resid=pl.col("score") - pl.col("e_base"),
        score_per_elo=pl.col("slope_elo") * (lo * (1 - lo) + hi * (1 - hi)) / 2,
    )


STOP_RULES = ["first_loss", "two_losses"]


def stop_rule_sample(day_gap=12 * 3600):
    """2014 下半年的 arena 玩家局，加上一天的切分與兩條民間停止規則刪去的局；回傳 (d, del_days)。

    一天以全部對局（含大廳）的開局間隔切分：大廳對局不進序列，但玩家在下大廳就不算離開。
    del_<rule> 為該規則刪去的局（觸發的那一局本身保留）。del_days 每天一列，next_resid 為下一個含 arena 局的一天中
    第一局 arena 局的殘差（next_opponent 為該局的對手），先在全年的序列上找，再限於下半年開始的天。
    """
    days = (
        pl.scan_parquet("data/processed/player_games.parquet")
        .filter(pl.col("start") >= pl.datetime(2014, 6, 1, time_zone=UTC), pl.col("start") < pl.datetime(2015, 1, 1, time_zone=UTC))
        .select("game_id", "player", "start", "prev_gap")
        .collect()
        .sort("player", "start", "game_id")
        .with_columns(day=(pl.col("prev_gap").is_null() | (pl.col("prev_gap") > day_gap)).cum_sum().over("player"))
        .with_columns(day_start=pl.col("start").first().over("player", "day"))
        .select("game_id", "player", "day", "day_start")
    )
    d = (
        player_sample_rd()
        .select("game_id", "player", "opponent", "start", "speed", "result", "resid", "rd", "opp_rd")
        .join(days, on=["game_id", "player"])
        .sort("player", "start", "game_id")
    )
    pd_ = ["player", "day"]
    loss = (pl.col("result") == "loss").cast(pl.Int32)
    d = (
        d.with_columns(run=(pl.col("result") != "loss").cum_sum().over(pd_))
        .with_columns(streak=loss.cum_sum().over(pd_ + ["run"]))
        .with_columns(
            del_first_loss=(loss.cum_sum().over(pd_) - loss) > 0,
            del_two_losses=((pl.col("streak") >= 2).cast(pl.Int32).cum_sum().over(pd_) - (pl.col("streak") >= 2).cast(pl.Int32)) > 0,
            novice_rd=novice.novice_rd("rd", "opp_rd"),
        )
    )
    first = d.group_by(pd_, maintain_order=True).agg(pl.col("resid").first(), pl.col("speed").first(), pl.col("opponent").first())
    first = first.with_columns(
        next_resid=pl.col("resid").shift(-1).over("player"), next_speed=pl.col("speed").shift(-1).over("player"), next_opponent=pl.col("opponent").shift(-1).over("player")
    ).select(pd_ + ["next_resid", "next_speed", "next_opponent"])
    d = d.filter(pl.col("day_start") >= pl.datetime(2014, 7, 1, time_zone=UTC)).drop("run", "streak")
    del_days = d.group_by(pd_).agg(*[pl.col(f"del_{r}").any().alias(f"any_{r}") for r in STOP_RULES]).join(first, on=pd_, how="left")
    return d, del_days


def rating_filter_sample(nq=5):
    """有上一局的 arena 玩家局，加上一局的結果 prev_result、上一局的等級分變動 dr（每 100 分）、
    主基準期望得分的 logit L，與自身 RD 的 nq 等分 rdq（依速度類別各自取切點，0 起算）及其虛擬變數 q*、dr_q*。"""
    grp = ["player", "tournament_id"]
    t = (
        player_sample_rd()
        .sort("player", "tournament_id", "start", "game_id")
        .with_columns(
            prev_result=pl.col("result").shift(1).over(grp),
            dr=((pl.col("elo") - pl.col("elo").shift(1).over(grp)) / 100).cast(pl.Float64),
        )
        .filter(pl.col("prev_result").is_not_null())
        .with_columns(L=(pl.col("e_base") / (1 - pl.col("e_base"))).log(), const=pl.lit(1.0))
    )
    return pl.concat([
        s.with_columns(rdq=pl.Series(np.searchsorted(np.quantile(s["rd"].to_numpy(), np.linspace(0, 1, nq + 1)[1:-1]), s["rd"].to_numpy(), side="right")))
        for s in t.partition_by("speed")
    ]).with_columns(*[(pl.col("rdq") == q).cast(pl.Float64).alias(f"q{q}") for q in range(1, nq)], *[(pl.col("dr") * (pl.col("rdq") == q)).alias(f"dr_q{q}") for q in range(1, nq)])


def player_slopes(r, h, player, min_rows):
    """每位玩家殘差對 h 的 OLS 斜率，至少 min_rows 列的玩家，等權重 t 檢定。"""
    df = pl.DataFrame({"p": player, "r": r, "h": h}).group_by("p").agg(
        n=pl.len(), b=pl.cov("r", "h") / pl.col("h").var()
    ).filter(pl.col("n") >= min_rows, pl.col("b").is_finite())
    b = df["b"].to_numpy()
    tt = b.mean() / (b.std(ddof=1) / np.sqrt(len(b)))
    return {"players": len(b), "mean_slope": b.mean(), "se": b.std(ddof=1) / np.sqrt(len(b)), "t": tt, "p": 2 * stats.t.sf(abs(tt), len(b) - 1)}


def continuation_sample(lags):
    """兩層（multi、single）的 arena 玩家局，加上之後第 k 局的殘差 resid_next{k} 與續下 cont{k}。

    cont{k} 只在第 k 局之後的風險集內有定義：tour_left 有定義（錦標賽不被資料終點或停擺截斷），
    且本局之後錦標賽仍有 k 局名目長度的時間。
    """
    grp = ["player", "tournament_id"]
    d = (
        player_sample_rd()
        .filter(pl.col("layer").is_in(["multi", "single"]))
        .sort("player", "tournament_id", "start", "game_id")
        .with_columns(
            *[pl.col("resid").shift(-k).over(grp).alias(f"resid_next{k}") for k in lags],
            *[pl.col("score_per_elo").shift(-k).over(grp).alias(f"spe_next{k}") for k in lags],
            n_after=pl.len().over(grp) - 1 - pl.int_range(pl.len()).over(grp),
            novice_rd=novice.novice_rd("rd", "opp_rd").cast(pl.Float64),
            loss=(pl.col("result") == "loss").cast(pl.Float64),
        )
        # 跨年的錦標賽在 2015 年的局不在樣本中，續下會被誤記為離開
        .filter(pl.col("tour_last") < pl.datetime(2015, 1, 1, time_zone="UTC"))
        .with_columns(elo_c2=pl.col("elo_c") ** 2, h_win=pl.col("h") * pl.col("win"), h_draw=pl.col("h") * pl.col("draw"))
    )
    for k in lags:
        d = d.with_columns(
            pl.when(pl.col("tour_left").is_not_null() & ((pl.col("tour_last") - pl.col("start")).dt.total_seconds() >= k * pl.col("est_duration")))
            .then(pl.col("n_after") >= k)
            .alias(f"cont{k}")
        )
    # 第一局之後的續下是 tour_left 的反面
    assert d.filter(pl.col("cont1").is_not_null(), pl.col("cont1") == pl.col("tour_left")).height == 0
    return d
