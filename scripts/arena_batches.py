# arena 配對批次的重建，與以批次為單位的 2014 年 arena 玩家局樣本。
# 批次：同一場錦標賽中開局時間相同（到秒）的對局視為同一次配對（依據見 docs/random-pairing.md）。

import numpy as np
import polars as pl

from calibrated import probs, rd_baseline

UTC = "UTC"
TCS = ["Blitz", "Bullet", "Classical"]


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
        novice=(pl.col("player_game_no") < 30).cast(pl.Float64),
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


def e_grid(elos):
    """E[速度類別, 自身, 對手]：color=False 的校準期望得分，供重新配對時查表。elos 須已排序。"""
    grid = (
        pl.DataFrame({"elo": elos})
        .join(pl.DataFrame({"opp_elo": elos}), how="cross")
        .join(pl.DataFrame({"speed": TCS}), how="cross")
        .with_columns(color=pl.lit("white"))
    )
    grid = probs(grid, color=False).sort("speed", "elo", "opp_elo")
    return grid["e_cal"].to_numpy().reshape(len(TCS), len(elos), len(elos))


def _matchings(items):
    if not items:
        return [[]]
    a, rest = items[0], items[1:]
    return [[(a, b)] + mt for i, b in enumerate(rest) for mt in _matchings(rest[:i] + rest[i + 1 :])]


def multi_null(m, E, elos):
    """兩局以上的批依 lila 規則的虛無配對結構。m 須依 bid、game_id、color 排序，每局兩列相鄰。

    純隨機的批（開賽批或 7 局以上）：批內均勻隨機兩兩配對。
    12 人以下的批：在不含「上一局剛對過」的配法中均勻抽出；實際配法本身含重配的批，表示規則當時沒有作用，改用不加限制的配法。
    v_pre 另把 2014-11-30 以前的 12 人以下批全部改為不加限制：此前一局一批的重配不符原始碼，規則是否完整作用存疑。
    """
    bi = np.unique(m["bid"].to_numpy(), return_inverse=True)[1]
    rule = m["rule"].to_numpy()
    ng = m["n_games"].to_numpy()
    after = m["after_1130"].to_numpy()
    prev_code = m.select(pl.col("prev").cast(pl.Categorical).to_physical().cast(pl.Int64).fill_null(-1)).to_series().to_numpy()
    starts = np.r_[0, np.flatnonzero(np.diff(bi)) + 1]
    smart = {}
    for g in range(2, 7):
        bs = starts[(rule[starts] == "smart") & (ng[starts] == g)]
        if len(bs) == 0:
            continue
        mt = np.array(_matchings(list(range(2 * g))))
        slots = bs[:, None] + np.arange(2 * g)
        pc = prev_code[slots]
        pa, pb = pc[:, mt[:, :, 0]], pc[:, mt[:, :, 1]]
        valid = ~((pa == pb) & (pa >= 0)).any(axis=2)
        actual_rematch = ~valid[:, 0]
        v_main = valid | actual_rematch[:, None]
        v_pre = v_main | ~after[bs][:, None]
        smart[g] = (slots, mt, v_main, v_pre)
    return dict(
        bi=bi, nb=bi.max() + 1, starts=starts, smart=smart, rand_rows=np.flatnonzero(rule == "random"),
        ui=np.searchsorted(elos, m["elo"].to_numpy()), tci=np.searchsorted(TCS, m["speed"].to_numpy()), E=E,
    )


def repair(ns, null, rng):
    """依 multi_null 的虛無配對結構重抽一次批內配法，回傳每個玩家位置的新對手位置。

    null 為 "main" 或 "unrestricted_before_1130"（12 人以下批在 2014-11-30 以前不加「不重配」的限制）。
    """
    rand_rows = ns["rand_rows"]
    opp = np.empty(len(ns["bi"]), int)
    o = rand_rows[np.argsort(ns["bi"][rand_rows] + rng.random(len(rand_rows)))]
    opp[o[0::2]] = o[1::2]
    opp[o[1::2]] = o[0::2]
    for slots, mt, v_main, v_pre in ns["smart"].values():
        v = v_main if null == "main" else v_pre
        r = np.floor(rng.random(len(slots)) * v.sum(1))
        pick = np.argmax(v.cumsum(1) > r[:, None], axis=1)
        pairs = mt[pick]
        a = np.take_along_axis(slots, pairs[:, :, 0], 1).ravel()
        b = np.take_along_axis(slots, pairs[:, :, 1], 1).ravel()
        opp[a] = b
        opp[b] = a
    return opp


def expected_multi(ns):
    """主虛無分布下每個玩家位置的期望 e 與期望 q（精確值，不靠模擬）。"""
    E, ui, tci, bi = ns["E"], ns["ui"], ns["tci"], ns["bi"]
    e_bar = np.zeros(len(bi))
    q_bar = np.zeros(len(bi))
    rr = ns["rand_rows"]
    rb = bi[rr]
    st = np.r_[0, np.flatnonzero(np.diff(rb)) + 1, len(rr)]
    for a, b in zip(st[:-1], st[1:]):
        sl = rr[a:b]
        eb = E[tci[sl][:, None], ui[sl][:, None], ui[sl][None, :]]
        np.fill_diagonal(eb, np.nan)
        e_bar[sl] = np.nanmean(eb, axis=1)
        q_bar[sl] = np.nanmean(np.abs(eb - 0.5), axis=1)
    for slots, mt, v_main, _ in ns["smart"].values():
        k = slots.shape[1]
        partner = np.empty((len(mt), k), int)
        for j in range(mt.shape[1]):
            partner[np.arange(len(mt)), mt[:, j, 0]] = mt[:, j, 1]
            partner[np.arange(len(mt)), mt[:, j, 1]] = mt[:, j, 0]
        w = v_main / v_main.sum(1, keepdims=True)
        for a in range(k):
            sa = slots[:, a]
            sp = np.take_along_axis(slots, np.broadcast_to(partner[:, a], (len(slots), len(mt))), 1)
            ev = E[tci[sa][:, None], ui[sa][:, None], ui[sp]]
            e_bar[sa] = (w * ev).sum(1)
            q_bar[sa] = (w * np.abs(ev - 0.5)).sum(1)
    return e_bar, q_bar


def expected_window(d, E, elos, chunk=100_000):
    """一局一批的近似：同一場錦標賽同一 5 分鐘內開局的其他玩家為候選對手，取對他們的平均 e 與 q。"""
    w = d["tour_win5"].cast(pl.Categorical).to_physical().to_numpy().astype(np.int64)
    order = np.argsort(w, kind="stable")
    ws = w[order]
    ui = np.searchsorted(elos, d["elo"].to_numpy())[order]
    tci = np.searchsorted(TCS, d["speed"].to_numpy())[order]
    pid = d["player"].to_numpy()[order]
    _, start, cnt = np.unique(ws, return_index=True, return_counts=True)
    wi = np.searchsorted(np.unique(ws), ws)
    se = np.zeros(len(ws))
    sq = np.zeros(len(ws))
    n = np.zeros(len(ws))
    for c0 in range(0, len(ws), chunk):
        i = np.arange(c0, min(c0 + chunk, len(ws)))
        lens = cnt[wi[i]]
        a = np.repeat(i, lens)
        off = np.repeat(np.cumsum(lens) - lens, lens)
        b = np.repeat(start[wi[i]], lens) + (np.arange(lens.sum()) - off)
        ok = pid[a] != pid[b]
        a, b = a[ok], b[ok]
        ev = E[tci[a], ui[a], ui[b]]
        se += np.bincount(a, ev, len(ws))
        sq += np.bincount(a, np.abs(ev - 0.5), len(ws))
        n += np.bincount(a, None, len(ws))
    e_bar = np.full(len(ws), np.nan)
    q_bar = np.full(len(ws), np.nan)
    e_bar[order] = np.where(n > 0, se / np.maximum(n, 1), np.nan)
    q_bar[order] = np.where(n > 0, sq / np.maximum(n, 1), np.nan)
    return e_bar, q_bar


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
    elos = np.unique(np.r_[d["elo"].to_numpy(), d["opp_elo"].to_numpy()])
    E = e_grid(elos)
    m = d.filter(pl.col("layer") == "multi").sort("bid", "game_id", "color")
    e_m, q_m = expected_multi(multi_null(m, E, elos))
    e_w, q_w = expected_window(d, E, elos)
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
