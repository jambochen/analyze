# 兩局以上的配對批次在批內依 lila 規則重新配對的虛無分布，供 permutation test 共用。
# 配法的枚舉、期望得分的查表、重抽的種子與平行池都收在這裡：caller 只拿每列的期望得分，
# 不必知道查表的索引，也不必自己維持「每局兩列相鄰」的排序。

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import polars as pl

from calibrated import probs

TCS = ["Blitz", "Bullet", "Classical"]
NULLS = ["main", "unrestricted_before_1130"]


def _e_grid(elos):
    # E[速度類別, 自身, 對手]：color=False 的校準期望得分
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


def structure(d):
    """d 為 arena_batches 的玩家局（含 layer、rule、n_games、after_1130、prev），任意順序。回傳 (m, ns)。

    m 為兩局以上的批，依 bid、game_id、color 排序；ns 的陣列與 m 的列對齊。
    ns["e_act"] 為實際對手下的期望得分，ns["bi"]、ns["nb"] 為批的代碼與批數。

    純隨機的批（開賽批或 7 局以上）：批內均勻隨機兩兩配對。
    12 人以下的批：在不含「上一局剛對過」的配法中均勻抽出；實際配法本身含重配的批，表示規則當時沒有作用，改用不加限制的配法。
    unrestricted_before_1130 另把 2014-11-30 以前的 12 人以下批全部改為不加限制：此前一局一批的重配不符原始碼，規則是否完整作用存疑。
    """
    elos = np.unique(np.r_[d["elo"].to_numpy(), d["opp_elo"].to_numpy()])
    E = _e_grid(elos)
    m = d.filter(pl.col("layer") == "multi").sort("bid", "game_id", "color")
    gid = m["game_id"].to_numpy()
    # 每局兩列相鄰，實際對手即相鄰的那一列；重抽與查表都依賴這個排列
    assert (gid == gid[np.arange(m.height) ^ 1]).all()
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
    ui = np.searchsorted(elos, m["elo"].to_numpy())
    tci = np.searchsorted(TCS, m["speed"].to_numpy())
    ns = dict(
        bi=bi, nb=bi.max() + 1, starts=starts, smart=smart, rand_rows=np.flatnonzero(rule == "random"),
        ui=ui, tci=tci, E=E, elos=elos, e_act=E[tci, ui, ui[np.arange(m.height) ^ 1]],
    )
    return m, ns


def _repair(ns, null, rng):
    # 重抽一次批內配法，回傳每個玩家位置的新對手位置
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


def draw(ns, null, seed):
    """依 null 重抽一次配法，回傳 m 每列對新對手的期望得分。"""
    ui = ns["ui"]
    return ns["E"][ns["tci"], ui, ui[_repair(ns, null, np.random.default_rng(seed))]]


_job_state = None


def _job(job):
    ns, stat, chunk = _job_state
    k, i0 = job
    out = [stat(draw(ns, NULLS[k], [k, i])) for i in range(i0, i0 + chunk)]
    print(NULLS[k], i0 + chunk, "draws", flush=True)
    return out


def run(ns, stat, B, chunk=20):
    """每個虛無分布各重抽 B 次，回傳 {null: [stat(e) 依抽次排列]}；e 為 draw 的輸出。

    第 i 次的種子為 [null 的序號, i]，結果不隨工作分到哪個行程而變。stat 在 fork 出的行程內執行，
    呼叫 run 之前不可啟動 jax 的執行緒（fork 後可能死結）。
    """
    global _job_state
    _job_state = (ns, stat, chunk)
    # 兩個虛無分布的工作一起送出，最後一波才不會只剩少數行程在跑；
    # 每個工作一次跑 chunk 次重抽，避免時間花在行程間的往返
    jobs = [(k, i) for k in range(len(NULLS)) for i in range(0, B, chunk)]
    with ProcessPoolExecutor(os.cpu_count() - 2, mp_context=mp.get_context("fork")) as ex:
        res = list(ex.map(_job, jobs))
    return {nl: [s for (k, _), r in zip(jobs, res) if k == kn for s in r] for kn, nl in enumerate(NULLS)}


def expected(ns):
    """主虛無分布下 m 每列的期望 e 與期望 q（精確值，不靠模擬）。"""
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


def expected_window(d, ns, chunk=100_000):
    """一局一批的近似：d 每列以同一場錦標賽同一 5 分鐘內開局的其他玩家為候選對手，取對他們的平均 e 與 q。
    ns 須由含 d 全部列的資料建構，查表才涵蓋 d 的等級分。"""
    E, elos = ns["E"], ns["elos"]
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
