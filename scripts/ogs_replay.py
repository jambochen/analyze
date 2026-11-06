# OGS 等級分的重放：資料庫沒有對局前的等級分，依 OGS 公開的分析程式（ref/goratings-src，
# analysis/analyze_glicko2_one_game_at_a_time.py 與其預設參數）逐局重算，得到每局對局前雙方的等級分、RD 與系統的期望勝率。
# 規則照抄參考實作：Glicko-2（tao 0.5、RD 夾在 10 至 500，未設 aging period），等級分與段位以 log 制換算（a = 525、c = 23.15），
# 讓子與貼目換算成段位差再換回等級分；通訊棋只把連續超時的第一局算進等級分；排除舊機器人（id <= 50000）與 randombot。
# 與參考實作的差別：同一秒結束的對局另以對局代碼排序，結果才可重現；極端的讓子（如 9 路讓 9 子，換算成數十段）使期望勝率
# 捨入成 0 或 1，參考實作的 volatility 迭代在爆冷時溢位而中斷，這類對局改記為不計分（overflow），不更新雙方的等級分。
# 期望勝率照參考實作的 expected_win_probability(ignore_g=True)：乘上的 g 取白方的 RD，對手的 RD 不進入

import sqlite3
from math import exp, log, pi, sqrt

import polars as pl

TAO, EPS, MIN_RD, MAX_RD = 0.5, 0.000001, 10.0, 500.0
SCALE = 173.7178
A_RANK, C_RANK = 525.0, 23.15
RULES = {"aga": "aga", "chinese": "chinese", "ing": "ing", "japanese": "japanese", "korean": "korean", "nz": "nz",
         "Japanese": "japanese", "age": "aga", "ing sst": "ing", "ogs": "japanese"}


def rank_of(r):
    return log(r / A_RANK) * C_RANK


def rating_of(k):
    return A_RANK * exp(k / C_RANK)


def rank_diff(handicap, size, komi, rules):
    extra = handicap - 1 if handicap > 1 else 0
    if rules in ("japanese", "korean"):
        area, bonus = 0, 0
    else:
        area = 1
        bonus = handicap if rules == "chinese" else extra if rules == "aga" else 0
    head = 6 + area - (komi + bonus) + (12 + area) * extra
    return head * 6 / 12 if size == 9 else head * 3 / 12 if size == 13 else head / 12


def adjust(color, r, hd):
    return rating_of(rank_of(r) + (hd if color == "black" else -hd)) - r


def aged_rd(p, t):
    # 參考實作的 after_aging_to_timestamp：未設 aging period 時 aging_factor 為 1，只要時間前進，
    # 每局之前 RD 就以一期的 volatility 膨脹一次；新帳號沒有時間戳，不膨脹
    r, rd, vol, ts = p
    if ts and t > ts:
        return min(MAX_RD, max(MIN_RD, SCALE * sqrt((rd / SCALE) ** 2 + vol**2)))
    return rd


def update(p, o_rating, o_rd, outcome, t):
    # p 為 (rating, rd, volatility, 上次更新的時間)；對手為已加讓子調整的等級分與膨脹後的 RD。參考實作 glicko2_update 的單局情形
    r, _, vol, _ = p
    rd = aged_rd(p, t)
    mu, phi = (r - 1500) / SCALE, rd / SCALE
    mu_j, phi_j = (o_rating - 1500) / SCALE, o_rd / SCALE
    g = 1 / sqrt(1 + 3 * phi_j**2 / pi**2)
    E = 1 / (1 + exp(-g * (mu - mu_j)))
    v_sum = g**2 * E * (1 - E)
    d_sum = g * (outcome - E)
    v = 1.0 / v_sum if v_sum else 9999
    delta = v * d_sum
    a = log(vol**2)

    def f(x):
        ex = exp(x)
        return ex * (delta**2 - phi**2 - v - ex) / (2 * (phi**2 + v + ex) ** 2) - (x - a) / TAO**2

    A = a
    if delta**2 > phi**2 + v:
        B = log(delta**2 - phi**2 - v)
    else:
        k, safety = 1, 100
        while f(a - k * TAO) < 0 and safety > 0:
            safety -= 1
            k += 1
        B = a - k * TAO
    fA, fB, safety = f(A), f(B), 100
    while abs(B - A) > EPS and safety > 0:
        C = A + (A - B) * fA / (fB - fA)
        fC = f(C)
        if fC * fB <= 0:
            A, fA = B, fB
        else:
            fA = fA / 2
        B, fB = C, fC
        safety -= 1
    new_vol = exp(A / 2)
    phi_star = sqrt(phi**2 + new_vol**2)
    phi_p = 1 / sqrt(1 / phi_star**2 + 1 / v)
    mu_p = mu + phi_p**2 * d_sum
    return (min(6000.0, max(100.0, SCALE * mu_p + 1500)), min(MAX_RD, max(MIN_RD, SCALE * phi_p)), min(0.15, max(0.01, new_vol)), t)


con = sqlite3.connect("file:data/raw/ogs/ogs-data.db?mode=ro", uri=True)
q = """
SELECT g.id, size, handicap, komi, black_id, white_id, time_per_move, timeout, winner_id, ended, rules
FROM game_records g
LEFT JOIN players bp ON black_id = bp.id
LEFT JOIN players wp ON white_id = wp.id
WHERE (bp.is_bot = 0 OR bp.id > 50000) AND (wp.is_bot = 0 OR wp.id > 50000)
  AND black_id != 82957 AND white_id != 82957
  AND (bp.is_bot != 1 OR timeout = 0) AND (wp.is_bot != 1 OR timeout = 0)
ORDER BY ended, g.id
"""
state, flag = {}, {}
NEW = (1500.0, 350.0, 0.06, None)
cols = {k: [] for k in ["game_id", "black_rating", "black_rd", "white_rating", "white_rd", "handicap_rank", "expected_black", "skipped", "overflow"]}
for gid, size, hc, komi, b, w, tpm, timeout, win, ended, rules in con.execute(q):
    rules = RULES[rules]
    corr = tpm == 0 or tpm > 3600
    skip = False
    if timeout and corr:
        loser = b if b != win else w
        other = b if b == win else w
        skip = flag.get(b, False) or flag.get(w, False)
        flag[loser], flag[other] = True, False
    elif corr:
        flag[b], flag[w] = False, False
    pb, pw = state.get(b, NEW), state.get(w, NEW)
    hd = rank_diff(hc, size, komi, rules)
    adj_b = adjust("black", pb[0], hd)
    adj_w = adjust("white", pw[0], hd)
    cols["game_id"].append(gid)
    cols["black_rating"].append(pb[0])
    cols["black_rd"].append(pb[1])
    cols["white_rating"].append(pw[0])
    cols["white_rd"].append(pw[1])
    cols["handicap_rank"].append(hd)
    cols["expected_black"].append(1 / (1 + exp(-(1 / sqrt(1 + 3 * (pw[1] / SCALE) ** 2 / pi**2)) * (pb[0] + adj_b - pw[0]) / SCALE)))
    cols["skipped"].append(skip)
    if skip:
        cols["overflow"].append(False)
        continue
    try:
        nb = update(pb, pw[0] + adjust("white", pw[0], hd), aged_rd(pw, ended), 1.0 if win == b else 0.0, ended)
        nw = update(pw, pb[0] + adj_b, aged_rd(pb, ended), 1.0 if win == w else 0.0, ended)
    except OverflowError:
        cols["overflow"].append(True)
        continue
    cols["overflow"].append(False)
    state[b], state[w] = nb, nw

out = pl.DataFrame(cols)
print(out.height, out["skipped"].sum(), out["overflow"].sum())
out.write_parquet("data/processed/ogs_replay.parquet")
