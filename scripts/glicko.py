# Lichess 等級分更新的 Python 參考實作，逐行移植自 lila 的 org.goochjs.glicko2
# （modules/rating/src/main/java/glicko2/RatingCalculator.java @ bfad573339）與
# PerfsUpdater.scala。Rust 版（rust/glicko_replay）須與此逐局一致。
#
# 一局即一個 rating period，期間只有一個結果，所以 RatingCalculator 對閒置玩家
# 的 RD 上升分支不會觸發：兩人都有結果。
# Java 的 Math.pow(x, 2) 以 x * x 實作；兩者在 IEEE 雙精度下的差至多 1 ulp。

import math

MULTIPLIER = 173.7178
DEFAULT_RATING = 1500.0
DEFAULT_DEVIATION = 350.0
DEFAULT_VOLATILITY = 0.06
TAU = 0.75
CONVERGENCE_TOLERANCE = 0.000001
ITERATION_MAX = 3000
MIN_RATING = 800.0
# 資料起點前已有對局的帳號（首局表頭不是 1500），RD 未知，以此值起算；
# 依 scripts/glicko_replay.py 的 rd_trunc 變體取相符最高者
RD_TRUNC = 60.0
PI2 = math.pi * math.pi


def g(phi):
    return 1.0 / math.sqrt(1.0 + 3.0 * phi * phi / PI2)


def f(x, delta, phi, v, a, tau):
    ex = math.exp(x)
    d = phi * phi + v + ex
    return (ex * (delta * delta - phi * phi - v - ex) / (2.0 * d * d)) - ((x - a) / (tau * tau))


def new_rating(mu, phi, sigma, mu_o, phi_o, s, tau):
    # 回傳 None 表示 Java 端拋出 Convergence fail
    go = g(phi_o)
    e = 1.0 / (1.0 + math.exp(-1.0 * go * (mu - mu_o)))
    v = 1.0 / (go * go * e * (1.0 - e))
    outcome = go * (s - e)
    delta = v * outcome
    a = math.log(sigma * sigma)
    A = a
    if delta * delta > phi * phi + v:
        B = math.log(delta * delta - phi * phi - v)
    else:
        k = 1.0
        B = a - k * abs(tau)
        while f(B, delta, phi, v, a, tau) < 0:
            k += 1
            B = a - k * abs(tau)
    fA = f(A, delta, phi, v, a, tau)
    fB = f(B, delta, phi, v, a, tau)
    it = 0
    while abs(B - A) > CONVERGENCE_TOLERANCE and it < ITERATION_MAX:
        it += 1
        C = A + ((A - B) * fA) / (fB - fA)
        fC = f(C, delta, phi, v, a, tau)
        # Java 用嚴格的 < 0，Glickman 原文是 <= 0
        if fC * fB < 0:
            A = B
            fA = fB
        else:
            fA = fA / 2.0
        B = C
        fB = fC
    if it == ITERATION_MAX:
        return None
    new_sigma = math.exp(A / 2.0)
    phi_star = math.sqrt(phi * phi + new_sigma * new_sigma)
    new_phi = 1.0 / math.sqrt((1.0 / (phi_star * phi_star)) + (1.0 / v))
    return mu + new_phi * new_phi * outcome, new_phi, new_sigma


def update(w, b, s, tau=TAU, floor=MIN_RATING):
    # Perf.toRating 只在計算的輸入端套下限，存回去的是計算結果本身，可低於下限
    rw = max(floor, w[0])
    rb = max(floor, b[0])
    mw = (rw - DEFAULT_RATING) / MULTIPLIER
    pw = w[1] / MULTIPLIER
    mb = (rb - DEFAULT_RATING) / MULTIPLIER
    pb = b[1] / MULTIPLIER
    nw = new_rating(mw, pw, w[2], mb, pb, s, tau)
    nb = new_rating(mb, pb, b[2], mw, pw, 1.0 - s, tau)
    # 任一方不收斂時 updateRatings 在 finaliseRating 之前就拋出，雙方都保留輸入值，
    # 但 mkPerfs 仍把這個（已套下限的）輸入值存回
    if nw is None or nb is None:
        return (rw, w[1], w[2]), (rb, b[1], b[2])
    return (
        (nw[0] * MULTIPLIER + DEFAULT_RATING, nw[1] * MULTIPLIER, nw[2]),
        (nb[0] * MULTIPLIER + DEFAULT_RATING, nb[1] * MULTIPLIER, nb[2]),
    )


def observe(st, h, anchor, truncate, rd_trunc):
    # st = [lo, hi, rd, vol, updated]，就地修改。旗標位元：0 校正前的點估計與表頭相符、
    # 1 校正前的區間與表頭相容、2 不相容而重設、3 左截斷起算（首局表頭不是 1500）
    mid = (st[0] + st[1]) / 2.0
    point = int(mid) == h if truncate else round_half_away(mid) == h
    if not anchor or h is None:
        return point | point << 1
    hl, hh = (float(h), h + 1.0) if truncate else (h - 0.5, h + 0.5)
    if not st[4] and h != 1500:
        st[:] = [hl, hh, rd_trunc, DEFAULT_VOLATILITY, False]
        return point | 1 << 3
    nlo, nhi = max(st[0], hl), min(st[1], hh)
    if nlo < nhi or (st[0] == st[1] and hl <= st[0] < hh):
        if st[0] != st[1]:
            st[0], st[1] = nlo, nhi
        return point | 1 << 1
    st[0], st[1] = hl, hh
    return point | 1 << 2


def round_half_away(x):
    # Rust 的 f64::round 為遠離零的四捨五入，Python 的 round 為銀行家捨入
    return int(math.floor(x + 0.5)) if x >= 0 else -int(math.floor(-x + 0.5))


def advance(st, new, mid, floor):
    # 區間退化為一點時直接採用結果；整個區間在下限之下時，結果與區間內位置無關
    if st[0] == st[1] or st[1] <= floor:
        st[0] = st[1] = new[0]
    else:
        shift = new[0] - mid
        st[0] += shift
        st[1] += shift
    st[2], st[3], st[4] = new[1], new[2], True


def replay(white, black, perf, score, rated, white_anchor, black_anchor, tau=TAU, floor=MIN_RATING, anchor=True, truncate=True, rd_trunc=RD_TRUNC):
    # 輸入為依時間排序的序列；score 為白方得分；rated 為假時不更新但仍記錄對局前狀態。
    # 錨點（表頭等級分）不可用時為 None。回傳每局 (白 rating, RD, vol, 黑 rating, RD, vol, 白旗標, 黑旗標)
    state = {}
    out = []
    for w, b, p, s, r, ha, hb in zip(white, black, perf, score, rated, white_anchor, black_anchor):
        sw = state.setdefault((w, p), [DEFAULT_RATING, DEFAULT_RATING, DEFAULT_DEVIATION, DEFAULT_VOLATILITY, False])
        sb = state.setdefault((b, p), [DEFAULT_RATING, DEFAULT_RATING, DEFAULT_DEVIATION, DEFAULT_VOLATILITY, False])
        fw = observe(sw, ha, anchor, truncate, rd_trunc)
        fb = observe(sb, hb, anchor, truncate, rd_trunc)
        pw = ((sw[0] + sw[1]) / 2.0, sw[2], sw[3])
        pb = ((sb[0] + sb[1]) / 2.0, sb[2], sb[3])
        out.append(pw + pb + (fw, fb))
        if r:
            nw, nb = update(pw, pb, s, tau, floor)
            advance(sw, nw, pw[0], floor)
            advance(sb, nb, pb[0], floor)
    return out


# 速度類別規則依對局建立時間分段。commit 時間只是上線時間的下限，實際上線時間由
# scripts/explore/speed_boundaries.py 從表頭等級分鏈推定
BULLET, BLITZ, CLASSICAL, CORRESPONDENCE = 0, 1, 2, 3
N_PERF = 4
PERF_NAMES = ["Bullet", "Blitz", "Classical", "Correspondence"]
# scalachess 1fc3f0d44a 把 estimateTotalTime 的每步加秒倍數由 30 改為 40
T_EST40 = "2015-04-25 07:00:00"
# lila 804567893e 使通信棋獨立計分；此前無限時對局與 Classical 共用等級分
T_CORR = "2014-12-01 09:46:00"
# lila e6321758a9 起連無 daysPerTurn 的無限時對局也算通信棋
T_UNL = "2014-12-30 15:52:00"


def speed(t_est40=T_EST40, t_corr=T_CORR, unlimited_classical_until_unl=False):
    # 對局當時等級分所屬的速度類別（PERF_NAMES 的索引）。月檔 Event 的時間類別是匯出時
    # 一律以每步加秒乘 40 事後標的，2015-04-25 以前與當時計分所用的類別不同
    import polars as pl

    ts = lambda t: pl.lit(t).str.to_datetime(time_zone="UTC")
    est = pl.col("base_seconds") + pl.when(pl.col("start") < ts(t_est40)).then(30).otherwise(40) * pl.col("increment_seconds")
    t_classical_end = T_UNL if unlimited_classical_until_unl else t_corr
    return (
        pl.when(est.is_null() & (pl.col("start") < ts(t_classical_end))).then(CLASSICAL)
        .when(est.is_null()).then(CORRESPONDENCE)
        .when(est < 180).then(BULLET)
        .when(est < 480).then(BLITZ)
        .otherwise(CLASSICAL)
        .cast(pl.UInt8)
    )


def inputs(games, **speed_rule):
    import polars as pl

    return games.with_columns(
        perf=speed(**speed_rule),
        # 未完成（*）的局不更新，得分值無作用
        score=pl.when(pl.col("result") == "1-0").then(1.0).when(pl.col("result") == "0-1").then(0.0).otherwise(0.5),
        # RatingDiff 缺失表示系統沒有更新這局（作弊或灌分帳號、未滿兩步等）
        rated=pl.col("white_rating_diff").is_not_null()
        & pl.col("black_rating_diff").is_not_null()
        & (pl.col("white") != pl.col("black"))
        & (pl.col("result") != "*"),
    )


def player_codes(games):
    import polars as pl

    # 白黑兩欄須共用同一套代碼；以全部帳號名建 Enum，比把兩欄串起來轉 categorical 省記憶體
    names = pl.concat([games.select(n="white"), games.select(n="black")]).unique().collect(engine="streaming")["n"].sort()
    enum = pl.Enum(names)
    return games.with_columns(
        w_id=pl.col("white").cast(enum).to_physical().cast(pl.UInt32),
        b_id=pl.col("black").cast(enum).to_physical().cast(pl.UInt32),
    )


def header_chain(white, black, perf, white_elo, black_elo, white_diff, black_diff, rated):
    # 只用表頭的對局史檢查；旗標位元與 rust/glicko_replay 的 header_chain 相同。
    # 表頭不可用時為 None：不檢查、不改變預期；但若該局有更新，預期值變為未知
    st = {}
    out = []
    for w, b, p, we, be, wd, bd, r in zip(white, black, perf, white_elo, black_elo, white_diff, black_diff, rated):
        row = []
        for pid, elo, diff in [(w, we, wd), (b, be, bd)]:
            s = st.setdefault((pid, p), {"expect": 1500, "nb": 0, "seen": False, "origin": False, "broken": False, "prev_unrated": False})
            first = s["nb"] == 0
            ok = False
            if elo is not None:
                if not s["seen"]:
                    s["origin"] = elo == 1500
                    s["seen"] = True
                ok = elo == s["expect"]
                s["broken"] = s["broken"] or not ok
            row += [ok | first << 1 | s["origin"] << 2 | s["broken"] << 3 | s["prev_unrated"] << 4, s["nb"]]
            s["nb"] += 1
            if elo is not None:
                s["expect"] = elo + diff if r else elo
            elif r:
                s["expect"] = None
            s["prev_unrated"] = not r
        out.append((row[0], row[2], row[1], row[3]))
    return out
