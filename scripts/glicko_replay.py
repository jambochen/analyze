# 從對局史逐局重放 Lichess 的 Glicko-2，能否重現表頭的對局前等級分？
# 能重現之處，重放的 RD 與 volatility 才可採用。規則來源見 scripts/glicko.py 與
# docs/glicko-replay.md。
#
# 資料始於 2013-01，此前已有 rated 對局，多數帳號在資料內的首局表頭不是 1500，
# 從 1500 起的純重放因此大多不相符，誤差又經對手擴散。主分析改用以表頭校正的重放：
# 每局先以表頭隱含的區間校正對局前等級分，再照 lila 的規則更新；RD 與 volatility
# 完全由重放產生。相符的判準是「校正前」由上一局推出的等級分截斷後等於表頭。

import shutil
import tempfile
import time

import glicko
import glicko_rs
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

MN = np.iinfo(np.int32).min
# RatingFest 的結束時間：此前表頭是重算結果，此後為線上即時計算。只影響未更新對局的
# 表頭能否作為比對與校正的依據
T_FEST = "2014-07-31 00:00:00"
PERF_NAMES = ["Bullet", "Blitz", "Classical", "Correspondence"]
ERA_NAMES = ["recomputed (to 2014-07)", "online (2014-08 to 2014-11)", "online (2014-12 to 2015-06)"]
NM = 31
month_labels = [f"{2012 + (m + 11) // 12}-{(m + 11) % 12 + 1:02d}" for m in range(NM)]
EXP = ["0", "1-4", "5-9", "10-29", "30-99", "100-299", "300+"]
CH = 2_000_000

src = pl.scan_parquet("data/processed/lichess_games.parquet").select(
    "game_id", "start", "white", "black", "result", "white_elo", "black_elo",
    "white_rating_diff", "black_rating_diff", "base_seconds", "increment_seconds",
)
ids = glicko.player_codes(src.select("white", "black")).select("w_id", "b_id").collect(engine="streaming")
w_id = ids["w_id"].to_numpy()
b_id = ids["b_id"].to_numpy()
del ids

ts = lambda t: pl.lit(t).str.to_datetime(time_zone="UTC")
# 分兩次收集，避免整張寬表同時留在記憶體
base = glicko.inputs(src).select(
    "perf", "score", "rated",
    # 月檔依巴黎時間切月，2013-01 檔含 UTC 2012-12-31 的對局，故以 2012-12 為第 0 個月
    month=((pl.col("start").dt.year() - 2012) * 12 + pl.col("start").dt.month() - 12).cast(pl.UInt8),
    day=pl.col("start").dt.date(),
    era=(pl.when(pl.col("start") < ts(T_FEST)).then(0).when(pl.col("start") < ts(glicko.T_CORR)).then(1).otherwise(2)).cast(pl.UInt8),
).collect(engine="streaming")
n = base.height
perf = base["perf"].to_numpy()
score = base["score"].to_numpy()
rated = base["rated"].to_numpy()
month = base["month"].to_numpy()
era = base["era"].to_numpy()
day = base["day"]
del base
hdr = src.select(pl.col("white_elo", "black_elo", "white_rating_diff", "black_rating_diff").fill_null(MN)).collect(engine="streaming")
we, be, wdiff, bdiff = [hdr[c].to_numpy().copy() for c in hdr.columns]
del hdr

# 重算期間未更新的對局，表頭留著重算前的舊值（RatingFest 不改寫它），不可比、不作校正；
# 線上期間的表頭是建立對局時的等級分，不論之後是否更新都可比
w_cmp = (we != MN) & (rated | (era > 0))
b_cmp = (be != MN) & (rated | (era > 0))
print("recomputed-era unrated games excluded from comparison:", ((~rated) & (era == 0)).sum())
we[~w_cmp] = MN
be[~b_cmp] = MN

# 規則解讀與參數的敏感度：每組只回傳 observe 的旗標，以 rayon 平行
perf_variants = {
    "adopted": {},
    "est40_from_commit_time": {"t_est40": "2015-04-24 14:27:00"},
    "est30_throughout": {"t_est40": "2099-01-01 00:00:00"},
    "unlimited_classical_until_2014-12-30": {"unlimited_classical_until_unl": True},
    "correspondence_from_2014-12-01_00h": {"t_corr": "2014-12-01 00:00:00"},
    "correspondence_from_2014-12-02_00h": {"t_corr": "2014-12-02 00:00:00"},
}
perfs = [perf] + [glicko.inputs(src, **kw).select("perf").collect(engine="streaming")["perf"].to_numpy() for kw in list(perf_variants.values())[1:]]
rated_all = src.select((pl.col("result") != "*") & (pl.col("white") != pl.col("black"))).collect(engine="streaming").to_series().to_numpy()
# 分出勝負卻雙方 RatingDiff 皆為 0 的對局；兩個新帳號 1500 對 1500 時不可能為 0，
# 可能是未更新卻記了 0，但也包含分差懸殊、變動不到 1 分的真實更新
rated_nozero = rated & ~((score != 0.5) & (wdiff == 0) & (bdiff == 0))
T, F, A = glicko.TAU, glicko.MIN_RATING, glicko.RD_TRUNC
configs = {name: (k, 0, T, F, True, True, A) for k, name in enumerate(perf_variants)} | {
    "pure_replay_from_1500": (0, 0, T, F, False, True, A),
    "header_as_rounded": (0, 0, T, F, True, False, A),
    "rd_trunc_350": (0, 0, T, F, True, True, 350.0),
    "rd_trunc_150": (0, 0, T, F, True, True, 150.0),
    "rd_trunc_80": (0, 0, T, F, True, True, 80.0),
    "rd_trunc_45": (0, 0, T, F, True, True, 45.0),
    "update_regardless_of_rating_diff": (0, 1, T, F, True, True, A),
    "zero_diff_decisive_not_updated": (0, 2, T, F, True, True, A),
    "tau_0.5": (0, 0, 0.5, F, True, True, A),
    "tau_1.0": (0, 0, 1.0, F, True, True, A),
    "tau_1.2": (0, 0, 1.2, F, True, True, A),
    "no_rating_floor": (0, 0, T, 0.0, True, True, A),
}
win = (
    day.is_between(pl.date(2014, 7, 20), pl.date(2014, 8, 10))
    | day.is_between(pl.date(2014, 11, 25), pl.date(2015, 1, 5))
    | day.is_between(pl.date(2015, 4, 20), pl.date(2015, 5, 1))
).to_numpy()
rows, drows = [], []
t_variants = 0.0
names = list(configs)
# 分批跑，每批的旗標陣列在彙總後即釋放
for b0 in range(0, len(names), 6):
    batch = names[b0 : b0 + 6]
    t0 = time.perf_counter()
    bits = glicko_rs.replay_match(w_id, b_id, score, we, be, perfs, [rated, rated_all, rated_nozero], [configs[c] for c in batch], glicko.N_PERF)
    t_variants += time.perf_counter() - t0
    for name, m in zip(batch, bits):
        pv = perfs[configs[name][0]]
        for cmp, sh in [(w_cmp, 0), (b_cmp, 4)]:
            f = m[cmp] >> sh
            idx = month[cmp].astype(np.int64) * 4 + pv[cmp]
            rows.append(pl.DataFrame({
                "variant": name, "cell": np.arange(NM * 4),
                "n": np.bincount(idx, minlength=NM * 4),
                "n_point": np.bincount(idx, weights=(f & 1) > 0, minlength=NM * 4),
                "n_consistent": np.bincount(idx, weights=(f & 2) > 0, minlength=NM * 4),
            }))
            if name in perf_variants or name == "pure_replay_from_1500":
                c = cmp & win
                drows.append(
                    pl.DataFrame({"day": day.filter(c), "perf": pv[c], "point": ((m[c] >> sh) & 1) > 0})
                    .group_by("day", "perf").agg(n=pl.len(), n_point=pl.col("point").sum())
                    .with_columns(variant=pl.lit(name))
                )
    del bits, m, f
del perfs, rated_all, rated_nozero

vm = (
    pl.concat(rows).group_by("variant", "cell").agg(pl.col("n", "n_point", "n_consistent").sum())
    .with_columns(month=(pl.col("cell") // 4).cast(pl.UInt8), perf=(pl.col("cell") % 4).cast(pl.UInt8)).drop("cell")
    .with_columns(era=pl.when(pl.col("month") < 20).then(0).when(pl.col("month") < 24).then(1).otherwise(2))
)
variants = (
    vm.group_by("variant", "era").agg(pl.col("n", "n_point", "n_consistent").sum())
    .with_columns(share_point=pl.col("n_point") / pl.col("n"), share_consistent=pl.col("n_consistent") / pl.col("n"))
    .sort("era", "share_point", descending=[False, True])
    .with_columns(era=pl.col("era").replace_strict(range(3), ERA_NAMES))
)
print(variants)
variants.write_csv("outputs/tables/glicko_replay_variants.csv")
(
    vm.filter(pl.col("n") > 0)
    .with_columns(share_point=pl.col("n_point") / pl.col("n"), month=pl.col("month").replace_strict(range(NM), month_labels), perf=pl.col("perf").replace_strict(range(4), PERF_NAMES))
    .sort("variant", "month", "perf")
    .write_csv("outputs/tables/glicko_replay_variants_month.csv")
)
daily = (
    pl.concat(drows).group_by("variant", "day", "perf").agg(pl.col("n", "n_point").sum())
    .with_columns(share_point=pl.col("n_point") / pl.col("n"), perf=pl.col("perf").replace_strict(range(4), PERF_NAMES))
    .sort("variant", "day", "perf")
)
daily.write_csv("outputs/tables/glicko_replay_daily.csv")
del rows, drows, win

# 純重放的差距分布：表頭減重放，說明左截斷造成的偏離有多大
pr = glicko_rs.replay(w_id, b_id, perf, score, rated, we, be, glicko.N_PERF, anchor=False)
pw, pb = pr[0], pr[3]
del pr
cnt = np.zeros(3 * 121)
for c, h, r in [(w_cmp, we, pw), (b_cmp, be, pb)]:
    cnt += np.bincount(era[c].astype(np.int64) * 121 + np.clip(h[c] - r[c].astype(np.int32), -60, 60) + 60, minlength=3 * 121)
pure_err = pl.DataFrame({"era": np.repeat(np.arange(3), 121), "err": np.tile(np.arange(-60, 61), 3), "n": cnt}).filter(pl.col("n") > 0)
pure_err.with_columns(era=pl.col("era").replace_strict(range(3), ERA_NAMES)).write_csv("outputs/tables/glicko_replay_pure_error.csv")
del pw, pb, cnt

t0 = time.perf_counter()
wr, wrd, wv, br_, brd, bv, wfl, bfl = glicko_rs.replay(w_id, b_id, perf, score, rated, we, be, glicko.N_PERF, rd_trunc=glicko.RD_TRUNC)
t_replay = time.perf_counter() - t0
print("full replay seconds", t_replay, "variants seconds", t_variants)
pl.DataFrame({"task": ["anchored_replay", "variants_parallel"], "n_games": n, "n_runs": [1, len(configs)], "seconds": [t_replay, t_variants]}).write_csv("outputs/tables/glicko_replay_full_timing.csv")
# RD 與 volatility 存單精度已足夠；rating 留雙精度，截斷比對才不受影響
wrd, wv, brd, bv = [a.astype(np.float32) for a in (wrd, wv, brd, bv)]
wf, bf, wnb, bnb = glicko_rs.header_chain(w_id, b_id, perf, we, be, wdiff, bdiff, rated, glicko.N_PERF)
# 移植本身的檢驗：雙方都是資料內首局且表頭為 1500 的對局，更新不受任何未知狀態影響，
# 從初始值算出的 RatingDiff 應與表頭相同
k = np.flatnonzero((wnb == 0) & (bnb == 0) & (we == 1500) & (be == 1500) & rated)
one = np.ones(len(k))
nw, nb_ = glicko_rs.step(1500 * one, 350 * one, 0.06 * one, 1500 * one, 350 * one, 0.06 * one, score[k])
newcomers = (
    pl.DataFrame({
        "era": era[k], "decisive": score[k] != 0.5,
        "match": (nw.astype(np.int32) - 1500 == wdiff[k]) & (nb_.astype(np.int32) - 1500 == bdiff[k]),
        "zero_diff": (wdiff[k] == 0) & (bdiff[k] == 0),
    })
    .group_by("era", "decisive").agg(n=pl.len(), share_match=pl.col("match").mean(), n_mismatch_zero_diff=(~pl.col("match") & pl.col("zero_diff")).sum(), n_mismatch=(~pl.col("match")).sum())
    .sort("era", "decisive")
    .with_columns(era=pl.col("era").replace_strict(range(3), ERA_NAMES))
)
print(newcomers)
newcomers.write_csv("outputs/tables/glicko_replay_newcomers.csv")
del k, one, nw, nb_
del w_id, b_id
clean = lambda f: ((f & 4) > 0) & ((f & 8) == 0)

# 以校正後的對局前等級分配上重放的 RD 與 volatility 做一次更新，預測表頭的 RatingDiff。
# 對照組把 RD 換成重放 RD 的中位數、volatility 換成初始值：若重放的 RD 帶有資訊，
# 對照組的預測應明顯較差。分段計算以控制記憶體
rated_both = rated & (we != MN) & (be != MN)
sub = np.flatnonzero(rated_both)[::10]
med_rd = float(np.median(np.concatenate([wrd[sub], brd[sub]])))
del sub
cnt = np.zeros((2, 3, 12))
for s0 in range(0, n, CH):
    k = np.flatnonzero(rated_both[s0 : s0 + CH]) + s0
    ek = era[k].astype(np.int64) * 4 + perf[k]
    m = len(k)
    for j, (rdw, rdb, vw, vb) in enumerate([
        (wrd[k].astype(np.float64), brd[k].astype(np.float64), wv[k].astype(np.float64), bv[k].astype(np.float64)),
        (np.full(m, med_rd), np.full(m, med_rd), np.full(m, glicko.DEFAULT_VOLATILITY), np.full(m, glicko.DEFAULT_VOLATILITY)),
    ]):
        nw, nb_ = glicko_rs.step(wr[k], rdw, vw, br_[k], rdb, vb, score[k])
        for new, old, d in [(nw, wr[k], wdiff[k]), (nb_, br_[k], bdiff[k])]:
            err = np.abs(new.astype(np.int32) - old.astype(np.int32) - d)
            cnt[j, 0] += np.bincount(ek, minlength=12)
            cnt[j, 1] += np.bincount(ek, weights=err == 0, minlength=12)
            cnt[j, 2] += np.bincount(ek, weights=err <= 1, minlength=12)
step = (
    pl.DataFrame({
        "rd_source": np.repeat(["replayed_rd", "constant_median_rd"], 12), "cell": np.tile(np.arange(12), 2),
        "n": cnt[:, 0].ravel(), "n_exact": cnt[:, 1].ravel(), "n_within_1": cnt[:, 2].ravel(),
    })
    .filter(pl.col("n") > 0)
    .with_columns(share_exact=pl.col("n_exact") / pl.col("n"), share_within_1=pl.col("n_within_1") / pl.col("n"), median_rd_used=med_rd)
    .with_columns(era=(pl.col("cell") // 4).replace_strict(range(3), ERA_NAMES), perf=(pl.col("cell") % 4).replace_strict(range(4), PERF_NAMES))
    .drop("cell")
)
print(step)
step.write_csv("outputs/tables/glicko_replay_step.csv")
del k, ek, nw, nb_, err, rdw, rdb, vw, vb, score, wdiff, bdiff, rated_both

# 分段寫出再串流合併，避免整張輸出表與 game_id 字串同時留在記憶體
tmp = tempfile.mkdtemp()
parts = []
for s0 in range(0, n, CH):
    sl = slice(s0, s0 + CH)
    part = pl.DataFrame({
        "game_id": src.slice(s0, CH).select("game_id").collect().to_series(),
        "perf": pl.Series(perf[sl]).replace_strict(range(4), PERF_NAMES),
        "white_rating": wr[sl], "white_rd": wrd[sl], "white_volatility": wv[sl],
        "black_rating": br_[sl], "black_rd": brd[sl], "black_volatility": bv[sl],
        # 表頭不可比（缺值，或重算期間未更新而留舊值）時為 null
        "white_match": pl.Series((wfl[sl] & 1) > 0).set(pl.Series(we[sl] == MN), None),
        "black_match": pl.Series((bfl[sl] & 1) > 0).set(pl.Series(be[sl] == MN), None),
        "white_left_truncated": (wf[sl] & 4) == 0,
        "black_left_truncated": (bf[sl] & 4) == 0,
        "white_prior_games": wnb[sl], "black_prior_games": bnb[sl],
        "white_header_clean": clean(wf[sl]), "black_header_clean": clean(bf[sl]),
    })
    parts.append(f"{tmp}/{s0:09d}.parquet")
    part.write_parquet(parts[-1])
del part
pl.scan_parquet(parts).sink_parquet("data/processed/glicko_replay.parquet")
shutil.rmtree(tmp)
del wv, bv, wr, br_

# 以下為逐「方」的分層：白黑兩方各一列，只留可比的方。52M 列的分組平均以 bincount 計算，
# 比 polars group_by 省記憶體
cat2 = lambda w, b_: np.concatenate([w[w_cmp], b_[b_cmp]])
s_month, s_era, s_perf = cat2(month, month), cat2(era, era), cat2(perf, perf)
s_point, s_cons = cat2((wfl & 1) > 0, (bfl & 1) > 0), cat2((wfl & 2) > 0, (bfl & 2) > 0)
s_nb, s_f, s_of = cat2(wnb, bnb), cat2(wf, bf), cat2(bf, wf)
s_rd = cat2(wrd, brd)
del wrd, brd, wnb, bnb, wfl, bfl, wf, bf
s_clean = clean(s_f)
s_both = s_clean & clean(s_of)
s_lt = (s_f & 4) == 0
# 首局沒有上一局可推，point 只反映起始值是否為 1500，不計入相符比例
s_later = s_nb > 0
s_exp = np.searchsorted([0, 4, 9, 29, 99, 299], s_nb, side="left").astype(np.uint8)
# 不相符的歸因：依只看表頭的對局史分類，類別依序互斥
CATEGORIES = ["header_break_at_this_game", "left_truncated", "earlier_header_break", "own_history_clean_opponent_not", "both_histories_clean"]
s_cat = np.select([(s_f & 1) == 0, s_lt, (s_f & 8) > 0, ~clean(s_of)], [0, 1, 2, 3], 4).astype(np.uint8)


def tab(keys, sizes, labels):
    idx = np.ravel_multi_index(keys, sizes)
    m = int(np.prod(sizes))
    bc = lambda w=None: np.bincount(idx, weights=w, minlength=m)
    out = pl.DataFrame({
        **{name: np.unravel_index(np.arange(m), sizes)[i] for i, name in enumerate(labels)},
        "n": bc(), "n_later": bc(s_later), "n_point": bc(s_point & s_later), "n_consistent": bc(s_cons & s_later),
        "n_both_clean": bc(s_both), "n_later_both_clean": bc(s_later & s_both), "n_point_both_clean": bc(s_point & s_later & s_both),
        "n_left_truncated": bc(s_lt), "n_prev_unrated": bc(s_later & ((s_f & 16) > 0)),
        "n_chain_ok_prev_unrated": bc(s_later & ((s_f & 16) > 0) & ((s_f & 1) > 0)),
        "n_chain_ok_prev_rated": bc(s_later & ((s_f & 16) == 0) & ((s_f & 1) > 0)),
    }).filter(pl.col("n") > 0)
    return out.with_columns(
        share_point=pl.col("n_point") / pl.col("n_later"),
        share_consistent=pl.col("n_consistent") / pl.col("n_later"),
        share_point_both_clean=pl.col("n_point_both_clean") / pl.col("n_later_both_clean"),
        share_both_clean=pl.col("n_both_clean") / pl.col("n"),
        share_left_truncated=pl.col("n_left_truncated") / pl.col("n"),
    )


main_cols = ["n", "share_point", "share_consistent", "share_point_both_clean", "share_both_clean", "share_left_truncated"]
by_month = tab([s_month, s_perf], (NM, 4), ["month", "perf"]).select(
    pl.col("month").replace_strict(range(NM), month_labels), pl.col("perf").replace_strict(range(4), PERF_NAMES), *main_cols
)
print(by_month)
by_month.write_csv("outputs/tables/glicko_replay_month.csv")

by_era = tab([s_era, s_perf], (3, 4), ["era", "perf"]).select(
    pl.col("era").replace_strict(range(3), ERA_NAMES), pl.col("perf").replace_strict(range(4), PERF_NAMES), *main_cols
)
print(by_era)
by_era.write_csv("outputs/tables/glicko_replay_era.csv")

sub = slice(None, None, 20)
rd_q = (
    pl.DataFrame({"era": s_era[sub], "exp": s_exp[sub], "lt": s_lt[sub], "rd": s_rd[sub]})
    .group_by("era", "exp", "lt").agg(median_rd=pl.col("rd").median(), p90_rd=pl.col("rd").quantile(0.9))
)
by_exp = (
    tab([s_era, s_exp, s_lt.astype(np.uint8)], (3, 7, 2), ["era", "exp", "lt"])
    .join(rd_q.with_columns(pl.col("era", "exp").cast(pl.Int64), lt=pl.col("lt").cast(pl.Int64)), on=["era", "exp", "lt"], how="left")
    .sort("era", "lt", "exp")
    .select(
        pl.col("era").replace_strict(range(3), ERA_NAMES), pl.col("exp").replace_strict(range(7), EXP),
        (pl.col("lt") == 1).alias("left_truncated"), *main_cols, "median_rd", "p90_rd",
    )
)
print(by_exp)
by_exp.write_csv("outputs/tables/glicko_replay_experience.csv")

attr = (
    tab([s_era, s_cat], (3, 5), ["era", "category"])
    .with_columns(n_mismatch=pl.col("n_later") - pl.col("n_point"))
    .with_columns(share_of_sides=pl.col("n_later") / pl.col("n_later").sum().over("era"), share_of_mismatch=pl.col("n_mismatch") / pl.col("n_mismatch").sum().over("era"))
    .select(pl.col("era").replace_strict(range(3), ERA_NAMES), pl.col("category").replace_strict(range(5), CATEGORIES), n="n_later", n_mismatch="n_mismatch", share_point="share_point", share_of_sides="share_of_sides", share_of_mismatch="share_of_mismatch")
)
print(attr)
attr.write_csv("outputs/tables/glicko_replay_attribution.csv")

# RatingDiff 缺失 = 未更新的檢驗，只看表頭：上一局未更新時，本局表頭應等於上一局的表頭
unrated_check = tab([s_era], (3,), ["era"]).select(
    pl.col("era").replace_strict(range(3), ERA_NAMES),
    n_after_unrated="n_prev_unrated",
    share_chain_ok_after_unrated=pl.col("n_chain_ok_prev_unrated") / pl.col("n_prev_unrated"),
    share_chain_ok_after_rated=pl.col("n_chain_ok_prev_rated") / (pl.col("n_later") - pl.col("n_prev_unrated")),
)
print(unrated_check)
unrated_check.write_csv("outputs/tables/glicko_replay_unrated.csv")
del s_month, s_era, s_perf, s_point, s_cons, s_nb, s_f, s_of, s_rd, s_clean, s_both, s_lt, s_later, s_exp, s_cat

colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
ink, ink2, grid = "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({"axes.edgecolor": ink2, "axes.labelcolor": ink, "xtick.color": ink2, "ytick.color": ink2, "font.size": 9})


def style(ax):
    ax.grid(axis="y", color=grid, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)


fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, col, title in [(axes[0], "share_point", "All comparable sides"), (axes[1], "share_point_both_clean", "Both header histories clean")]:
    for p, c in zip(PERF_NAMES, colors):
        d = by_month.filter(pl.col("perf") == p, pl.col("n") >= 500)
        ax.plot([month_labels.index(m) for m in d["month"]], d[col], color=c, lw=2, label=p)
    for xv, lab in [(19.97, "recomputation ends"), (23.03, "correspondence split")]:
        ax.axvline(xv, color=ink2, lw=0.8, ls="--")
        ax.text(xv + 0.2, 0.02, lab, rotation=90, color=ink2, fontsize=8, va="bottom")
    ax.set_xticks(range(1, NM, 3), month_labels[1::3], rotation=45)
    ax.set_title(title, color=ink, loc="left")
    ax.set_ylim(0, 1)
    style(ax)
axes[0].set_ylabel("Share of replayed pre-game ratings equal to header")
axes[0].legend(frameon=False, loc="lower left", bbox_to_anchor=(0.1, 0.05))
fig.suptitle("Anchored Glicko-2 replay vs. header pre-game rating, by month and speed", color=ink, x=0.01, ha="left")
fig.tight_layout()
fig.savefig("outputs/figures/glicko_replay_month.png", dpi=150)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, col, ylab in [(axes[0], "share_point", "Share equal to header"), (axes[1], "median_rd", "Median replayed RD")]:
    for e, c in zip(range(3), colors):
        for lt, ls in [(False, "-"), (True, ":")]:
            d = by_exp.filter(pl.col("era") == ERA_NAMES[e], pl.col("left_truncated") == lt)
            ax.plot([EXP.index(x) for x in d["exp"]], d[col], color=c, lw=2, ls=ls, marker="o", ms=4,
                    label=f"{ERA_NAMES[e]}{', history before 2013' if lt else ''}")
    ax.set_xticks(range(7), EXP)
    ax.set_xlabel("Prior games of the account in the same speed (in data)")
    ax.set_ylabel(ylab)
    style(ax)
axes[0].set_ylim(0, 1)
axes[1].legend(frameon=False, fontsize=7)
fig.suptitle("Replay agreement and replayed RD by account experience", color=ink, x=0.01, ha="left")
fig.tight_layout()
fig.savefig("outputs/figures/glicko_replay_experience.png", dpi=150)

fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
for ax, (lo, hi, title) in zip(axes, [((2014, 7, 20), (2014, 8, 10), "Recomputation ends"), ((2014, 11, 25), (2015, 1, 5), "Correspondence split"), ((2015, 4, 20), (2015, 5, 1), "Clock estimate 30 to 40 s/move")]):
    for p, c in zip(PERF_NAMES, colors):
        d = daily.filter(pl.col("variant") == "adopted", pl.col("perf") == p, pl.col("day").is_between(pl.date(*lo), pl.date(*hi)), pl.col("n") >= 50)
        ax.plot(d["day"], d["share_point"], color=c, lw=2, label=p)
    ax.set_title(title, color=ink, loc="left")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=45)
    style(ax)
axes[0].set_ylabel("Share equal to header")
axes[0].legend(frameon=False, loc="lower left")
fig.suptitle("Daily agreement around rule changes (adopted rules)", color=ink, x=0.01, ha="left")
fig.tight_layout()
fig.savefig("outputs/figures/glicko_replay_daily.png", dpi=150)

fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
for ax, e in zip(axes, range(3)):
    d = pure_err.filter(pl.col("era") == e)
    ax.bar(d["err"], d["n"], width=1.0, color=colors[0])
    ax.set_title(ERA_NAMES[e], color=ink, loc="left", fontsize=9)
    ax.set_xlabel("Header minus pure replay (clipped at ±60)")
    style(ax)
axes[0].set_ylabel("Sides")
fig.suptitle("Pure replay from 1500: size of disagreement with header", color=ink, x=0.01, ha="left")
fig.tight_layout()
fig.savefig("outputs/figures/glicko_replay_pure_error.png", dpi=150)
