# arena 的配對是否與等級分無關。以同一場錦標賽內開局時間相同（到秒）的對局為同一批配對，
# 在批內重新隨機配對作為虛無分布。依 lila 原始碼，批的首次配對與空閒超過 12 人時是純隨機，
# 其餘情形避免與上一局對手重複；兩種規則分開檢驗。
# 樣本為 2014 年全年與 2015 年 1 至 3 月（改制在 4 月 20 日）的 arena 對局。

import sys

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

sys.path.insert(0, "scripts")
from arena_batches import games, prev_games

UTC = "UTC"
B = 2000
rng = np.random.default_rng(0)

period = pl.when(pl.col("start").dt.year() == 2014).then(pl.lit("2014")).otherwise(pl.lit("2015Q1"))
g, batch, bad = games(pl.datetime(2014, 1, 1, time_zone=UTC), pl.datetime(2015, 4, 1, time_zone=UTC))
g = g.with_columns(period=period)
batch = batch.with_columns(period=period)
print(g.null_count())

sizes = (
    batch.group_by("period", "rule", "first", n_games=pl.col("n_games").clip(1, 10))
    .agg(batches=pl.len(), games=pl.col("n_games").sum(), dup_player_batches=(pl.col("n_players") < 2 * pl.col("n_games")).sum())
    .sort("period", "rule", "first", "n_games")
)
print(sizes)
sizes.write_csv("outputs/tables/arena_random_pairing_batch_size.csv")

# 含伺服器異常批次的錦標賽整場排除
sample = (
    g.group_by("period", excluded=pl.col("tournament_id").is_in(bad))
    .agg(tournaments=pl.col("tournament_id").n_unique(), batches=pl.col("bid").n_unique(), games=pl.len())
    .sort("period", "excluded")
)
print(sample)
sample.write_csv("outputs/tables/arena_random_pairing_sample.csv")
g = g.filter(~pl.col("tournament_id").is_in(bad))
batch = batch.filter(~pl.col("tournament_id").is_in(bad))

gaps = (
    batch.filter(pl.col("gap").is_not_null())
    .group_by("period", gap=pl.col("gap").clip(0, 61))
    .agg(batches=pl.len(), mean_games=pl.col("n_games").mean())
    .sort("period", "gap")
)
print(gaps.head(15))
gaps.write_csv("outputs/tables/arena_random_pairing_gaps.csv")

# API 的 createdAt 到毫秒。若表頭時間是 createdAt 截斷到秒，且一批的建立跨過秒界時會被拆成相隔 1 秒的兩批，
# 則 createdAt 的毫秒數越接近 999，下一秒緊接另一批的機率越高
api = (
    pl.read_ndjson("data/raw/lichess-tournament-source/games.ndjson")
    .select("id", "createdAt")
    .join(g.select("game_id", "start", "tournament_id", "bid"), left_on="id", right_on="game_id")
    .join(batch.select("bid", "n_games"), on="bid")
    .with_columns(offset=pl.col("createdAt") - pl.col("start").dt.epoch("ms"))
)
next_sec = batch.filter(pl.col("gap") == 1).select("tournament_id", start=pl.col("start") - pl.duration(seconds=1)).with_columns(next_1s=pl.lit(True))
api_ms = (
    api.join(next_sec, on=["tournament_id", "start"], how="left")
    .group_by(ms_bin=(pl.col("offset") // 100) * 100)
    .agg(games=pl.len(), next_batch_1s_later=pl.col("next_1s").fill_null(False).mean(), mean_batch_games=pl.col("n_games").mean())
    .sort("ms_bin")
)
print(api.select(n=pl.len(), in_second=((pl.col("offset") >= 0) & (pl.col("offset") < 1000)).sum()))
print(api_ms)
api_ms.write_csv("outputs/tables/arena_random_pairing_api_ms.csv")

# 批內置換：把批內 2g 位玩家重新隨機兩兩配對，與 lila 的 shuffle 後 grouped 2 同分布。
# 統計量對顏色對稱：批內去平均後，每對雙方等級分的乘積和。有限母體下其虛無期望為 -1/(2g-1)，不是 0
m = g.filter(pl.col("n_games") >= 2)
r = np.column_stack([m["white_elo"].to_numpy(), m["black_elo"].to_numpy()]).ravel().astype(float)
sbid = np.repeat(m["bid"].to_numpy(), 2)
uniq, sidx = np.unique(sbid, return_inverse=True)
d = r - (np.bincount(sidx, r) / np.bincount(sidx))[sidx]
labels = m.select("period", "rule", "size").with_columns(all=pl.lit("all"))
groups = {}
for cols in [["period", "rule"], ["period", "rule", "size"]]:
    key = labels.select(pl.concat_str(cols, separator="|")).to_series()
    names, gi = np.unique(key.to_numpy(), return_inverse=True)
    groups[tuple(cols)] = (names, gi)


def pair_stats(rr, dd, gi, k):
    num = np.bincount(gi, 2 * dd[0::2] * dd[1::2], k)
    den = np.bincount(gi, dd[0::2] ** 2 + dd[1::2] ** 2, k)
    absd = np.bincount(gi, np.abs(rr[0::2] - rr[1::2]), k) / np.bincount(gi, None, k)
    return num / den, absd


perm_rows = []
null_draws = []
for cols, (names, gi) in groups.items():
    k = len(names)
    act_c, act_a = pair_stats(r, d, gi, k)
    nc = np.empty((B, k))
    na = np.empty((B, k))
    for i in range(B):
        o = np.argsort(sidx + rng.random(len(sidx)))
        nc[i], na[i] = pair_stats(r[o], d[o], gi, k)
    for j, nm in enumerate(names):
        perm_rows.append(
            dict(
                group=nm,
                games=int(np.bincount(gi, None, k)[j]),
                corr=act_c[j],
                corr_null_mean=nc[:, j].mean(),
                corr_null_lo=np.quantile(nc[:, j], 0.025),
                corr_null_hi=np.quantile(nc[:, j], 0.975),
                corr_p=(1 + (np.abs(nc[:, j] - nc[:, j].mean()) >= abs(act_c[j] - nc[:, j].mean())).sum()) / (B + 1),
                absdiff=act_a[j],
                absdiff_null_mean=na[:, j].mean(),
                absdiff_null_lo=np.quantile(na[:, j], 0.025),
                absdiff_null_hi=np.quantile(na[:, j], 0.975),
                absdiff_p=(1 + (np.abs(na[:, j] - na[:, j].mean()) >= abs(act_a[j] - na[:, j].mean())).sum()) / (B + 1),
            )
        )
        if len(cols) == 2:
            null_draws.append((nm, act_c[j], nc[:, j]))
perm = pl.DataFrame(perm_rows)
with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250):
    print(perm)
perm.write_csv("outputs/tables/arena_random_pairing_permutation.csv")

# 上一局剛對過的兩人同時空閒時，12 人以下的規則不讓他們重配（剩兩人時寧可不配，除非整場只剩兩位活躍玩家）；
# 純隨機規則則以 1/(2g-1) 的機率重配。這是檢驗批次重建與原始碼是否相符的直接指標
long = prev_games(g)
rematch_game = long.group_by("game_id").agg(rematch=(pl.col("prev").n_unique() == 1) & pl.col("prev").is_not_null().all())
eligible = (
    long.filter(pl.col("prev").is_not_null())
    .group_by("bid", "prev")
    .agg(n=pl.len())
    .group_by("bid")
    .agg(eligible=(pl.col("n") == 2).sum())
)
rematch = (
    batch.join(eligible, on="bid", how="left")
    .join(g.select("bid", "game_id").join(rematch_game, on="game_id").group_by("bid").agg(rematches=pl.col("rematch").sum()), on="bid")
    .with_columns(pl.col("eligible").fill_null(0), expected=pl.col("eligible").fill_null(0) / (2 * pl.col("n_games") - 1))
    # 2014-11-30 起一局一批的重配幾乎消失，PairingSystem.scala 在此前後未改，原因不明，分開列出
    .group_by("period", "rule", "size", after_20141130=pl.col("start") >= pl.datetime(2014, 11, 30, time_zone=UTC))
    .agg(batches=pl.len(), eligible_pairs=pl.col("eligible").sum(), expected_rematches=pl.col("expected").sum(), rematches=pl.col("rematches").sum())
    .sort("period", "after_20141130", "rule", "size")
)
print(rematch)
rematch.write_csv("outputs/tables/arena_random_pairing_rematch.csv")

# 12 人以下且一批兩局（空閒 4 或 5 人）時，lila 列舉三種兩兩配法，含剛對過的一對者記 100 分；
# 其餘配法的分數都是全體等待配對數之和取負，彼此相同，故等待時間不影響批內怎麼配，只影響奇數時誰被剩下。
# 列舉順序來自打亂後的名單，因此實際配法是在不含重配的配法中均勻抽出。以此為精確的虛無分布，與不加限制的版本並列
m2 = (
    g.filter(pl.col("rule") == "smart", pl.col("n_games") == 2)
    .join(long.select("game_id", w="player", pw="prev"), on=["game_id", "w"])
    .join(long.select("game_id", b="player", pb="prev"), on=["game_id", "b"])
    .sort("bid", "game_id")
)
r4 = np.column_stack([m2["white_elo"].to_numpy(), m2["black_elo"].to_numpy()]).astype(float).reshape(-1, 4)
d4 = r4 - r4.mean(axis=1, keepdims=True)
pv = np.column_stack([m2["pw"].fill_null("").to_numpy(), m2["pb"].fill_null("").to_numpy()]).reshape(-1, 4)
per4 = m2["period"].to_numpy()[0::2]
match = [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]
absm = np.column_stack([np.abs(r4[:, a] - r4[:, b]) + np.abs(r4[:, c] - r4[:, e]) for (a, b), (c, e) in match])
prodm = np.column_stack([d4[:, a] * d4[:, b] + d4[:, c] * d4[:, e] for (a, b), (c, e) in match])
validm = np.column_stack([~(((pv[:, a] == pv[:, b]) & (pv[:, a] != "")) | ((pv[:, c] == pv[:, e]) & (pv[:, c] != ""))) for (a, b), (c, e) in match])
den = (d4**2).sum(axis=1)
smart2_rows = []
for period in ["2014", "2015Q1"]:
    sel = per4 == period
    act_c = 2 * prodm[sel, 0].sum() / den[sel].sum()
    act_a = absm[sel, 0].sum() / (2 * sel.sum())
    for null, valid in [("unrestricted", np.ones_like(validm[sel])), ("no rematch", validm[sel])]:
        nc = np.empty(B)
        na = np.empty(B)
        for i in range(B):
            pick = np.argmin(np.where(valid, rng.random(valid.shape), 2), axis=1)
            nc[i] = 2 * np.take_along_axis(prodm[sel], pick[:, None], 1).sum() / den[sel].sum()
            na[i] = np.take_along_axis(absm[sel], pick[:, None], 1).sum() / (2 * sel.sum())
        smart2_rows.append(
            dict(
                period=period,
                null=null,
                batches=int(sel.sum()),
                actual_is_rematch=int((~validm[sel, 0]).sum()),
                corr=act_c,
                corr_null_mean=nc.mean(),
                corr_p=(1 + (np.abs(nc - nc.mean()) >= abs(act_c - nc.mean())).sum()) / (B + 1),
                absdiff=act_a,
                absdiff_null_mean=na.mean(),
                absdiff_p=(1 + (np.abs(na - na.mean()) >= abs(act_a - na.mean())).sum()) / (B + 1),
            )
        )
smart2 = pl.DataFrame(smart2_rows)
with pl.Config(tbl_cols=-1, tbl_width_chars=250):
    print(smart2)
smart2.write_csv("outputs/tables/arena_random_pairing_smart2.csv")

# 執子顏色：lila 以配對中的第一位為白方，而配對前已打亂順序。若顏色隨機，白方減黑方的等級分差對稱於 0。
# 這個檢驗連一局一批的配對也適用，因為顏色在批內仍是隨機指派
color = (
    g.with_columns(x=(pl.col("white_elo") - pl.col("black_elo")).cast(pl.Float64))
    .group_by("period", "rule", "size")
    .agg(games=pl.len(), white_higher=(pl.col("x") > 0).mean(), mean_diff=pl.col("x").mean(), z=pl.col("x").sum() / (pl.col("x") ** 2).sum().sqrt())
    .sort("period", "rule", "size")
)
print(color)
color.write_csv("outputs/tables/arena_random_pairing_color.csv")

# 先前在整場錦標賽內打亂黑方的檢查：同一玩家在整場有多局，打亂後會出現自己對自己，實際配對不可能如此，
# 分差因而被低估。依打亂範圍由寬到窄比較，並列出排除自配後的虛無平均
g = g.with_columns(
    min1=pl.col("start").dt.truncate("1m"),
    min5=pl.col("start").dt.truncate("5m"),
)
scope_rows = []
for scope in ["tournament_id", "min5", "min1", "bid"]:
    cols = ["tournament_id", scope] if scope in ["min1", "min5"] else [scope]
    s = g.sort(cols).with_columns(sc=pl.struct(cols).rle_id())
    sc = s["sc"].to_numpy()
    wr = s["white_elo"].to_numpy().astype(float)
    br = s["black_elo"].to_numpy().astype(float)
    wid = s["w"].to_numpy()
    bidn = s["b"].to_numpy()
    cnt = np.bincount(sc)
    dw = wr - (np.bincount(sc, wr) / cnt)[sc]
    db = br - (np.bincount(sc, br) / cnt)[sc]
    sels = {p: (s["period"] == p).to_numpy() for p in ["2014", "2015Q1"]}
    res = {p: [] for p in sels}
    for i in range(200):
        o = np.argsort(sc + rng.random(len(sc)))
        same = wid == bidn[o]
        for p, sel in sels.items():
            res[p].append(
                (
                    (dw * db[o])[sel].sum() / np.sqrt((dw**2)[sel].sum() * (db**2)[sel].sum()),
                    np.abs(wr - br[o])[sel].mean(),
                    np.abs(wr - br[o])[sel & ~same].mean(),
                    same[sel].mean(),
                    (dw * db[o])[sel & ~same].sum() / np.sqrt((dw**2)[sel & ~same].sum() * (db[o] ** 2)[sel & ~same].sum()),
                )
            )
    for p, sel in sels.items():
        rp = np.array(res[p])
        scope_rows.append(
            dict(
                period=p,
                scope={"tournament_id": "tournament", "min5": "5 min", "min1": "1 min", "bid": "batch"}[scope],
                games=int(sel.sum()),
                corr=(dw * db)[sel].sum() / np.sqrt((dw**2)[sel].sum() * (db**2)[sel].sum()),
                corr_null_mean=rp[:, 0].mean(),
                corr_null_mean_no_self=rp[:, 4].mean(),
                absdiff=np.abs(wr - br)[sel].mean(),
                absdiff_null_mean=rp[:, 1].mean(),
                absdiff_null_mean_no_self=rp[:, 2].mean(),
                self_pair_share=rp[:, 3].mean(),
            )
        )
scope_tab = pl.DataFrame(scope_rows)
with pl.Config(tbl_cols=-1, tbl_width_chars=250):
    print(scope_tab)
scope_tab.write_csv("outputs/tables/arena_random_pairing_scope.csv")

fig, axes = plt.subplots(1, len(null_draws), figsize=(3.2 * len(null_draws), 3), sharey=True)
for ax, (nm, act, nc) in zip(axes, null_draws):
    ax.hist(nc, bins=40, color="0.7")
    ax.axvline(act, color="C3")
    ax.set_title(nm.replace("|", ", "), fontsize=9)
    ax.set_xlabel("within-batch rating correlation")
axes[0].set_ylabel("permutations")
fig.suptitle("Actual (red) vs within-batch re-pairing null, batches with 2+ games", fontsize=10)
fig.tight_layout()
fig.savefig("outputs/figures/arena_random_pairing_null.png", dpi=150)

bs = sizes.group_by("n_games", "first").agg(pl.col("games").sum()).sort("n_games")
fig, ax = plt.subplots(figsize=(5, 3))
for k, (first, lab) in enumerate([(True, "first batch of tournament"), (False, "later batches")]):
    x = bs.filter(pl.col("first") == first)
    ax.bar(x["n_games"].to_numpy() + (k - 0.5) * 0.4, x["games"].to_numpy(), width=0.4, label=lab)
ax.set_yscale("log")
ax.set_xticks(range(1, 11), [str(i) for i in range(1, 10)] + ["10+"])
ax.set_xlabel("games in batch (same tournament, same second)")
ax.set_ylabel("games")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("outputs/figures/arena_random_pairing_batch_size.png", dpi=150)
