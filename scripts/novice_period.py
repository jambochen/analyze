# 新手期的界定：以重放的 RD 界定 provisional，與累計局數的代理比較，並看校準斜率在門檻兩側是否不同。
# 樣本為 2014 年 Bullet、Blitz、Classical 的有鐘對局；2013 年起的對局只用來算累計局數與首局。

import json
import os
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, "scripts")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import likelihood as lk
import novice
import ordinal

UTC = "UTC"
CLASSES = ["Bullet", "Blitz", "Classical"]
RD_MAIN, RD_ALT, N_EST = novice.RD, novice.RD_ALT, novice.GAMES
ELO = ordinal.Q

g = (
    pl.scan_parquet("data/processed/lichess_games.parquet")
    .filter(pl.col("speed").is_in(CLASSES), pl.col("base_seconds").is_not_null(), pl.col("start") < pl.datetime(2015, 1, 1, time_zone=UTC))
    .select(
        "game_id", "start", "speed", "white_elo", "black_elo",
        arena=pl.col("tournament_id").is_not_null(),
        y=pl.col("result").replace_strict({"0-1": 0, "1/2-1/2": 1, "1-0": 2}, default=None, return_dtype=pl.Int8),
        w="white_id", b="black_id",
    )
    .with_row_index("i")
    .collect(engine="streaming")
)
# 累計局數的代理照 calibration.py 的定義：含 "*" 與缺等級分的對局，同一速度類別，自 2013-01 起算
nprev = pl.concat([
    pl.concat([g.filter(pl.col("speed") == tc).select("i", "start", p="w"), g.filter(pl.col("speed") == tc).select("i", "start", p="b")])
    .sort("i")
    .with_columns(n_prev=pl.int_range(pl.len(), dtype=pl.UInt32).over("p"))
    .filter(pl.col("start") >= pl.datetime(2014, 1, 1, time_zone=UTC))
    .select("i", "p", "n_prev")
    for tc in CLASSES
])
rp = pl.scan_parquet("data/processed/glicko_replay.parquet").select(
    "game_id", "perf", "white_rd", "black_rd", "white_left_truncated", "black_left_truncated", "white_prior_games", "black_prior_games",
)
d = (
    g.lazy()
    .filter(
        pl.col("start") >= pl.datetime(2014, 1, 1, time_zone=UTC),
        pl.col("y").is_not_null(),
        pl.col("white_elo").is_not_null(),
        pl.col("black_elo").is_not_null(),
    )
    .join(nprev.lazy().rename({"p": "w", "n_prev": "nw"}), on=["i", "w"])
    .join(nprev.lazy().rename({"p": "b", "n_prev": "nb"}), on=["i", "b"])
    .join(rp, on="game_id", how="left")
    .with_columns(
        score=pl.col("y") / 2,
        L=(pl.col("white_elo") - pl.col("black_elo")) * ELO,
        n_min=pl.min_horizontal("nw", "nb"),
        rd_max=pl.max_horizontal("white_rd", "black_rd"),
        source=pl.when(pl.col("arena")).then(pl.lit("arena")).otherwise(pl.lit("lobby")),
    )
    .collect(engine="streaming")
)
del nprev
print(d.select(n=pl.len(), no_replay=pl.col("perf").is_null().sum(), perf_mismatch=(pl.col("perf") != pl.col("speed")).sum()))

# 每一方各一列：RD 隨此前局數下降的軌跡。左截斷帳號的起始 RD 是設定值，分開看
side = pl.concat([
    d.select("speed", "source", rd=f"{c}_rd", lt=f"{c}_left_truncated", k=f"{c}_prior_games", n=x)
    for c, x in [("white", "nw"), ("black", "nb")]
])
traj = (
    side.with_columns(k=pl.col("k").clip(0, 100))
    .group_by("speed", "lt", "k")
    .agg(
        n=pl.len(),
        rd_p10=pl.col("rd").quantile(0.1), rd_med=pl.col("rd").median(), rd_p90=pl.col("rd").quantile(0.9),
        prov_main=(pl.col("rd") >= RD_MAIN).mean(), prov_alt=(pl.col("rd") >= RD_ALT).mean(),
    )
    .sort("speed", "lt", "k")
)
traj.write_csv("outputs/tables/novice_period_rd_by_games.csv")
print(traj.filter(~pl.col("lt"), pl.col("k").is_in([0, 5, 10, 15, 20, 30, 50, 100])))

# 兩種界定在玩家局上的一致程度：RD 門檻對累計局數門檻
def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return po, (po - pe) / (1 - pe)


agree_rows = []
for tc in CLASSES:
    s = side.filter(pl.col("speed") == tc)
    for nm, rd in [("rd110", RD_MAIN), ("rd100", RD_ALT)]:
        a = (s["rd"] >= rd).to_numpy()
        for n_est in [10, 20, N_EST, 50]:
            b = (s["n"] < n_est).to_numpy()
            po, k = kappa(a, b)
            agree_rows.append({"speed": tc, "rd_def": nm, "games_threshold": n_est, "n": s.height, "share_rd_novice": a.mean(), "share_games_novice": b.mean(), "agreement": po, "kappa": k})
agree = pl.DataFrame(agree_rows)
agree.write_csv("outputs/tables/novice_period_agreement.csv")
print(agree)
del side

# 各界定下新手對局（任一方為新手）所占的比例
d = d.with_columns(
    nov_main=pl.col("rd_max") >= RD_MAIN,
    nov_alt=pl.col("rd_max") >= RD_ALT,
    nov_games=pl.col("n_min") < N_EST,
    lt_any=pl.col("white_left_truncated") | pl.col("black_left_truncated"),
)
share = (
    d.group_by("speed", "source")
    .agg(n=pl.len(), nov_main=pl.col("nov_main").mean(), nov_alt=pl.col("nov_alt").mean(), nov_games=pl.col("nov_games").mean(), left_truncated=pl.col("lt_any").mean())
    .sort("speed", "source")
)
share.write_csv("outputs/tables/novice_period_share.csv")
print(share)


# 分層的 Cox 校準：logit E[S] = a + b logit(E)，fractional logit 以 Newton 法估計。
# 推論照 calibration.py 用 dyadic cluster：任兩局只要有共同玩家就相關
def cox(m):
    X = np.stack([np.ones(m.height), m["L"].to_numpy()], 1)
    s = m["score"].to_numpy()
    t = np.array([0.0, 1.0])
    for _ in range(30):
        p = 1 / (1 + np.exp(-X @ t))
        H = X.T @ (X * (p * (1 - p))[:, None])
        step = np.linalg.solve(H, X.T @ (s - p))
        t += step
        if np.abs(step).max() < 1e-10:
            break
    p = 1 / (1 + np.exp(-X @ t))
    Hi = np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]))
    V = Hi @ lk.dyad_meat(X * (s - p)[:, None], m["w"].to_numpy(), m["b"].to_numpy()) @ Hi
    return t, V


RD_BINS = [60, 70, 80, 90, 100, 110, 130, 160, 200, 260]
GAME_BINS = [0, 2, 5, 10, 15, 20, 30, 50, 100, 200]
# 左截斷帳號的 RD 從設定值起算，不反映經驗，RD 分層排除含左截斷一方的對局
strata = {
    "rd_max": (pl.col("rd_max").cut(RD_BINS, left_closed=True).cast(pl.String), ~pl.col("lt_any")),
    "n_min": (pl.col("n_min").cut(GAME_BINS, left_closed=False).cast(pl.String), pl.lit(True)),
    "def_rd110": (pl.when(pl.col("nov_main")).then(pl.lit("novice")).otherwise(pl.lit("established")), pl.lit(True)),
    "def_rd100": (pl.when(pl.col("nov_alt")).then(pl.lit("novice")).otherwise(pl.lit("established")), pl.lit(True)),
    "def_games30": (pl.when(pl.col("nov_games")).then(pl.lit("novice")).otherwise(pl.lit("established")), pl.lit(True)),
    # 兩種界定不一致的對局：RD 說已穩定但局數未滿，或反之
    "rd110_vs_games30": (pl.concat_str(pl.when(pl.col("nov_main")).then(pl.lit("rd_novice")).otherwise(pl.lit("rd_established")), pl.when(pl.col("nov_games")).then(pl.lit("games_novice")).otherwise(pl.lit("games_established")), separator="|"), pl.lit(True)),
}
rows = []
for tc in CLASSES:
    dt = d.filter(pl.col("speed") == tc)
    for k, (lab, keep) in strata.items():
        dk = dt.filter(keep).with_columns(stratum=lab)
        for v in dk["stratum"].drop_nulls().unique().sort().to_list():
            m = dk.filter(pl.col("stratum") == v)
            if m.height < 2000:
                continue
            t, V = cox(m)
            rows.append({"speed": tc, "stratum_type": k, "stratum": v, "n": m.height, "rd_max_median": m["rd_max"].median(), "n_min_median": m["n_min"].median(), "intercept": t[0], "intercept_se": np.sqrt(V[0, 0]), "slope": t[1], "slope_se": np.sqrt(V[1, 1])})
    print(tc, "done")
cx = pl.DataFrame(rows)
cx.write_csv("outputs/tables/novice_period_cox.csv")
print(cx)

# 累計局數代理的另一個誤差來源：資料始於 2013-01，之前已有對局的帳號被當成新帳號。
# 2014 年 arena 玩家有 API 的帳號建立時間，可檢查左截斷旗標與「建立於資料開始之前」是否一致
users = pl.DataFrame([json.loads(l) for l in open("data/raw/lichess-users/users.ndjson")], infer_schema_length=None).select("id", "createdAt")
users = users.filter(pl.col("createdAt").is_not_null()).select("id", created=pl.from_epoch("createdAt", time_unit="ms").dt.replace_time_zone(UTC))
names = (
    pl.scan_parquet("data/processed/lichess_games.parquet")
    .filter(pl.col("tournament_id").is_not_null(), pl.col("start").dt.year() == 2014)
    .select(pl.concat_list(pl.struct(p="white_id", name="white"), pl.struct(p="black_id", name="black")).alias("x"))
    .explode("x").unnest("x").unique("p")
    .collect()
    .with_columns(id=pl.col("name").str.to_lowercase())
    .join(users, on="id")
    .select("p", "created")
)
# 每個帳號在各速度類別的首局，以及該局的左截斷旗標
first = (
    pl.concat([
        pl.scan_parquet("data/processed/lichess_games.parquet").select("game_id", "start", "speed", p=pl.col(f"{c}_id"))
        .join(pl.scan_parquet("data/processed/glicko_replay.parquet").select("game_id", lt=f"{c}_left_truncated"), on="game_id")
        for c in ["white", "black"]
    ])
    .filter(pl.col("speed").is_in(CLASSES))
    .join(names.lazy(), on="p")
    .sort("start", "game_id")
    .group_by("p", "speed")
    .agg(pl.col("start").first(), pl.col("lt").first(), pl.col("created").first())
    .collect(engine="streaming")
)
created_check = (
    first.with_columns(
        created_before_data=pl.col("created") < pl.datetime(2013, 1, 1, time_zone=UTC),
        first_in_2013_01=pl.col("start") < pl.datetime(2013, 2, 1, time_zone=UTC),
        days_created_to_first=(pl.col("start") - pl.col("created")).dt.total_hours() / 24,
    )
    .group_by("speed", "created_before_data", "lt")
    .agg(n=pl.len(), first_in_2013_01=pl.col("first_in_2013_01").mean(), days_created_to_first_med=pl.col("days_created_to_first").median())
    .sort(pl.all())
)
created_check.write_csv("outputs/tables/novice_period_created.csv")
print(created_check)

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for ax, tc in zip(axes, CLASSES):
    t = traj.filter(pl.col("speed") == tc, ~pl.col("lt"), pl.col("k") <= 60)
    ax.fill_between(t["k"], t["rd_p10"], t["rd_p90"], alpha=0.3, color=colors[0], label="10th-90th percentile")
    ax.plot(t["k"], t["rd_med"], color=colors[0], label="median")
    ax.axhline(RD_MAIN, color="gray", ls="--", lw=0.8, label=f"RD = {RD_MAIN}")
    ax.set_title(tc)
    ax.set_xlabel("prior rated games in the same speed (replay)")
axes[0].set_ylabel("pre-game RD (replay)")
axes[0].legend(fontsize=8)
fig.suptitle("Replayed Glicko-2 RD by number of prior games (accounts starting at 1500 within the data)")
fig.tight_layout()
fig.savefig("outputs/figures/novice_period_rd_by_games.png", dpi=150)

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for j, tc in enumerate(CLASSES):
    c = cx.filter(pl.col("speed") == tc, pl.col("stratum_type") == "rd_max").sort("rd_max_median")
    axes[0].errorbar(c["rd_max_median"] + (j - 1) * 1.5, c["slope"], yerr=1.96 * c["slope_se"], marker="o", ms=4, capsize=2, label=tc, color=colors[j])
    c = cx.filter(pl.col("speed") == tc, pl.col("stratum_type") == "n_min").sort("n_min_median")
    axes[1].errorbar(c["n_min_median"] + 1, c["slope"], yerr=1.96 * c["slope_se"], marker="o", ms=4, capsize=2, label=tc, color=colors[j])
axes[0].axvline(RD_MAIN, color="gray", ls="--", lw=0.8)
axes[0].set_xlabel("max pre-game RD of the two players (bin median)")
axes[1].axvline(N_EST + 1, color="gray", ls="--", lw=0.8)
axes[1].set_xscale("log")
axes[1].set_xlabel("min prior games of the two players + 1 (bin median, log scale)")
axes[0].set_ylabel("Cox calibration slope of Elo E")
axes[0].legend(fontsize=8)
fig.suptitle("Calibration slope by RD and by prior games (dyadic 95% CI)")
fig.tight_layout()
fig.savefig("outputs/figures/novice_period_cox.png", dpi=150)
