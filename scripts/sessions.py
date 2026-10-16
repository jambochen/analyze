# 「續下」「離開」「回訪」的操作定義：以相鄰開局間隔的混合分布決定連續對局的
# 門檻，建立玩家局層級的反應變數，並檢查描述結論對門檻是否敏感。

import datetime as dt
import shutil
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pyarrow.parquet as pq

UTC = dt.timezone.utc
BEGIN = dt.datetime(2014, 1, 1, tzinfo=UTC)
# 月檔依巴黎時間切月，2015-06 月檔止於 UTC 22:00，之後的對局在未下載的 2015-07 月檔
END = dt.datetime(2015, 6, 30, 22, tzinfo=UTC)
# 混合分布只用之後 30 天完整可見的對局，避免期末截斷壓低長間隔的密度
FIT_END = END - dt.timedelta(days=30)
# 敏感度的門檻最長 2 小時，之後 1 天完整可見即可
SENS_END = END - dt.timedelta(days=1)
H8 = 8 * 3600
D14 = 14 * 86400
NB = 16
CLASSES = ["Bullet", "Blitz", "Classical"]
EDGES = np.arange(0, 6.5001, 0.02)
MID = (EDGES[1:] + EDGES[:-1]) / 2
tmp = tempfile.mkdtemp()

raw = pl.scan_parquet("data/processed/lichess_games.parquet")
# 無鐘對局在 2014 年 12 月以前計入 Classical 的等級分，但與通信棋一樣沒有時限
no_clock = pl.col("base_seconds").is_null()
sample = raw.select(
    games=pl.len(),
    correspondence=(pl.col("speed") == "Correspondence").sum(),
    no_clock_classical=(no_clock & (pl.col("speed") == "Classical")).sum(),
    self_games=((pl.col("white_id") == pl.col("black_id")) & ~no_clock).sum(),
).collect()

# 無鐘的對局一局可跨數日，開局時間不代表玩家當下在線；同帳號自己對自己的局無法拆成兩名玩家
games = raw.filter(~no_clock, pl.col("white_id") != pl.col("black_id"))

# 全站在 10 分鐘以上沒有任何新局，視為伺服器停擺：此時的「沒有下一局」不是玩家的選擇
outages = (
    games.select("start").sort("start")
    .with_columns(outage_start=pl.col("start").shift(1), outage_seconds=(pl.col("start") - pl.col("start").shift(1)).dt.total_seconds())
    .filter(pl.col("outage_seconds") > 600)
    .select("outage_start", "outage_seconds")
    .collect()
)
outages.write_csv("outputs/tables/sessions_outages.csv")

tours = (
    games.filter(pl.col("tournament_id").is_not_null())
    .group_by("tournament_id")
    .agg(tour_first=pl.col("start").min(), tour_last=pl.col("start").max(), tour_games=pl.len(), tour_players=pl.concat_list("white_id", "black_id").explode().n_unique())
    .collect()
    .with_columns(tour_span_minutes=(pl.col("tour_last") - pl.col("tour_first")).dt.total_seconds() / 60)
)
(
    tours.filter(pl.col("tour_first") >= BEGIN, pl.col("tour_first") < END)
    .group_by(span_minutes=(pl.col("tour_span_minutes") // 5 * 5).cast(pl.Int32))
    .agg(tournaments=pl.len(), median_players=pl.col("tour_players").median())
    .sort("span_minutes")
    .write_csv("outputs/tables/sessions_tournament_span.csv")
)

cols = ["game_id", "start", "speed", "time_control", "base_seconds", "increment_seconds", "tournament_id"]
score_white = pl.col("result").replace_strict({"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}, default=None)
long = pl.concat([
    games.select(*cols, player=pl.col("white_id"), opponent=pl.col("black_id"), color=pl.lit("white"),
                 elo=pl.col("white_elo"), opp_elo=pl.col("black_elo"), score=score_white),
    games.select(*cols, player=pl.col("black_id"), opponent=pl.col("white_id"), color=pl.lit("black"),
                 elo=pl.col("black_elo"), opp_elo=pl.col("white_elo"), score=1 - score_white),
])

# 依玩家代碼分桶：每位玩家的完整序列落在同一桶，視窗運算的分組數隨之縮小
hist = {}
for b in range(NB):
    d = (
        long.filter(pl.col("player") % NB == b)
        .collect(engine="streaming")
        .sort("player", "start", "game_id")
        .with_columns(
            # 以每方 40 步、時鐘用盡估計一局的名目長度
            est_duration=2 * (pl.col("base_seconds") + 40 * pl.col("increment_seconds")),
            prev_gap=(pl.col("start") - pl.col("start").shift(1)).over("player").dt.total_seconds(),
            next_gap=(pl.col("start").shift(-1) - pl.col("start")).over("player").dt.total_seconds(),
            next_class=pl.col("speed").shift(-1).over("player"),
            player_game_no=pl.int_range(1, pl.len() + 1).over("player"),
            next_in_tour=pl.col("start").shift(-1).over("player", "tournament_id").is_not_null(),
        )
        .sort("start")
        .join_asof(outages.select(next_outage=pl.col("outage_start")), left_on="start", right_on="next_outage", strategy="forward")
        .with_columns(to_outage=(pl.col("next_outage") - pl.col("start")).dt.total_seconds())
        .drop("next_outage")
        .sort("player", "start", "game_id")
    )
    d.write_parquet(f"{tmp}/{b}.parquet")
    f = d.filter(pl.col("start") >= BEGIN, pl.col("start") < FIT_END, pl.col("next_gap") <= 30 * 86400)
    for tc in CLASSES:
        for arena in [False, True]:
            x = f.filter(pl.col("speed") == tc, pl.col("tournament_id").is_not_null() == arena)
            # 同秒開局的間隔記為 1 秒，才能取對數
            g = np.log10(np.maximum(x["next_gap"].to_numpy(), 1))
            r = np.log10(np.maximum(x["next_gap"].to_numpy(), 1) / x["est_duration"].to_numpy())
            hist[tc, arena, "gap"] = hist.get((tc, arena, "gap"), 0) + np.histogram(g, EDGES)[0]
            hist[tc, arena, "ratio"] = hist.get((tc, arena, "ratio"), 0) + np.histogram(r, EDGES - 3)[0]
    print(b, d.height)
    del d, f


# 兩成分 Gaussian mixture 在直方圖上的 EM；分界為第一成分後驗機率降到 0.5 之處
def em(h, mid):
    mu = np.array([np.quantile(np.repeat(mid, (h / h.sum() * 1e5).astype(int)), q) for q in [0.25, 0.75]])
    sd = np.array([0.5, 0.5])
    w = np.array([0.5, 0.5])
    for _ in range(1000):
        p = w * np.exp(-0.5 * ((mid[:, None] - mu) / sd) ** 2) / sd
        r = p / p.sum(1, keepdims=True) * h[:, None]
        n = r.sum(0)
        w = n / n.sum()
        mu = (r * mid[:, None]).sum(0) / n
        sd = np.sqrt((r * (mid[:, None] - mu) ** 2).sum(0) / n)
    o = np.argsort(mu)
    mu, sd, w = mu[o], sd[o], w[o]
    p = w * np.exp(-0.5 * ((mid[:, None] - mu) / sd) ** 2) / sd
    cross = mid[np.argmax((p[:, 0] / p.sum(1) < 0.5) & (mid > mu[0]))]
    return mu, sd, w, cross


fits = []
for tc in CLASSES:
    for source in ["pooled", "lobby", "arena"]:
        for var in ["gap", "ratio"]:
            h = {"pooled": hist[tc, False, var] + hist[tc, True, var], "lobby": hist[tc, False, var], "arena": hist[tc, True, var]}[source].astype(float)
            mid = MID if var == "gap" else MID - 3
            # 擬合範圍的上限：30 天含日週期的峰，1 天與 6 小時逐步排除
            for upper in [30 * 86400, 86400, 6 * 3600]:
                hi = np.log10(upper) if var == "gap" else np.inf
                if var == "ratio" and upper != 30 * 86400:
                    continue
                keep = mid <= hi
                mu, sd, w, cross = em(h[keep], mid[keep])
                # 低谷：第一個峰之後、日週期峰之前，平滑後對數密度的最低點
                sm = np.convolve(h, np.ones(9) / 9, mode="same")
                peak = np.argmax(np.where(mid < mid[np.argmax(h)] + 1, sm, 0))
                lo_range = (mid > mid[peak]) & (mid < (np.log10(86400) if var == "gap" else mid[peak] + 2.5))
                valley = mid[lo_range][np.argmin(sm[lo_range])]
                fits.append({
                    "speed": tc, "source": source, "variable": var, "fit_upper_seconds": upper if var == "gap" else None,
                    "n": int(h.sum()), "mu1": mu[0], "sd1": sd[0], "w1": w[0], "mu2": mu[1], "sd2": sd[1], "w2": w[1],
                    "crossing_log10": cross, "crossing": 10**cross, "valley_log10": valley, "valley": 10**valley,
                })
fits = pl.DataFrame(fits)
fits.write_csv("outputs/tables/sessions_threshold_fit.csv")
print(fits)

# 主門檻：各速度類別合併 arena 與大廳、擬合範圍最寬的混合分布分界，取整到分鐘
main = fits.filter(pl.col("source") == "pooled", pl.col("variable") == "gap", pl.col("fit_upper_seconds") == 30 * 86400)
T = {r["speed"]: round(r["crossing"] / 60) * 60 for r in main.iter_rows(named=True)}
ratio_main = fits.filter(pl.col("source") == "pooled", pl.col("variable") == "ratio")
K = {r["speed"]: round(r["crossing"] * 4) / 4 for r in ratio_main.iter_rows(named=True)}
pl.DataFrame({"speed": CLASSES, "threshold_seconds": [T[c] for c in CLASSES], "ratio_threshold": [K[c] for c in CLASSES]}).write_csv("outputs/tables/sessions_threshold.csv")
print(T, K)

# 敏感度比較的規則：門檻一律依前一局的速度類別，因為開局間隔包含前一局本身的長度
RULES = {
    "main": pl.col("speed").replace_strict(T, return_dtype=pl.Float64),
    "main_half": pl.col("speed").replace_strict(T, return_dtype=pl.Float64) / 2,
    "main_double": pl.col("speed").replace_strict(T, return_dtype=pl.Float64) * 2,
    # arXiv 2503.21713 未給 Classical 的門檻，以主門檻代入
    "literature": pl.col("speed").replace_strict({"Bullet": 300.0, "Blitz": 420.0, "Classical": float(T["Classical"])}),
    "ratio": pl.col("speed").replace_strict(K, return_dtype=pl.Float64) * pl.col("est_duration"),
}
GRID_MIN = [1, 2, 3, 5, 7, 10, 15, 20, 30, 45, 60, 90, 120]
GRID_RATIO = [1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

OUTCOMES = ["continue_session", "leave_8h", "churn_14d", "tour_left"]
grid, seq, cross_class, rates, months, window, tour_margin = [], [], [], [], [], [], []
for b in range(NB):
    d = pl.read_parquet(f"{tmp}/{b}.parquet").with_columns(
        arena=pl.col("tournament_id").is_not_null(),
        result=pl.col("score").replace_strict({1.0: "win", 0.5: "draw", 0.0: "loss"}, default=None, return_dtype=pl.String),
    )
    s = d.filter(pl.col("start") >= BEGIN, pl.col("start") < SENS_END)
    for m in GRID_MIN:
        grid.append(s.group_by("speed", "arena", "result").agg(
            rule=pl.lit("fixed"), threshold=pl.lit(float(m)), n=pl.len(), cont=(pl.col("next_gap") <= m * 60).sum()))
    for k in GRID_RATIO:
        grid.append(s.group_by("speed", "arena", "result").agg(
            rule=pl.lit("ratio"), threshold=pl.lit(k), n=pl.len(), cont=(pl.col("next_gap") <= k * pl.col("est_duration")).sum()))
    for name, thr in RULES.items():
        # 局序須從 2013 年起累計，2014 年初的連續對局才不會被切斷
        x = d.with_columns(cont=(pl.col("next_gap") <= thr).fill_null(False)).with_columns(
            new=pl.col("cont").shift(1, fill_value=False).over("player").not_(),
            idx=pl.int_range(pl.len()).over("player"),
        ).with_columns(
            game_seq=pl.col("idx") - pl.when(pl.col("new")).then(pl.col("idx")).forward_fill().over("player") + 1,
        ).filter(pl.col("start") >= BEGIN, pl.col("start") < SENS_END)
        seq.append(x.group_by("speed", "arena", game_seq=pl.col("game_seq").clip(upper_bound=20)).agg(
            rule=pl.lit(name), n=pl.len(), cont=pl.col("cont").sum()))
        cross_class.append(x.filter(pl.col("cont")).group_by("speed").agg(
            rule=pl.lit(name), n=pl.len(), switch=(pl.col("next_class") != pl.col("speed")).sum()))

    thr = RULES["main"]
    d = (
        d.with_columns(cont=(pl.col("next_gap") <= thr).fill_null(False))
        .with_columns(
            new=pl.col("cont").shift(1, fill_value=False).over("player").not_(),
            idx=pl.int_range(pl.len()).over("player"),
        )
        .with_columns(
            session_no=pl.col("new").cum_sum().over("player"),
            game_seq=pl.col("idx") - pl.when(pl.col("new")).then(pl.col("idx")).forward_fill().over("player") + 1,
        )
        .with_columns(session_games=pl.col("game_seq").max().over("player", "session_no"))
        .join(tours.select("tournament_id", "tour_first", "tour_last"), on="tournament_id", how="left")
        .with_columns(
            # 時間窗超出資料終點者，無論是否已觀察到下一局都記為缺值，使各期的定義一致
            continue_session=pl.when(pl.col("start") + pl.duration(seconds=thr.cast(pl.Int64)) <= END).then(pl.col("cont")),
            leave_8h=pl.when(pl.col("start") <= END - dt.timedelta(seconds=H8)).then(pl.col("next_gap").is_null() | (pl.col("next_gap") > H8)),
            churn_14d=pl.when(pl.col("start") <= END - dt.timedelta(seconds=D14)).then(pl.col("next_gap").is_null() | (pl.col("next_gap") > D14)),
            tour_last_game=pl.when(pl.col("arena")).then(pl.col("next_in_tour").not_()),
            # 風險集只取本局名目長度之後錦標賽仍有新局開出者，不依是否續下而定；錦標賽最長約 2 小時，終點前 2 小時內結束的場次可能被截斷
            tour_left=pl.when(
                pl.col("arena")
                & ((pl.col("tour_last") - pl.col("start")).dt.total_seconds() >= pl.col("est_duration"))
                & (pl.col("tour_last") < END - dt.timedelta(hours=2))
            ).then(pl.col("next_in_tour").not_()),
            return_seconds=pl.col("next_gap").fill_null((END - pl.col("start")).dt.total_seconds()),
            return_censored=pl.col("next_gap").is_null(),
            session_threshold=thr,
        )
        .filter(pl.col("start") >= BEGIN, pl.col("start") < END)
        .select(
            "game_id", "player", "opponent", "color", "start", "speed", "time_control", "base_seconds", "increment_seconds",
            "est_duration", "tournament_id", "arena", "elo", "opp_elo", "score", "result",
            "player_game_no", "session_no", "game_seq", "session_games", "session_threshold", "prev_gap",
            "continue_session", "leave_8h", "churn_14d", "tour_last_game", "tour_left", "tour_first", "tour_last",
            "return_seconds", "return_censored", "to_outage",
        )
    )
    # 摘要在每桶內先聚合成和與個數：在合併後的 5,000 萬列上做 group_by 會超出記憶體上限
    agg = {"n": pl.len(), **{f"{o}_n": pl.col(o).count() for o in OUTCOMES}, **{f"{o}_sum": pl.col(o).sum() for o in OUTCOMES}}
    rates.append(d.group_by("speed", "arena").agg(result=pl.lit("all"), **agg))
    rates.append(d.filter(pl.col("result").is_not_null()).group_by("speed", "arena", "result").agg(**agg))
    months.append(d.group_by(month=pl.col("start").dt.strftime("%Y-%m"), arena=pl.col("arena")).agg(**agg))
    for m in [0.5, 1.0, 2.0, 3.0]:
        tour_margin.append(d.filter(
            pl.col("arena"), pl.col("tour_last") < END - dt.timedelta(hours=2),
            (pl.col("tour_last") - pl.col("start")).dt.total_seconds() >= m * pl.col("est_duration"),
        ).group_by("speed", "result").agg(margin=pl.lit(m), n=pl.len(), left=pl.col("tour_last_game").sum()))
    window.append(d.select(
        **{f"{o}_n": pl.col(o).count() for o in OUTCOMES},
        **{f"{o}_last_start": pl.col("start").filter(pl.col(o).is_not_null()).max() for o in OUTCOMES},
        n=pl.len(), players=pl.col("player").n_unique(), censored=pl.col("return_censored").sum(),
        outage_in_session_window=(pl.col("to_outage") <= pl.col("session_threshold")).sum(),
        outage_in_8h=(pl.col("to_outage") <= H8).sum(),
        sessions=(pl.col("game_seq") == 1).sum(),
    ))
    d.write_parquet(f"{tmp}/final_{b}.parquet")
    print(b, d.height)
    del d, s, x

# 以 sink_parquet 串流合併 16 個暫存檔在 8 GB 上限內仍被殺掉，改為逐檔讀入、逐檔附加寫出
writer = None
for b in range(NB):
    t = pl.read_parquet(f"{tmp}/final_{b}.parquet").to_arrow()
    writer = writer or pq.ParquetWriter("data/processed/player_games.parquet", t.schema, compression="zstd")
    writer.write_table(t, row_group_size=500_000)
    del t
writer.close()
shutil.rmtree(tmp)

grid = pl.concat(grid).group_by("rule", "threshold", "speed", "arena", "result").agg(pl.col("n", "cont").sum()).with_columns(rate=pl.col("cont") / pl.col("n")).sort("rule", "threshold", "speed", "arena", "result")
grid.write_csv("outputs/tables/sessions_sensitivity.csv")
seq = pl.concat(seq).group_by("rule", "speed", "arena", "game_seq").agg(pl.col("n", "cont").sum()).with_columns(rate=pl.col("cont") / pl.col("n")).sort("rule", "speed", "arena", "game_seq")
seq.write_csv("outputs/tables/sessions_sensitivity_seq.csv")
cross_class = pl.concat(cross_class).group_by("rule", "speed").agg(pl.col("n", "switch").sum()).with_columns(share=pl.col("switch") / pl.col("n")).sort("rule", "speed")
cross_class.write_csv("outputs/tables/sessions_cross_class.csv")
print(cross_class)

rate_cols = [pl.col(f"{o}_sum") / pl.col(f"{o}_n") for o in OUTCOMES]
rate_names = {f"{o}_sum": o for o in OUTCOMES}
rates = (
    pl.concat(rates).group_by("speed", "arena", "result").agg(pl.exclude("speed", "arena", "result").sum())
    .with_columns(rate_cols).rename(rate_names).sort("speed", "arena", "result")
)
rates.write_csv("outputs/tables/sessions_rates.csv")
print(rates)
(
    pl.concat(months).group_by("month", "arena").agg(pl.exclude("month", "arena").sum())
    .with_columns(rate_cols).rename(rate_names).sort("month", "arena")
    .write_csv("outputs/tables/sessions_rates_month.csv")
)
(
    pl.concat(tour_margin).group_by("margin", "speed", "result").agg(pl.col("n", "left").sum())
    .with_columns(rate=pl.col("left") / pl.col("n")).sort("margin", "speed", "result")
    .write_csv("outputs/tables/sessions_tour_margin.csv")
)
# 各桶的玩家互不重疊，玩家數可直接相加
window = pl.concat(window).select(
    pl.col("^.*_last_start$").max(), pl.exclude("^.*_last_start$").sum(),
)
pl.concat([sample, window], how="horizontal").write_csv("outputs/tables/sessions_sample.csv")
print(pl.concat([sample, window], how="horizontal").transpose(include_header=True))

fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for j, tc in enumerate(CLASSES):
    for i, var in enumerate(["gap", "ratio"]):
        ax = axes[i, j]
        mid = MID if var == "gap" else MID - 3
        h = (hist[tc, False, var] + hist[tc, True, var]).astype(float)
        dens = h / h.sum() / 0.02
        ax.plot(mid, dens, color="black", lw=1, label="observed")
        r = fits.filter(pl.col("speed") == tc, pl.col("source") == "pooled", pl.col("variable") == var, pl.col("fit_upper_seconds").is_null() | (pl.col("fit_upper_seconds") == 30 * 86400)).row(0, named=True)
        for k, c in [("1", "tab:blue"), ("2", "tab:orange")]:
            ax.plot(mid, r[f"w{k}"] * np.exp(-0.5 * ((mid - r[f"mu{k}"]) / r[f"sd{k}"]) ** 2) / (r[f"sd{k}"] * np.sqrt(2 * np.pi)), color=c, lw=1, ls="--", label=f"component {k}")
        ax.axvline(r["crossing_log10"], color="tab:red", lw=1, label="mixture boundary")
        ax.axvline(r["valley_log10"], color="tab:green", lw=1, ls=":", label="density valley")
        if var == "gap" and tc != "Classical":
            ax.axvline(np.log10({"Bullet": 300, "Blitz": 420}[tc]), color="gray", lw=1, ls="-.", label="arXiv 2503.21713")
        if var == "gap":
            ax.set_xticks(np.log10([10, 60, 600, 3600, 6 * 3600, 86400, 7 * 86400]), ["10s", "1m", "10m", "1h", "6h", "1d", "7d"])
            ax.set_xlim(0.5, 6.5)
            ax.set_xlabel("gap to next game start (log scale)")
        else:
            ax.set_xticks(np.log10([0.1, 0.5, 1, 2, 5, 10, 100, 1000]), ["0.1", "0.5", "1", "2", "5", "10", "100", "1000"])
            ax.set_xlim(-1.5, 3.5)
            ax.set_xlabel("gap / nominal game duration (log scale)")
        ax.set_title(tc)
        ax.set_ylabel("density per log10 unit")
axes[0, 0].legend(fontsize=7)
fig.tight_layout()
fig.savefig("outputs/figures/sessions_gap.png", dpi=150)

fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
g = grid.filter(pl.col("rule") == "fixed").group_by("threshold", "speed", "arena", "result").agg(pl.col("n", "cont").sum()).with_columns(rate=pl.col("cont") / pl.col("n"))
for j, tc in enumerate(CLASSES):
    ax = axes[j]
    for res, c in [("loss", "tab:red"), ("draw", "tab:gray"), ("win", "tab:blue")]:
        for arena, ls in [(False, "-"), (True, "--")]:
            x = g.filter(pl.col("speed") == tc, pl.col("result") == res, pl.col("arena") == arena).sort("threshold")
            ax.plot(x["threshold"], x["rate"], color=c, ls=ls, marker=".", label=f"{res}, {'arena' if arena else 'lobby'}")
    ax.axvline(T[tc] / 60, color="black", lw=1, label="main threshold")
    ax.set_xscale("log")
    ax.set_xticks(GRID_MIN[::2], [str(m) for m in GRID_MIN[::2]])
    ax.set_xlabel("session threshold (minutes)")
    ax.set_title(tc)
axes[0].set_ylabel("share of games followed by another game")
axes[0].legend(fontsize=7)
fig.tight_layout()
fig.savefig("outputs/figures/sessions_sensitivity.png", dpi=150)

fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
sq = seq.filter(pl.col("game_seq") <= 15)
for i, arena in enumerate([False, True]):
    for j, tc in enumerate(CLASSES):
        ax = axes[i, j]
        for name in RULES:
            x = sq.filter(pl.col("rule") == name, pl.col("speed") == tc, pl.col("arena") == arena).sort("game_seq")
            ax.plot(x["game_seq"], x["rate"], marker=".", label=name)
        ax.set_title(f"{tc}, {'arena' if arena else 'lobby'}")
        ax.set_xlabel("position in session")
    axes[i, 0].set_ylabel("continuation rate")
axes[0, 0].legend(fontsize=7)
fig.tight_layout()
fig.savefig("outputs/figures/sessions_seq.png", dpi=150)

fig, ax = plt.subplots(figsize=(7, 4))
x = tours.filter(pl.col("tour_first") >= BEGIN, pl.col("tour_first") < END)["tour_span_minutes"].to_numpy()
ax.hist(x, bins=np.arange(0, 181, 1), color="gray")
ax.set_xlabel("first to last game start within tournament (minutes)")
ax.set_ylabel("tournaments")
fig.tight_layout()
fig.savefig("outputs/figures/sessions_tournament_span.png", dpi=150)
