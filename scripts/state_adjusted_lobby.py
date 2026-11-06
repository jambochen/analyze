# 大廳對局的依狀態調整的勝率：與 arena 相同的四步與保留判準，第一步改為大廳的整體校正。
# 規格見 docs/methods.md「依狀態調整的勝率」的「大廳對局」。對手由玩家自選，但這裡只做預測。
# 上一局為同一速度類別、開局間隔不超過 GAP 分鐘的大廳對局；主分析 60 分鐘，30 與 120 分鐘為敏感度。
# 另把 arena 上半年估得的模型直接套到大廳下半年，檢查可移轉性。2015 年的前瞻驗證樣本不再動用。

import os
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, "scripts")

import numpy as np
import polars as pl

import calibrated
import likelihood as lk
import novice
from state_adjusted import MODEL_DIR, PARAMS, XCOLS, diff_row, losses, ll, probs

UTC = "UTC"
SPEEDS = ["Bullet", "Blitz", "Classical"]
GAPS = [60, 30, 120]
H2 = pl.datetime(2014, 7, 1, time_zone=UTC)

rp = pl.scan_parquet("data/processed/glicko_replay.parquet").select("game_id", "white_rd", "black_rd")

step_rows, coef_rows, pred_rows, sample_rows = [], [], [], []
for spd in SPEEDS:
    base = (
        pl.scan_parquet("data/processed/player_games.parquet")
        .filter(pl.col("speed") == spd, pl.col("source") == "lobby", pl.col("start").dt.year() == 2014, pl.col("score").is_not_null())
        .select("game_id", "player", "opponent", "color", "start", "speed", "elo", "opp_elo", "result", "score")
        .join(rp, on="game_id")
        .with_columns(
            rd=pl.when(pl.col("color") == "white").then("white_rd").otherwise("black_rd").cast(pl.Float64),
            opp_rd=pl.when(pl.col("color") == "white").then("black_rd").otherwise("white_rd").cast(pl.Float64),
        )
        .drop("white_rd", "black_rd")
        .collect(engine="streaming")
    )
    base = calibrated.rd_baseline(base).sort("player", "start", "game_id").with_columns(
        gap_min=(pl.col("start") - pl.col("start").shift(1).over("player")).dt.total_seconds() / 60,
        prev_result0=pl.col("result").shift(1).over("player"),
        dr0=((pl.col("elo") - pl.col("elo").shift(1).over("player")) / 100).cast(pl.Float64),
    )
    for gap in GAPS:
        ok = pl.col("gap_min") <= gap
        b = base.with_columns(prev_result=pl.when(ok).then("prev_result0"), dr=pl.when(ok).then("dr0"))
        opp = b.select("game_id", opponent="player", opp_prev_result="prev_result", opp_dr="dr")
        t = (
            b.join(opp, on=["game_id", "opponent"], how="left")
            .filter(pl.col("prev_result").is_not_null())
            .with_columns(
                y=(pl.col("score") * 2).cast(pl.Int64),
                prev_draw=(pl.col("prev_result") == "draw").cast(pl.Float64),
                prev_win=(pl.col("prev_result") == "win").cast(pl.Float64),
                opp_missing=pl.col("opp_prev_result").is_null().cast(pl.Float64),
                opp_prev_draw=(pl.col("opp_prev_result") == "draw").fill_null(False).cast(pl.Float64),
                opp_prev_win=(pl.col("opp_prev_result") == "win").fill_null(False).cast(pl.Float64),
                opp_dr=pl.col("opp_dr").fill_null(0.0),
                novice_rd=novice.novice_rd("rd", "opp_rd"),
                train=pl.col("start") < H2,
            )
            .with_columns(
                d_prev_draw=pl.col("prev_draw") - pl.col("opp_prev_draw"),
                d_prev_win=pl.col("prev_win") - pl.col("opp_prev_win"),
                d_dr=pl.col("dr") - pl.col("opp_dr"),
            )
            .select("player", "opponent", "eta", "w", "y", "novice_rd", "train", *XCOLS)
        )
        sample_rows.append({"speed": spd, "gap_min": gap, "rows_train": t["train"].sum(), "rows_eval": (~t["train"]).sum(), "players": t["player"].n_unique(), "opp_missing": t["opp_missing"].mean()})
        tr, te = t.filter(pl.col("train")), t.filter(~pl.col("train"))
        args = lambda d: (d["eta"].to_numpy(), d["w"].to_numpy(), d.select(XCOLS).to_numpy().astype(np.float64), d["y"].to_numpy())
        atr, ate = args(tr), args(te)
        wte, bte = te["player"].to_numpy(), te["opponent"].to_numpy()
        T0 = np.zeros(len(PARAMS))
        fits, loss = {}, {}

        def run(nm, free):
            fits[nm] = lk.fit(ll, T0, np.array(free, dtype=int), atr, tr["player"].to_numpy(), tr["opponent"].to_numpy())
            loss[nm] = losses(probs(fits[nm][0], *ate[:3]), ate[3])

        run("baseline", [])
        kept, cur = [], "baseline"
        for nm, idx in [("lobby", [0, 1]), ("prev_result", [2, 3]), ("dr", [4])]:
            run(nm, kept + idx)
            r = diff_row(spd, nm, cur, loss[nm], loss[cur], np.ones(te.height, bool), wte, bte, "all")[0]
            keep = r["diff"] + 1.96 * r["diff_se_dyadic"] < 0
            step_rows.append({"speed": spd, "gap_min": gap, "step": nm, "vs": cur, "diff": r["diff"], "diff_se_dyadic": r["diff_se_dyadic"], "kept": keep})
            if keep:
                kept, cur = kept + idx, nm
        own = [j for j in kept if 2 <= j <= 4]
        sym = [j for j in kept if j < 2] + [j + 3 for j in own] + [8]
        run("opp_sym", sym)
        r = diff_row(spd, "opp_sym", cur, loss["opp_sym"], loss[cur], np.ones(te.height, bool), wte, bte, "all")[0]
        keep = r["diff"] + 1.96 * r["diff_se_dyadic"] < 0
        step_rows.append({"speed": spd, "gap_min": gap, "step": "opp_sym", "vs": cur, "diff": r["diff"], "diff_se_dyadic": r["diff_se_dyadic"], "kept": keep})
        if keep:
            kept, cur = sym, "opp_sym"
        th, _, V, *_ = fits[cur]
        for k, j in enumerate(kept):
            coef_rows.append({"speed": spd, "gap_min": gap, "model": cur, "param": PARAMS[j], "est": th[j], "se_dyadic": np.sqrt(V[k, k]), "n": tr.height})
        # 可移轉性：arena 上半年的模型直接套到大廳下半年
        loss["arena_model"] = losses(probs(np.load(f"{MODEL_DIR}/{spd}.npz")["th"], *ate[:3]), ate[3])
        nov = te["novice_rd"].to_numpy()
        for group, msk in [("all", np.ones(te.height, bool)), ("novice", nov), ("established", ~nov)]:
            for nm in [cur, "arena_model"]:
                for row in diff_row(spd, nm, "baseline", loss[nm], loss["baseline"], msk, wte, bte, group):
                    pred_rows.append(row | {"gap_min": gap})
        print(spd, gap, "final", cur, flush=True)
    del base

for nm, rows in [("sample", sample_rows), ("steps", step_rows), ("coef", coef_rows), ("prediction", pred_rows)]:
    df = pl.DataFrame(rows)
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250):
        print(df)
    df.write_csv(f"outputs/tables/state_adjusted_lobby_{nm}.csv")
