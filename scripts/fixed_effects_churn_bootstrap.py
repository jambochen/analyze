# 玩家 block bootstrap 對 cluster sandwich 標準誤的檢查（模型見 fixed_effects_churn.py）。
# 每次從抽樣玩家上兩樣本的玩家聯集有放回地抽；重抽的樣本偶爾使固定效應去均值不收斂，該次跳過並記下次數，
# 其餘重抽的標準差仍是 sandwich 的有效對照。

import sys

sys.path.insert(0, "scripts")

import numpy as np
import polars as pl

from fixed_effects import diff_rows, heliyon_sample, load, player_subsample

B = 100
rng = np.random.default_rng(0)

base = load().filter(pl.col("cut") == "utc", pl.col("day_pure"))
sub = player_subsample({a: heliyon_sample(base.filter(pl.col("arena") == a)) for a in [False, True]})

union = pl.concat([sub[False]["player"], sub[True]["player"]]).unique().sort()
boot, failed = [], []
for rep in range(B):
    cnt = pl.DataFrame({"player": union, "copies": rng.multinomial(len(union), np.full(len(union), 1 / len(union)))}).filter(pl.col("copies") > 0)
    # 重複抽到的玩家各為一個新玩家（各自的固定效應與 cluster），以複本編號區分
    cnt = cnt.with_columns(copy=pl.int_ranges(pl.col("copies"))).explode("copy")
    db = {a: d.join(cnt.select("player", "copy"), on="player").with_columns(player=pl.col("player").cast(pl.String) + "#" + pl.col("copy").cast(pl.String)).drop("copy") for a, d in sub.items()}
    try:
        rows = diff_rows(db[False], db[True], "raw", "logit", "boot")
    except ValueError as e:
        # pyfixest 的去均值在 maxiter 內不收斂；只接這一種錯誤
        assert "Demeaning failed" in str(e)
        failed.append(rep)
        print("boot", rep, "failed", flush=True)
        continue
    boot += [{**r, "rep": rep} for r in rows]
    print("boot", rep, flush=True)
boot = pl.DataFrame(boot)
boot.write_csv("outputs/tables/fixed_effects_churn_bootstrap_draws.csv")
ref = pl.DataFrame(diff_rows(sub[False], sub[True], "raw", "logit", "subsample"))
qs = ["beta_lobby", "beta_arena", "diff_lobby_minus_arena", "ase_diff"]
ses = {"beta_lobby": "se_lobby", "beta_arena": "se_arena", "diff_lobby_minus_arena": "diff_se", "ase_diff": "ase_diff_se"}
check = pl.concat([
    pl.DataFrame({"term": [t] * len(qs), "quantity": qs,
                  "estimate": [ref.filter(pl.col("term") == t)[q][0] for q in qs],
                  "sandwich_se": [ref.filter(pl.col("term") == t)[ses[q]][0] for q in qs],
                  "bootstrap_se": [boot.filter(pl.col("term") == t)[q].std() for q in qs],
                  "B": [B] * len(qs), "B_failed": [len(failed)] * len(qs)})
    for t in ["gap_mean", "gap_sd"]
]).with_columns(ratio=pl.col("bootstrap_se") / pl.col("sandwich_se"))
print(check)
print("failed reps", failed)
check.write_csv("outputs/tables/fixed_effects_churn_bootstrap_check.csv")
