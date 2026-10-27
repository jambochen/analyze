# 高階 Markov：給定主基準，本局結果的分布是否依前 k 局（k = 1, 2, 3）的結果組合而定。
# 樣本、上半年估計、下半年評估與 dyadic 標準誤同 scripts/state_adjusted_probs.py；k = 1 即其「上一局結果」一步。
# 規格在執行前寫定：
# - 模型：有序 logistic，主基準的 eta、w 作 offset，加上前 k 局結果組合的格子效果（飽和，即給定主基準的 k 階 Markov）。
#   同一場錦標賽內不足 k 局時，缺的局記為「無」，三個 k 的樣本相同
# - 階層式編碼使三個模型巢狀：k = 1 為上一局的勝、和兩項；k = 2 加上「上一局 × 上上局」，上上局為「無」者為參照；
#   k = 3 再加「前兩局 × 第三局」，第三局為「無」者為參照
# - 判準：下半年的對數分數比 k - 1 改善，且差值以 dyadic sandwich 估得的 95% 區間不含 0
# - 檢定：上半年以修正的概似比檢定 k 對 k - 1，三個速度類別 × 兩個增量為一族，以 Holm 法校正
# - 分格計數會含玩家本身的強弱（剛贏的人本來就較強），以主基準作 offset 才是「系統沒算到的部分」

import os
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, "scripts")

import numpy as np
import polars as pl

import likelihood as lk
import ordinal
from arena_batches import player_sample_rd

UTC = "UTC"
SPEEDS = ["Bullet", "Blitz", "Classical"]

grp = ["player", "tournament_id"]
code = pl.col("result").replace_strict({"loss": 0, "draw": 1, "win": 2}, return_dtype=pl.Int64)
t = (
    player_sample_rd()
    .sort("player", "tournament_id", "start", "game_id")
    .with_columns(p1=code.shift(1).over(grp), p2=code.shift(2).over(grp), p3=code.shift(3).over(grp))
    .filter(pl.col("p1").is_not_null())
    .with_columns(
        y=(pl.col("score") * 2).cast(pl.Int64),
        p2=pl.col("p2").fill_null(3),
        p3=pl.col("p3").fill_null(3),
        train=pl.col("start") < pl.datetime(2014, 7, 1, time_zone=UTC),
    )
)


def design(d):
    p1, p2, p3 = (d[c].to_numpy() for c in ["p1", "p2", "p3"])
    k1 = [p1 == 1, p1 == 2]
    k2 = [(p1 == i) & (p2 == j) for i in range(3) for j in range(3)]
    k3 = [(p1 == i) & (p2 == j) & (p3 == m) for i in range(3) for j in range(3) for m in range(3)]
    return np.column_stack(k1 + k2 + k3).astype(np.float64)


FREE = {1: np.arange(2), 2: np.arange(11), 3: np.arange(38)}


def ll(th, eta, w, X, y):
    return ordinal.ll(eta + X @ th, w, y)


def losses(p, y):
    O = np.eye(3)[y]
    C, Co = np.cumsum(p, 1)[:, :2], np.cumsum(O, 1)[:, :2]
    return {"log_loss": -np.log(p[np.arange(len(y)), y]), "rps": ((C - Co) ** 2).sum(1) / 2}


sample = t.group_by("speed", "train").agg(
    n=pl.len(), players=pl.col("player").n_unique(), has_p2=(pl.col("p2") < 3).mean(), has_p3=(pl.col("p3") < 3).mean()
).sort("speed", "train")
print(sample)

pred_rows, lrt_rows = [], []
for spd in SPEEDS:
    ts = t.filter(pl.col("speed") == spd)
    tr, te = ts.filter(pl.col("train")), ts.filter(~pl.col("train"))
    atr = (tr["eta"].to_numpy(), tr["w"].to_numpy(), design(tr), tr["y"].to_numpy())
    eta_te, w_te, X_te, y_te = te["eta"].to_numpy(), te["w"].to_numpy(), design(te), te["y"].to_numpy()
    wtr, btr = tr["player"].to_numpy(), tr["opponent"].to_numpy()
    wte, bte = te["player"].to_numpy(), te["opponent"].to_numpy()
    T0 = np.zeros(38)
    L = {0: losses(ordinal.probs(eta_te, w_te), y_te)}
    for k in [1, 2, 3]:
        th = lk.fit(ll, T0, FREE[k], atr, wtr, btr)[0]
        L[k] = losses(ordinal.probs(eta_te + X_te @ th, w_te), y_te)
        if k > 1:
            lrt_rows.append({"speed": spd, "k": k, "vs": k - 1, "n": tr.height} | lk.lrt_player_dyad(ll, T0, FREE[k - 1], FREE[k], atr, wtr, btr)[2])
        for metric in ["log_loss", "rps"]:
            dlt = L[k][metric] - L[k - 1][metric]
            se = lk.dyad_se((dlt - dlt.mean()) / len(dlt), wte, bte)
            pred_rows.append({"speed": spd, "k": k, "vs": k - 1, "metric": metric, "n": te.height, "mean": L[k][metric].mean(), "diff": dlt.mean(), "diff_se_dyadic": se, "kept": dlt.mean() + 1.96 * se < 0})
    print(spd, "done", flush=True)

pred = pl.DataFrame(pred_rows)
lrt = pl.DataFrame(lrt_rows)
lrt = lrt.with_columns(p_holm=lk.holm(lrt["p"]))
with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250):
    print(pred)
    print(lrt)
sample.write_csv("outputs/tables/markov_order_sample.csv")
pred.write_csv("outputs/tables/markov_order_prediction.csv")
lrt.write_csv("outputs/tables/markov_order_lrt.csv")
