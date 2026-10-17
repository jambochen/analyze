# 診斷：直接效果的模型中，h_lag1 的修正權重 lambda 以依玩家分群與 dyadic 的 meat 各算一次。
# 兩者接近時，玩家摘要量的 t 檢定（以玩家為群）足夠，不需要 dyadic 的修正。只輸出 lambda，不輸出效果的估計。

import os
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, "scripts")

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl

import likelihood as lk
from arena_batches import player_sample

Q = np.log(10) / 400
ids = pl.scan_parquet("data/processed/lichess_games.parquet").select("game_id", "white_id", "black_id")
rp = pl.scan_parquet("data/processed/glicko_replay.parquet").select("game_id", "white_rd", "black_rd")
grp = ["player", "tournament_id"]
d = (
    player_sample()
    .join(ids.collect(), on="game_id")
    .join(rp.collect(), on="game_id")
    .with_columns(
        rd=pl.when(pl.col("color") == "white").then("white_rd").otherwise("black_rd").cast(pl.Float64),
        opp_rd=pl.when(pl.col("color") == "white").then("black_rd").otherwise("white_rd").cast(pl.Float64),
        opponent=pl.when(pl.col("color") == "white").then("black_id").otherwise("white_id"),
        sign=pl.when(pl.col("color") == "white").then(1.0).otherwise(-1.0),
        h=pl.col("e_bar") - pl.col("e"),
        y=pl.col("result").replace_strict({"loss": 0, "draw": 1, "win": 2}, return_dtype=pl.Int8).cast(pl.Float64),
    )
    .with_columns(
        xr=((pl.col("elo") + pl.col("opp_elo")) / 2 - 1500) / 100,
        xd=(pl.col("elo") - pl.col("opp_elo")) / 100,
        gs=1 / (1 + 3 * Q**2 * (pl.col("rd") ** 2 + pl.col("opp_rd") ** 2) / np.pi**2).sqrt(),
    )
    .sort("player", "tournament_id", "start", "game_id")
    .with_columns(h_lag1=pl.col("h").shift(1).over(grp), layer_lag1=pl.col("layer").shift(1).over(grp))
    .filter(pl.col("h_lag1").is_not_null())
)
COLS = ["sign", "sxr", "sxr2", "xd", "xdxr", "gxd", "elo_c", "elo_c2", "e_bar", "h_lag1"]
d = d.with_columns(sxr=pl.col("sign") * pl.col("xr"), sxr2=pl.col("sign") * pl.col("xr") ** 2, xdxr=pl.col("xd") * pl.col("xr"), gxd=pl.col("gs") * pl.col("xd"), elo_c2=pl.col("elo_c") ** 2)
K = len(COLS)


# t[0] 截距，t[1:K+1] 各共變數，t[K+1:K+4] 和局帶寬的對數（常數、平均等級分、其平方）
def ll(t, X, xr, y):
    eta = t[0] + X @ t[1 : K + 1]
    w = jnp.exp(t[K + 1] + t[K + 2] * xr + t[K + 3] * xr**2)
    u, v = eta + w, eta - w
    ldraw = jax.nn.log_sigmoid(u) + jax.nn.log_sigmoid(-v) + jnp.log(-jnp.expm1(-2 * w))
    return jnp.where(y == 2, jax.nn.log_sigmoid(v), jnp.where(y == 0, jax.nn.log_sigmoid(-u), ldraw))


T0 = np.zeros(K + 4)
T0[K + 1] = -2.5
rows = []
for spd in ["Bullet", "Blitz", "Classical"]:
    for layer in ["multi", "single"]:
        s = d.filter(pl.col("speed") == spd, pl.col("layer_lag1") == layer)
        args = (s.select(COLS).to_numpy().astype(np.float64), s["xr"].to_numpy(), s["y"].to_numpy())
        w, b = s["player"].to_numpy(), s["opponent"].to_numpy()
        free = np.arange(K + 4)
        _, _, V, Hi, S, _ = lk.fit(ll, T0, free, args, w, b)
        pc = np.unique(w, return_inverse=True)[1]
        Sp = np.stack([np.bincount(pc, S[:, j]) for j in range(S.shape[1])], 1)
        Vp = Hi @ (Sp.T @ Sp) @ Hi
        idx = [1 + COLS.index("h_lag1")]
        share = np.bincount(pc) / len(pc)
        rows.append({
            "speed": spd, "layer_lag1": layer, "n": s.height, "players": int(pc.max() + 1), "effective_players": 1 / (share**2).sum(),
            "lambda_dyadic": lk.lr_weights(Hi, V, idx)[0], "lambda_player": lk.lr_weights(Hi, Vp, idx)[0],
        })
        print(rows[-1], flush=True)
out = pl.DataFrame(rows).with_columns(ratio=pl.col("lambda_player") / pl.col("lambda_dyadic"))
print(out)
out.write_csv("outputs/tables/explore_crossover_lambda.csv")
