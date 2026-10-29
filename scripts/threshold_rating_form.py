# 切點對平均等級分的形式：白方優勢 a 與和局帶寬的對數 log w 依平均等級分分組各自估計，
# 不預設形式，再與線性、二次的擬合對照，並以修正的概似比檢定二次項是否需要、是否足夠。
# 模型與樣本同 expected_score_baselines.py 的加入 RD 的有序 logistic。

import os
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, "scripts")

import jax.numpy as jnp
import numpy as np
import polars as pl

import likelihood as lk
import ordinal

UTC = "UTC"
CLASSES = ["Bullet", "Blitz", "Classical"]
N_FIT = 1_000_000
# 等量分組：每組局數相同，各組的精度相近，不另定等級分的切點
K = 20

d = (
    pl.scan_parquet("data/processed/lichess_games.parquet")
    .filter(
        pl.col("speed").is_in(CLASSES), pl.col("base_seconds").is_not_null(),
        pl.col("start") >= pl.datetime(2014, 1, 1, time_zone=UTC), pl.col("start") < pl.datetime(2015, 1, 1, time_zone=UTC),
        pl.col("white_elo").is_not_null(), pl.col("black_elo").is_not_null(),
    )
    .select(
        "game_id", "speed", "white_elo", "black_elo",
        y=pl.col("result").replace_strict({"0-1": 0, "1/2-1/2": 1, "1-0": 2}, default=None, return_dtype=pl.Int8),
        w="white_id", b="black_id",
    )
    .filter(pl.col("y").is_not_null())
    .join(pl.scan_parquet("data/processed/glicko_replay.parquet").select("game_id", "perf", "white_rd", "black_rd"), on="game_id")
    .filter(pl.col("perf") == pl.col("speed"))
    .with_columns(
        rbar=(pl.col("white_elo") + pl.col("black_elo")) / 2,
        xd=(pl.col("white_elo") - pl.col("black_elo")).cast(pl.Float64) / 100,
        gxd=ordinal.g("white_rd", "black_rd") * (pl.col("white_elo") - pl.col("black_elo")).cast(pl.Float64) / 100,
    )
    .with_columns(xr=(pl.col("rbar") - 1500) / 100)
    .sort("game_id")
    .select("speed", "w", "b", "y", "rbar", "xr", "xd", "gxd")
    .collect(engine="streaming")
)
code = pl.concat([d["w"], d["b"]]).rank("dense").cast(pl.UInt32) - 1
d = d.with_columns(w=code[: d.height], b=code[d.height :])
del code


# t = [b0, b1, b4, a 的係數 (p 個), log w 的係數 (p 個)]；a 與 log w 用同一個設計矩陣 Z
def ll(t, Z, xr, xd, gxd, y):
    p = Z.shape[1]
    eta = Z @ t[3 : 3 + p] + (t[0] + t[1] * xr) * xd + t[2] * gxd
    return ordinal.ll(eta, jnp.exp(Z @ t[3 + p :]), y)


test_rows, bin_rows, curve_rows = [], [], []
for tc in CLASSES:
    m = d.filter(pl.col("speed") == tc).sample(N_FIT, seed=1)
    w, b = m["w"].to_numpy(), m["b"].to_numpy()
    xr, xd, gxd, y = (m[c].to_numpy().astype(np.float64) for c in ["xr", "xd", "gxd", "y"])
    edges = np.quantile(xr, np.linspace(0, 1, K + 1)[1:-1])
    k = np.searchsorted(edges, xr, side="right")
    D = np.eye(K)[k]

    # 分組模型：每組各自的 a 與 log w
    t0 = np.r_[0.6, 0, 0, np.full(K, 0.1), np.full(K, -2.5)]
    th, _, V, Hi, *_ = lk.fit(ll, t0, np.arange(len(t0)), (D, xr, xd, gxd, y), w, b)
    for j in range(K):
        for part, off in [("a", 3), ("log_w", 3 + K)]:
            bin_rows.append({"speed": tc, "bin": j, "rbar": m["rbar"].to_numpy()[k == j].mean(), "n": int((k == j).sum()), "part": part, "est": th[off + j], "se_dyadic": np.sqrt(V[off + j, off + j]), "se_iid": np.sqrt(Hi[off + j, off + j])})

    # 巢狀的多項式模型：Z = [1, x, x^2, 分組虛擬變數 2..K]。二次之外再加分組，檢查二次夠不夠
    Z = np.column_stack([np.ones_like(xr), xr, xr**2, D[:, 1:]])
    p = Z.shape[1]
    A, W = 3 + np.arange(p), 3 + p + np.arange(p)
    t0 = np.r_[0.6, 0, 0, 0.1, np.zeros(p - 1), -2.5, np.zeros(p - 1)]
    allf = np.arange(len(t0))
    fits = {
        "quad_plus_bins": allf,
        "quad": np.setdiff1d(allf, np.r_[A[3:], W[3:]]),
        "linear": np.setdiff1d(allf, np.r_[A[2:], W[2:]]),
        "a_linear": np.setdiff1d(allf, np.r_[A[2:], W[3:]]),
        "w_linear": np.setdiff1d(allf, np.r_[A[3:], W[2:]]),
        "a_quad_w_bins": np.setdiff1d(allf, A[3:]),
        "w_quad_a_bins": np.setdiff1d(allf, W[3:]),
    }
    res = {nm: lk.fit(ll, t0, f, (Z, xr, xd, gxd, y), w, b) for nm, f in fits.items()}
    # (名稱, 完整模型, 受限模型)；lr_weights 的索引是參數在完整模型 free 中的位置
    for name, full, rest in [
        ("a_quad_term", "quad", "a_linear"),
        ("w_quad_term", "quad", "w_linear"),
        ("a_bins_beyond_quad", "quad_plus_bins", "a_quad_w_bins"),
        ("w_bins_beyond_quad", "quad_plus_bins", "w_quad_a_bins"),
    ]:
        _, ll1, V1, Hi1, *_ = res[full]
        idx = np.flatnonzero(~np.isin(fits[full], fits[rest]))
        test_rows.append({"speed": tc, "test": name, **lk.adjusted_lrt(2 * (ll1 - res[rest][1]), lk.lr_weights(Hi1, V1, idx))})

    # 曲線只畫到 1% 與 99% 分位，兩端資料少，外推不作解讀
    grid = np.linspace(*np.quantile(xr, [0.01, 0.99]), 200)
    for nm in ["linear", "quad"]:
        t = res[nm][0]
        for part, off in [("a", 3), ("log_w", 3 + p)]:
            curve_rows.append(pl.DataFrame({"speed": tc, "model": nm, "part": part, "rbar": 1500 + 100 * grid, "est": t[off] + t[off + 1] * grid + t[off + 2] * grid**2}))
    print(tc, "done")

tests = pl.DataFrame(test_rows).with_columns(p_holm=lk.holm(pl.Series([r["p"] for r in test_rows])))
tests.write_csv("outputs/tables/threshold_rating_form_tests.csv")
bins = pl.DataFrame(bin_rows)
bins.write_csv("outputs/tables/threshold_rating_form_bins.csv")
curve = pl.concat(curve_rows)
curve.write_csv("outputs/tables/threshold_rating_form_curve.csv")
print(tests)
