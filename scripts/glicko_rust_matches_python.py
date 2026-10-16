# Rust 版重放（rust/glicko_replay，Python 模組名 glicko_rs）與 Python 參考實作（scripts/glicko.py）是否逐局一致。
# 對手的狀態互相依賴，子集必須是某時點以前的全部對局，取 2013 年第一季。
# 純重放與以表頭校正的重放各比一次，另比 replay_match、header_chain、step 三個輔助函式。

import time

import glicko
import glicko_rs
import numpy as np
import polars as pl

MN = np.iinfo(np.int32).min
games = glicko.inputs(
    pl.scan_parquet("data/processed/lichess_games.parquet")
    .filter(pl.col("start") < pl.datetime(2013, 4, 1, time_zone="UTC"))
    .select("start", "white_id", "black_id", "result", "white_elo", "black_elo", "white_rating_diff", "black_rating_diff", "base_seconds", "increment_seconds")
).collect()
args = [games[c].to_numpy() for c in ["white_id", "black_id", "perf", "score", "rated"]]
# 重算期間未更新的對局，表頭不是重算結果，不作錨點
wa = games["white_elo"].fill_null(MN).to_numpy() * 1
ba = games["black_elo"].fill_null(MN).to_numpy() * 1
wa[~args[4]] = MN
ba[~args[4]] = MN
wa_py = [None if x == MN else x for x in wa.tolist()]
ba_py = [None if x == MN else x for x in ba.tolist()]

names = ["white_rating", "white_rd", "white_vol", "black_rating", "black_rd", "black_vol", "white_flags", "black_flags"]
rows = []
timing = []
results = {}
for mode, anchor in [("pure", False), ("anchored", True)]:
    t0 = time.perf_counter()
    py = np.array(glicko.replay(*[a.tolist() for a in args], wa_py, ba_py, anchor=anchor), dtype=np.float64)
    t_py = time.perf_counter() - t0
    t0 = time.perf_counter()
    rs = np.column_stack([np.asarray(x, dtype=np.float64) for x in glicko_rs.replay(*args, wa, ba, glicko.N_PERF, anchor=anchor, rd_trunc=glicko.RD_TRUNC)])
    t_rs = time.perf_counter() - t0
    results[mode] = (py, rs)
    # 位元一致才算一致；另記最大絕對差，以便不一致時判斷是否只是浮點誤差
    rows.append(pl.DataFrame({
        "mode": mode, "quantity": names, "n_games": games.height,
        "n_bit_identical": [(py[:, k] == rs[:, k]).sum() for k in range(8)],
        "max_abs_diff": [np.abs(py[:, k] - rs[:, k]).max() for k in range(8)],
    }))
    timing.append(pl.DataFrame({"mode": mode, "implementation": ["python", "rust"], "n_games": games.height, "seconds": [t_py, t_rs]}))
res = pl.concat(rows)
print(res)

# replay_match 的旗標須與 replay 的旗標相同
m = glicko_rs.replay_match(
    args[0], args[1], args[3], wa, ba, [args[2]], [args[4]],
    [(0, 0, glicko.TAU, glicko.MIN_RATING, False, True, glicko.RD_TRUNC), (0, 0, glicko.TAU, glicko.MIN_RATING, True, True, glicko.RD_TRUNC)], glicko.N_PERF,
)
for mm, mode in zip(m, ["pure", "anchored"]):
    py = results[mode][0]
    print("replay_match agrees with python", mode, (mm == (py[:, 6].astype(np.uint8) | py[:, 7].astype(np.uint8) << 4)).all())

# 表頭鏈檢查的兩個實作也須一致
chain_in = [args[0], args[1], args[2], wa, ba, games["white_rating_diff"].fill_null(MN).to_numpy(), games["black_rating_diff"].fill_null(MN).to_numpy(), args[4]]
py_chain = np.array(glicko.header_chain(*[[None if x == MN else x for x in a.tolist()] if a.dtype == np.int32 else a.tolist() for a in chain_in]))
rs_chain = np.column_stack(glicko_rs.header_chain(*chain_in, glicko.N_PERF))
print("header_chain agrees with python:", (py_chain == rs_chain).all())

# 單局更新：以 Python 的校正重放輸出為輸入，兩邊各算一次
py = results["anchored"][0]
sp = np.array([(a[0], b[0]) for a, b in (glicko.update(tuple(r[:3]), tuple(r[3:6]), s) for r, s in zip(py.tolist(), args[3].tolist()))])
ss = np.column_stack(glicko_rs.step(*[np.ascontiguousarray(py[:, k]) for k in range(6)], args[3]))
print("step agrees with python:", (sp == ss).all())

timing = pl.concat(timing).with_columns(games_per_second=pl.col("n_games") / pl.col("seconds"))
print(timing)
res.write_csv("outputs/tables/glicko_replay_consistency.csv")
timing.write_csv("outputs/tables/glicko_replay_timing.csv")
