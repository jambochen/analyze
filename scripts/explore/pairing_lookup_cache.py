# 重抽時 E[tci, ui, ui[opp]] 的查表是否受快取限制。E 為 3 x n_elo x n_elo 的 float64，遠大於 L3，
# 對手的等級分在批內不相關，查表近乎隨機存取；改以 calibrated.probs 的封閉形式逐列計算則是循序存取。
# 多行程共用一份 L3 與記憶體頻寬，查表若受頻寬限制，單次耗時會隨行程數上升，計算版則大致不變。

import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, "scripts")
import pairing_null
from arena_batches import player_sample
from calibrated import coef

REPS = 30
PROCS = [1, 4, 9, os.cpu_count() - 2]

m, ns = pairing_null.structure(player_sample())
ui, tci, bi, nb = ns["ui"], ns["tci"], ns["bi"], ns["nb"]
elo = ns["elos"][ui].astype(float)
c = coef.sort("speed")
assert c["speed"].to_list() == pairing_null.TCS
s0, s1, w0, w1, w2 = (c[k].to_numpy() for k in ["slope_diff", "slope_diff_x_rating", "log_draw_width", "log_draw_width_x_rating", "log_draw_width_x_rating2"])
print("rows", m.height, "batches", nb, "elos", len(ns["elos"]), "E MB", ns["E"].nbytes / 2**20)


def e_formula(opp):
    # color=False 的 e_cal：白方優勢設 0，自身視角等同白方視角
    o = elo[opp]
    xr = ((elo + o) / 2 - 1500) / 100
    xd = (elo - o) / 100
    eta = (s0[tci] + s1[tci] * xr) * xd
    w = np.exp(w0[tci] + w1[tci] * xr + w2[tci] * xr**2)
    return (1 / (1 + np.exp(w - eta)) + 1 / (1 + np.exp(-eta - w))) / 2


def e_lookup(opp):
    return ns["E"][tci, ui, ui[opp]]


# 與 match_quality_leave 的 slopes 同形：依批去均值，再依子群累加，掃過全部列數次
lab = (m["rule"].to_numpy() == "smart").astype(int)
ec = np.random.default_rng(0).normal(size=m.height)
cnt = np.bincount(bi, minlength=nb)


def stat(e):
    xc = e - (np.bincount(bi, e, nb) / cnt)[bi]
    return np.bincount(lab, xc * ec) / np.bincount(lab, xc * xc)


opp = pairing_null._repair(ns, "main", np.random.default_rng(1))
assert np.allclose(e_formula(opp), e_lookup(opp), rtol=0, atol=1e-12)


def timed(job):
    which, seed = job
    rng = np.random.default_rng(seed)
    t = {"repair": 0.0, "e": 0.0, "stat": 0.0}
    for _ in range(REPS):
        a = time.perf_counter()
        o = pairing_null._repair(ns, "main", rng)
        b = time.perf_counter()
        e = e_lookup(o) if which == "lookup" else e_formula(o)
        c_ = time.perf_counter()
        stat(e)
        t["repair"] += b - a
        t["e"] += c_ - b
        t["stat"] += time.perf_counter() - c_
    return {k: v / REPS for k, v in t.items()}


for which in ["lookup", "formula"]:
    for p in PROCS:
        with ProcessPoolExecutor(p, mp_context=mp.get_context("fork")) as ex:
            res = list(ex.map(timed, [(which, i) for i in range(p)]))
        print(which, p, {k: round(np.mean([r[k] for r in res]) * 1000, 1) for k in res[0]}, "ms/draw")
