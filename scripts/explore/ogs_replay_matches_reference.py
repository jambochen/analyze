# ogs_replay.py 與 OGS 的參考實作（ref/goratings-src）逐局比對：以參考實作的 glicko2_update、
# get_handicap_adjustment 與 should_skip_game 重跑資料庫的前 N 局，比對每局對局前的等級分、RD 與期望勝率。
# 參考實作的 SQL 只以結束時間排序；這裡與 ogs_replay.py 同樣加上對局代碼，兩邊才逐局對得上。

import argparse
import sqlite3
import sys

sys.path.insert(0, "ref/goratings-src")

import numpy as np
import polars as pl

from analysis.util import RatingMath
from analysis.util.InMemoryStorage import InMemoryStorage
from analysis.util.SkipLogic import should_skip_game
from goratings.interfaces import GameRecord
from goratings.math.glicko2 import Glicko2Entry, glicko2_configure, glicko2_update

N = 300_000
glicko2_configure(tao=0.5, min_rd=10.0, max_rd=500.0, aging_period_days=None)
RatingMath.configure_rating_to_rank(argparse.Namespace(ranks="log", a=525.0, c=23.15, d=0.0, m=100.0, b=9.0, p=1.0))
adj = RatingMath.get_handicap_adjustment
RULES = {"aga": "aga", "chinese": "chinese", "ing": "ing", "japanese": "japanese", "korean": "korean", "nz": "nz",
         "Japanese": "japanese", "age": "aga", "ing sst": "ing", "ogs": "japanese"}

con = sqlite3.connect("file:data/raw/ogs/ogs-data.db?mode=ro", uri=True)
q = """
SELECT g.id, size, handicap, komi, black_id, white_id, time_per_move, timeout, winner_id, ended, rules
FROM game_records g
LEFT JOIN players bp ON black_id = bp.id
LEFT JOIN players wp ON white_id = wp.id
WHERE (bp.is_bot = 0 OR bp.id > 50000) AND (wp.is_bot = 0 OR wp.id > 50000)
  AND black_id != 82957 AND white_id != 82957
  AND (bp.is_bot != 1 OR timeout = 0) AND (wp.is_bot != 1 OR timeout = 0)
ORDER BY ended, g.id LIMIT ?
"""
st = InMemoryStorage(Glicko2Entry)
rows = []
for r in con.execute(q, (N,)):
    g = GameRecord(*r[:10], RULES[r[10]])
    skip = should_skip_game(g, st)
    b, w = st.get(g.black_id), st.get(g.white_id)
    kw = dict(komi=g.komi, size=g.size, rules=g.rules)
    e = b.expected_win_probability(w, adj("black", b.rating, g.handicap, **kw), ignore_g=True)
    rows.append((g.game_id, b.rating, b.deviation, w.rating, w.deviation, e, skip))
    if skip:
        continue
    nb = glicko2_update(b, [(w.copy(adj("white", w.rating, g.handicap, **kw)), g.winner_id == g.black_id)], timestamp=g.ended)
    nw = glicko2_update(w, [(b.copy(adj("black", b.rating, g.handicap, **kw)), g.winner_id == g.white_id)], timestamp=g.ended)
    st.set(g.black_id, nb)
    st.set(g.white_id, nw)

ref = pl.DataFrame(rows, schema=["game_id", "black_rating", "black_rd", "white_rating", "white_rd", "expected_black", "skipped"], orient="row")
mine = pl.read_parquet("data/processed/ogs_replay.parquet", n_rows=N)
assert (ref["game_id"] == mine["game_id"]).all()
assert (ref["skipped"] == mine["skipped"]).all()
for c in ["black_rating", "black_rd", "white_rating", "white_rd", "expected_black"]:
    print(c, np.abs(ref[c].to_numpy() - mine[c].to_numpy()).max())
