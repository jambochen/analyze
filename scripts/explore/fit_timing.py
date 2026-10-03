import sys
import time

import polars as pl

sys.path.insert(0, "scripts")
import simplex
import stratum

# 模擬研究與 bootstrap 都是同一次配適重複數百次，單次時間決定重複次數與是否須抽取部分的層。
b = stratum.load()
rows = []

t = time.perf_counter()
stratum.fit(b.drop_nulls(stratum.WX), "y1_attributed")
rows.append(("within estimator", time.perf_counter() - t))

t = time.perf_counter()
d, terms = simplex.data(b)
rows.append(("Poisson 長表整理", time.perf_counter() - t))

t = time.perf_counter()
simplex.fit(d, terms)
rows.append(("Poisson 配適", time.perf_counter() - t))

out = pl.DataFrame(rows, schema=["step", "seconds"], orient="row")
print(out)
out.write_csv("outputs/tables/fit_timing.csv")
