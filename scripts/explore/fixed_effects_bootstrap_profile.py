# fixed_effects_churn 的玩家 bootstrap 每輪時間花在哪：複製玩家（join 與字串代碼）、疊兩樣本並轉 pandas、feglm。
# 再以同一份重抽比較 demean 的兩種解法：MAP（交替投影，正式腳本現用）與預條件的 LSMR，看耗時、平均用到幾個核與係數差。
# 定義與樣本取自正式腳本的前段，避免複製後漂移；只跑幾輪，亂數狀態與正式腳本不同，不影響耗時。

import resource
import sys
import time

import numpy as np
import polars as pl
import pyfixest as pf

sys.path.insert(0, "scripts")

src = open("scripts/fixed_effects_churn.py").read()
g = {}
exec(src[: src.index("desc = []")], g)
samples, heliyon_sample, SPECS, N_SUB, rng = (g[k] for k in ["samples", "heliyon_sample", "SPECS", "N_SUB", "rng"])

lob_players = samples[False]["player"].unique().sort().sample(N_SUB, seed=1)
sub = {False: heliyon_sample(samples[False].filter(pl.col("player").is_in(lob_players.implode()))), True: samples[True]}
union = pl.concat([sub[False]["player"], sub[True]["player"]]).unique().sort()
print({a: d.height for a, d in sub.items()}, len(union))

xcols = SPECS["raw"]
for rep in range(3):
    t = [time.perf_counter()]
    cnt = pl.DataFrame({"player": union, "copies": rng.multinomial(len(union), np.full(len(union), 1 / len(union)))}).filter(pl.col("copies") > 0)
    cnt = cnt.with_columns(copy=pl.int_ranges(pl.col("copies"))).explode("copy")
    db = {a: d.join(cnt.select("player", "copy"), on="player").with_columns(player=pl.col("player").cast(pl.String) + "#" + pl.col("copy").cast(pl.String)).drop("copy") for a, d in sub.items()}
    t.append(time.perf_counter())
    st = pl.concat([db[False].with_columns(arena=pl.lit(0.0)), db[True].with_columns(arena=pl.lit(1.0))], how="diagonal").with_columns(
        ps=pl.col("player").cast(pl.String) + "_" + pl.col("arena").cast(pl.String), ds=pl.col("day").cast(pl.String) + "_" + pl.col("arena").cast(pl.String)
    )
    st = st.with_columns([(pl.col(x) * pl.col("arena")).alias(f"{x}__arena") for x in xcols])
    terms = xcols + [f"{x}__arena" for x in xcols]
    pdf = st.select("player", "ps", "ds", "churn", *terms).to_pandas()
    del st
    t.append(time.perf_counter())
    m = pf.feglm("churn ~ " + " + ".join(terms) + " | ps + ds", data=pdf, family="logit", vcov={"CRV1": "player"}, demeaner_backend="rust", lean=True, store_data=False, copy_data=False)
    t.append(time.perf_counter())
    print(rep, "rows", len(pdf), "copy", round(t[1] - t[0], 2), "stack+pandas", round(t[2] - t[1], 2), "feglm", round(t[3] - t[2], 2))

cpu = lambda: (lambda r: r.ru_utime + r.ru_stime)(resource.getrusage(resource.RUSAGE_SELF))
fml = "churn ~ " + " + ".join(terms) + " | ps + ds"
res = {}
for name, dm in [("map", pf.MapDemeaner()), ("lsmr_within", pf.LsmrDemeaner(backend="within"))]:
    w0, c0 = time.perf_counter(), cpu()
    m = pf.feglm(fml, data=pdf, family="logit", vcov={"CRV1": "player"}, demeaner=dm, lean=True, store_data=False, copy_data=False)
    w, c = time.perf_counter() - w0, cpu() - c0
    res[name] = (m.coef().to_numpy(), m.se().to_numpy())
    print(name, "wall", round(w, 2), "cpu", round(c, 2), "cores", round(c / w, 1))
(b0, s0), (b1, s1) = res["map"], res["lsmr_within"]
print("coef max rel diff", np.max(np.abs(b0 - b1) / np.abs(b0)), "se max rel diff", np.max(np.abs(s0 - s1) / s0))
