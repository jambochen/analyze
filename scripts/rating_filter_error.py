# 等級分在剛輸之後是否失準：Glicko 是近似的貝氏濾波器，等級分是實力的事後平均。估計正確時，得分減期望得分
# 不能由對局前已知的資訊預測，包括上一局的結果與上一局等級分的變動（規格見 docs/methods.md「等級分的量測誤差」）。
# Cox（1958）的校準迴歸：logit E[得分] = alpha + beta logit(e)，e 為主基準的期望得分，和局記 0.5
# 1. RD 五等分 × 上一局結果的每一格，檢定 alpha = 0、beta = 1
# 2. 加入上一局等級分的變動 dr，檢定其係數為 0
# 3. dr 的係數是否隨 RD 五等分而不同
# dr 與上一局的結果幾乎綁在一起，三個檢定都分不開濾波誤差與 tilt

import os
import sys

# 先於 likelihood import jax 設定：float64 在這張 GPU 上很慢
os.environ["JAX_PLATFORMS"] = "cpu"
sys.path.insert(0, "scripts")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import likelihood as lk
from arena_batches import rating_filter_sample
from ordinal import Q

SPEEDS = ["Bullet", "Blitz", "Classical"]
RESULTS = ["loss", "draw", "win"]
NQ = 5


def holm(p):
    o = np.argsort(p)
    out = np.empty(len(p))
    out[o] = np.minimum(1, np.maximum.accumulate(p[o] * (len(p) - np.arange(len(p)))))
    return out


t = rating_filter_sample(NQ)
sample = t.group_by("speed", "rdq", "prev_result").agg(n=pl.len(), players=pl.col("player").n_unique(), rd_min=pl.col("rd").min(), rd_max=pl.col("rd").max(), dr_mean=pl.col("dr").mean(), resid_mean=pl.col("resid").mean()).sort("speed", "rdq", "prev_result")
print(sample)
sample.write_csv("outputs/tables/rating_filter_error_sample.csv")

QD = [f"q{q}" for q in range(1, NQ)]
DRQ = [f"dr_q{q}" for q in range(1, NQ)]
XCOLS = ["const", "L", "dr"] + QD + DRQ


def free_of(cols):
    return np.array([XCOLS.index(c) for c in cols], dtype=int)


# 起始值即完全校準：受限模型把 alpha、beta 固定在起始值時，必須等於虛無假設的值
T0 = np.zeros(len(XCOLS))
T0[1] = 1.0
cell_rows, test_rows, coef_rows = [], [], []
for spd in SPEEDS:
    ts = t.filter(pl.col("speed") == spd)
    for q in range(NQ):
        for res in RESULTS:
            s = ts.filter(pl.col("rdq") == q, pl.col("prev_result") == res)
            a = (s.select(XCOLS).to_numpy().astype(np.float64), s["score"].to_numpy().astype(np.float64))
            th, V, r = lk.lrt_player_dyad(lk.logit_ll, T0, free_of([]), free_of(["const", "L"]), a, s["player"].to_numpy(), s["opponent"].to_numpy())
            cell_rows.append({"speed": spd, "rdq": q + 1, "prev_result": res, "n": s.height, "alpha": th[0], "alpha_se": np.sqrt(V[0, 0]), "beta": th[1], "beta_se": np.sqrt(V[1, 1]), **r})
    a = (ts.select(XCOLS).to_numpy().astype(np.float64), ts["score"].to_numpy().astype(np.float64))
    w_, b_ = ts["player"].to_numpy(), ts["opponent"].to_numpy()
    th, V, r = lk.lrt_player_dyad(lk.logit_ll, T0, free_of(["const", "L"]), free_of(["const", "L", "dr"]), a, w_, b_)
    test_rows.append({"speed": spd, "test": "dr", "n": ts.height, **r})
    coef_rows.append({"speed": spd, "model": "dr", "rdq": 0, "gamma": th[2], "gamma_se": np.sqrt(V[2, 2]), "beta": th[1]})
    # RD 各等分的截距另外估計，dr 的交互作用才不會吸收各等分本身的校準偏差
    f0 = free_of(["const", "L", "dr"] + QD)
    f1 = free_of(["const", "L", "dr"] + QD + DRQ)
    th, V, r = lk.lrt_player_dyad(lk.logit_ll, T0, f0, f1, a, w_, b_)
    test_rows.append({"speed": spd, "test": "dr_x_rd", "n": ts.height, **r})
    # 各等分的 dr 係數：第一等分為 dr 本身，其餘為 dr 加對應的交互作用；共變異取完整模型的 dyadic sandwich
    pos = {c: i for i, c in enumerate(XCOLS[j] for j in f1)}
    for q in range(NQ):
        c = np.zeros(len(f1))
        c[pos["dr"]] = 1
        if q > 0:
            c[pos[f"dr_q{q}"]] = 1
        coef_rows.append({"speed": spd, "model": "dr_x_rd", "rdq": q + 1, "gamma": c @ th[f1], "gamma_se": np.sqrt(c @ V @ c), "beta": th[1]})
    print(spd, "done", flush=True)

cells = pl.DataFrame(cell_rows)
fam = cells["speed"].is_in(["Bullet", "Blitz"]).to_numpy()
ph = np.full(cells.height, np.nan)
ph[fam] = holm(cells["p"].to_numpy()[fam])
cells = cells.with_columns(p_holm=pl.Series(ph))
tests = pl.DataFrame(test_rows)
fam = tests["speed"].is_in(["Bullet", "Blitz"]).to_numpy()
ph = np.full(tests.height, np.nan)
ph[fam] = holm(tests["p"].to_numpy()[fam])
tests = tests.with_columns(p_holm=pl.Series(ph))
# 換算：dr 每 100 分的 logit 偏移，相當於等級分差多少分（以校準後的 logit 斜率 beta Q 換算）；
# 再除以 100 即「上一局的更新中有多少比例是多調的」，負號使降過頭為正
coef = pl.DataFrame(coef_rows).with_columns(
    elo_per_100=pl.col("gamma") / (pl.col("beta") * Q),
    overreaction_share=-pl.col("gamma") / (pl.col("beta") * Q * 100),
    overreaction_share_se=pl.col("gamma_se") / (pl.col("beta") * Q * 100),
)
with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250):
    print(cells.select("speed", "rdq", "prev_result", "n", "alpha", "alpha_se", "beta", "beta_se", "lr", "p", "p_holm"))
    print(tests)
    print(coef)
cells.write_csv("outputs/tables/rating_filter_error_cells.csv")
tests.write_csv("outputs/tables/rating_filter_error_tests.csv")
coef.write_csv("outputs/tables/rating_filter_error_coef.csv")

# 校準曲線：依 e 分組（寬 0.1），平均得分減平均 e
curve = (
    t.filter(pl.col("speed").is_in(["Bullet", "Blitz"]))
    .with_columns(ebin=(pl.col("e_base") * 10).floor().clip(0, 9))
    .group_by("speed", "rdq", "prev_result", "ebin")
    .agg(n=pl.len(), e_mean=pl.col("e_base").mean(), resid_mean=pl.col("resid").mean(), resid_se=pl.col("resid").std() / pl.len().sqrt())
    .filter(pl.col("n") >= 30)
    .sort("speed", "rdq", "prev_result", "ebin")
)
curve.write_csv("outputs/tables/rating_filter_error_curve.csv")
fig, axes = plt.subplots(2, NQ, figsize=(15, 6), sharex=True, sharey=True)
for i, spd in enumerate(["Bullet", "Blitz"]):
    for q in range(NQ):
        ax = axes[i, q]
        for res, mk in zip(RESULTS, ["v", "o", "^"]):
            z = curve.filter(pl.col("speed") == spd, pl.col("rdq") == q, pl.col("prev_result") == res)
            ax.errorbar(z["e_mean"], z["resid_mean"], yerr=1.96 * z["resid_se"], marker=mk, ms=3, capsize=2, lw=1, label=f"previous {res}")
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_title(f"{spd}, RD quintile {q + 1}")
        if i == 1:
            ax.set_xlabel("expected score (main baseline)")
    axes[i, 0].set_ylabel("mean score minus expected")
axes[0, 0].legend(fontsize=8)
fig.tight_layout()
fig.savefig("outputs/figures/rating_filter_error_curve.png", dpi=150)
