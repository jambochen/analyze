import numpy as np
import polars as pl
import pymc as pm

# 「整段全掛天氣」（y2 == 1）是可觀測的，不需要模型去分群。真正不可觀測的是
# 它背後的機制：系統預設值、真實情況、刻意歸因，三者都產生 y2 == 1。
# 混合模型在給定協變數下估各成分的權重與各公司的成分歸屬傾向。
b = (
    pl.read_parquet("data/processed/bad_weather.parquet")
    .filter(pl.col("y2_share").is_not_null())
    .with_columns(
        (pl.col("y2_share") >= 1.0).cast(pl.Int8).alias("is_one"),
        pl.col("sched_dep_utc").dt.truncate("1mo").alias("ym"),
    )
)

# 判別成分的協變數：若 1.0 只是「延誤短、整段自然全掛天氣」的機械結果，
# arr_delay_min 應能解釋掉它；若與天氣嚴重度無關而與公司強相關，則指向
# 預設值或刻意歸因。
feat = ["arr_delay_min", "wx_origin_flight_category_ordinal", "wx_origin_visibility_m"]
b = b.drop_nulls(feat)

carriers = sorted(b["operating_carrier"].unique().to_list())
cidx = b["operating_carrier"].replace_strict({c: i for i, c in enumerate(carriers)}).to_numpy()
y = b["is_one"].to_numpy()
Z = b.select(feat).to_numpy().astype(float)
Z = (Z - Z.mean(0)) / Z.std(0)
print("n", len(y), "公司", len(carriers), "整段全掛比例", round(float(y.mean()), 4))

idatas = {}
# K=3 的抽樣不收斂（divergence 數百、ESS 個位數），已由診斷否決，不再跑。
for K in [2]:
    with pm.Model() as m:
        # 每個成分有自己的基準傾向與對延誤長度、天氣嚴重度的反應。若某成分
        # 對協變數不反應而基準極高，即為「預設值」型態。
        # 成分順序不可識別會使鏈之間標籤互換而 r_hat 爆掉。以 ordered transform
        # 固定基準 logit 遞增，最後一個成分即高傾向者。
        a = pm.Normal("a", 0.0, 2.0, shape=K, transform=pm.distributions.transforms.ordered,
                      initval=np.linspace(-6.0, 2.0, K), default_transform=None)
        beta = pm.Normal("beta", 0.0, 1.0, shape=(K, len(feat)))
        # 公司對成分的歸屬傾向，以 softmax 給出單體權重。階層先驗使樣本少的
        # 公司向整體收縮。
        tau = pm.HalfNormal("tau", 1.0)
        u = pm.Normal("u", 0.0, tau, shape=(len(carriers), K - 1))
        eta = pm.math.concatenate([pm.math.zeros((len(carriers), 1)), u], axis=1)
        w = pm.Deterministic("w", pm.math.softmax(eta, axis=1))
        p = pm.math.sigmoid(a[None, :] + pm.math.dot(Z, beta.T))
        mix = (w[cidx] * p).sum(axis=1)
        pm.Bernoulli("obs", p=mix, observed=y)
        idata = pm.sample(1500, tune=2000, chains=4, cores=4, target_accept=0.95, random_seed=0)
    idatas[K] = idata
    print(f"\n=== K={K}")
    print(pm.stats.summary(idata, var_names=["a", "beta", "tau"]).to_string())

for K, idata in idatas.items():
    post = idata.posterior
    # 成分依基準 logit 遞增排序，最後一個成分為高傾向者。
    wm = post["w"].mean(("chain", "draw")).values
    wl = post["w"].quantile(0.025, ("chain", "draw")).values
    wh = post["w"].quantile(0.975, ("chain", "draw")).values
    n = b.group_by("operating_carrier").agg(pl.len().alias("n"), pl.col("is_one").mean().alias("raw_p_one"))
    d = {"operating_carrier": carriers}
    for k in range(K):
        d[f"w{k + 1}"] = wm[:, k]
    d[f"w{K}_lo"], d[f"w{K}_hi"] = wl[:, K - 1], wh[:, K - 1]
    out = pl.DataFrame(d).join(n, on="operating_carrier").sort(f"w{K}", descending=True)
    print(f"\n=== K={K} 成分權重")
    print(out)
    out.write_csv(f"outputs/tables/mixture_weights_k{K}.csv")

    am = post["a"].mean(("chain", "draw")).values
    bm = post["beta"].mean(("chain", "draw")).values
    print("成分基準 logit", np.round(am, 3))
    for k in range(K):
        print(f"成分{k + 1} 對 {feat}", np.round(bm[k], 3))
