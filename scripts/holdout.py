import polars as pl
from scipy import stats

import stratum

b = stratum.load()


def estimate(d, outcome):
    return stratum.fit(d, outcome).filter(pl.col("term").str.starts_with("carrier_")).select("term", "coef", "se")


# 監理清單若年度間翻盤即不可用。前段估計、後段驗證。
train, test = b.filter(pl.col("yr") <= 2021), b.filter(pl.col("yr") >= 2022)
print("train", train.height, "test", test.height)

for outcome in ["y1_attributed", "y2_share", "y3_wx_vs_nas"]:
    a = estimate(train, outcome).rename({"coef": "coef_train", "se": "se_train"})
    c = estimate(test, outcome).rename({"coef": "coef_test", "se": "se_test"})
    j = a.join(c, on="term", how="inner").with_columns(
        (pl.col("coef_train") - pl.col("coef_test")).alias("diff"),
        ((pl.col("coef_train") - pl.col("coef_test")) / (pl.col("se_train") ** 2 + pl.col("se_test") ** 2).sqrt()).alias("z"),
    ).sort("coef_train", descending=True)
    rho = stats.spearmanr(j["coef_train"], j["coef_test"])
    # 排序一致性才是清單能否採信的判準，係數本身相等與否是次要的。
    print(f"\n=== {outcome} 兩段共同公司 {j.height}")
    print(f"排序 spearman {rho.statistic:.4f} (p={rho.pvalue:.4f})")
    print(f"係數差的 |z| 中位 {j['z'].abs().median():.2f}  |z|>1.96 者 {j.filter(pl.col('z').abs() > 1.96).height}")
    print(j.select("term", "coef_train", "coef_test", "diff", "z"))
    j.write_csv(f"outputs/tables/holdout_{outcome.split('_')[0]}.csv")

# 清單的實際用途是「該優先查誰」，故檢查前段的高位者在後段是否仍在高位。
y1a = estimate(train, "y1_attributed").sort("coef", descending=True)
y1b = estimate(test, "y1_attributed").sort("coef", descending=True)
for k in [3, 5]:
    ta, tb = set(y1a.head(k)["term"]), set(y1b.head(k)["term"])
    print(f"\nY1 前 {k} 名重疊 {len(ta & tb)}/{k}: 前段 {sorted(x.replace('carrier_', '') for x in ta)}"
          f" 後段 {sorted(x.replace('carrier_', '') for x in tb)}")
