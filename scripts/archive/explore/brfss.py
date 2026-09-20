"""候選 C（BRFSS 美國成人健康風險行為）的篩選用檢視。

僅供判斷此候選是否適合作為選題，非分析結果。
範圍限於 CLAUDE.md 允許的篩選檢視：筆數、欄位型別、缺失率、唯一值數、
描述統計、單變數分布、時間序列折線圖。

本候選的核心賣點是複雜抽樣設計（分層 + 群集 + 權重），故檢視重點為
設計變數是否完整、跨年是否可銜接。不配適任何模型，不做檢定。
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

RAW = Path("data/raw/brfss")
FIGS = Path("outputs/figures/explore")
YEARS = (2011, 2012, 2013, 2014, 2015)

# 複雜抽樣設計變數。BRFSS 2011 起改用 raking 加權，_LLCPWT 為最終個體權重。
DESIGN_VARS = {
    "_STATE": "州 FIPS 碼",
    "_PSU": "主要抽樣單元（群集）",
    "_STSTR": "抽樣分層",
    "_STRWT": "分層權重",
    "_RAWRAKE": "raking 前原始權重",
    "_WT2RAKE": "raking 輸入權重",
    "_LLCPWT": "最終個體權重",
}

# 結果變數候選。皆為 CDC 計算好的風險行為指標。
OUTCOME_VARS = {
    "_RFHLTH": "健康狀況不佳（自評）",
    "_RFSMOK3": "目前吸菸",
    "_RFBING5": "暴飲",
    "_TOTINDA": "休閒身體活動",
    "_RFBMI5": "過重或肥胖",
    "_HCVU651": "18-64 歲有醫療保險",
}


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def year_path(year: int) -> Path:
    return RAW / f"{year}.csv"


def scan(year: int) -> pl.LazyFrame:
    return pl.scan_csv(
        year_path(year), infer_schema_length=20000, ignore_errors=True
    )


def check_shape() -> dict[int, list[str]]:
    """逐年筆數與欄位數。欄位集合跨年變動是本候選的已知前置成本。"""
    section("1. 逐年筆數與欄位數")
    names: dict[int, list[str]] = {}
    rows = []
    for year in YEARS:
        lf = scan(year)
        cols = lf.collect_schema().names()
        names[year] = cols
        n = lf.select(pl.len()).collect().item()
        rows.append({"年": year, "筆數": n, "欄位數": len(cols)})
    print(pl.DataFrame(rows))
    return names


def check_column_stability(names: dict[int, list[str]]) -> set[str]:
    """跨年共同欄位。只有共同欄位能做 5 年趨勢。"""
    section("2. 欄位跨年穩定性")
    sets = {y: set(c) for y, c in names.items()}
    common = set.intersection(*sets.values())
    print(f"五年皆有的欄位數：{len(common)}")
    for year in YEARS:
        only = len(sets[year] - common)
        print(f"  {year}: {len(sets[year])} 欄，其中 {only} 欄非共同")
    return common


def check_design_vars(names: dict[int, list[str]], common: set[str]) -> None:
    """設計變數完整性。缺任一項則加權估計無法做，是本候選的成立前提。"""
    section("3. 複雜抽樣設計變數（本候選核心賣點）")
    rows = []
    for var, desc in DESIGN_VARS.items():
        rows.append(
            {
                "變數": var,
                "說明": desc,
                "有此欄的年數": sum(var in names[y] for y in YEARS),
                "五年皆有": var in common,
            }
        )
    print(pl.DataFrame(rows))


def check_design_quality() -> None:
    """設計變數的缺失與分層規模。權重缺失或分層過細都會破壞加權估計。"""
    section("4. 設計變數的品質（以 2015 為例）")
    lf = scan(2015)
    stats = lf.select(
        pl.len().alias("筆數"),
        pl.col("_LLCPWT").null_count().alias("權重缺失"),
        pl.col("_LLCPWT").min().alias("權重最小"),
        pl.col("_LLCPWT").median().alias("權重中位"),
        pl.col("_LLCPWT").max().alias("權重最大"),
        pl.col("_STSTR").n_unique().alias("分層數"),
        pl.col("_PSU").n_unique().alias("PSU 數"),
        pl.col("_STATE").n_unique().alias("州數"),
    ).collect()
    print(stats)

    # 每層樣本數。單一觀測的層在加權估計中無法估變異數。
    per_stratum = (
        lf.group_by("_STSTR")
        .agg(pl.len().alias("n"))
        .select(
            pl.len().alias("層數"),
            pl.col("n").min().alias("最小層樣本"),
            pl.col("n").median().alias("中位層樣本"),
            pl.col("n").max().alias("最大層樣本"),
            (pl.col("n") == 1).sum().alias("僅1筆的層數"),
        )
        .collect()
    )
    print("\n每個分層的樣本數：")
    print(per_stratum)

    per_state = (
        lf.group_by("_STATE")
        .agg(pl.len().alias("n"))
        .select(
            pl.col("n").min().alias("最小州樣本"),
            pl.col("n").median().alias("中位州樣本"),
            pl.col("n").max().alias("最大州樣本"),
        )
        .collect()
    )
    print("\n每州樣本數（第二層單位，多層次模型用）：")
    print(per_state)


def check_outcomes(common: set[str]) -> list[str]:
    """結果變數的缺失率與分布。缺失率過高者不適合當結果變數。"""
    section("5. 結果變數的可用性（2015）")
    lf = scan(2015)
    available = [v for v in OUTCOME_VARS if v in common]
    missing = [v for v in OUTCOME_VARS if v not in common]
    if missing:
        print(f"非五年共同欄位，排除：{missing}")

    rows = []
    for var in available:
        got = lf.select(
            pl.len().alias("n"),
            pl.col(var).null_count().alias("nulls"),
            pl.col(var).n_unique().alias("uniq"),
        ).collect()
        n, nulls, uniq = got.row(0)
        rows.append(
            {
                "變數": var,
                "說明": OUTCOME_VARS[var],
                "缺失率%": round(100 * nulls / n, 2),
                "唯一值數": uniq,
            }
        )
    print(pl.DataFrame(rows))

    # 編碼分布。BRFSS 慣例 7=不知道 9=拒答，須確認是否已編為缺失。
    print("\n各結果變數的編碼分布（BRFSS 慣例 7/9 為不知道/拒答）：")
    for var in available:
        dist = (
            lf.group_by(var)
            .agg(pl.len().alias("n"))
            .sort(var)
            .collect()
            .with_columns(
                (100 * pl.col("n") / pl.col("n").sum()).round(2).alias("占比%")
            )
        )
        print(f"\n{var}（{OUTCOME_VARS[var]}）:")
        print(dist)
    return available


def check_missingness(common: set[str]) -> None:
    """整體缺失結構。分州選作模組造成的缺失不隨機，是已知風險。"""
    section("6. 缺失率分布（2015，共同欄位）")
    lf = scan(2015)
    cols = sorted(common)
    n = lf.select(pl.len()).collect().item()
    nulls = lf.select(
        [pl.col(c).null_count().alias(c) for c in cols]
    ).collect()
    rates = pl.DataFrame(
        {
            "欄位": cols,
            "缺失率%": [round(100 * nulls[c][0] / n, 2) for c in cols],
        }
    )
    buckets = rates.with_columns(
        pl.when(pl.col("缺失率%") == 0)
        .then(pl.lit("0%"))
        .when(pl.col("缺失率%") < 5)
        .then(pl.lit("<5%"))
        .when(pl.col("缺失率%") < 30)
        .then(pl.lit("5-30%"))
        .when(pl.col("缺失率%") < 70)
        .then(pl.lit("30-70%"))
        .otherwise(pl.lit(">=70%"))
        .alias("區間")
    )
    print(buckets.group_by("區間").agg(pl.len().alias("欄位數")).sort("欄位數", descending=True))
    print("\n缺失率最高的 15 欄（疑為分州選作模組）：")
    print(rates.sort("缺失率%", descending=True).head(15))


def plot_trends(outcomes: list[str]) -> None:
    """結果變數的加權盛行率逐年走勢。折線圖屬允許範圍。"""
    section("7. 逐年趨勢圖")
    FIGS.mkdir(parents=True, exist_ok=True)

    # 盛行率 = 加權後 code 1 的占比。1=無風險 2=有風險（BRFSS _RF 慣例）。
    records = []
    for year in YEARS:
        lf = scan(year)
        cols = lf.collect_schema().names()
        for var in outcomes:
            if var not in cols:
                continue
            got = (
                lf.filter(pl.col(var).is_in([1, 2]))
                .select(
                    (
                        (pl.col("_LLCPWT") * (pl.col(var) == 2)).sum()
                        / pl.col("_LLCPWT").sum()
                        * 100
                    ).alias("pct")
                )
                .collect()
            )
            records.append({"年": year, "變數": var, "加權盛行率%": got.item()})

    trends = pl.DataFrame(records)
    print(trends.pivot(on="變數", index="年", values="加權盛行率%"))

    fig, ax = plt.subplots(figsize=(9, 5))
    for var in outcomes:
        sub = trends.filter(pl.col("變數") == var).sort("年")
        if sub.height:
            ax.plot(sub["年"], sub["加權盛行率%"], marker="o", label=var)
    ax.set_xlabel("年")
    ax.set_ylabel("加權盛行率 %")
    ax.set_title("BRFSS risk behaviour weighted prevalence (screening only)")
    ax.set_xticks(list(YEARS))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIGS / "brfss-trends.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"\n圖已存：{out}")
    print("僅供篩選，非分析結果")


def main() -> int:
    missing = [str(year_path(y)) for y in YEARS if not year_path(y).exists()]
    if missing:
        print(f"缺少資料檔：{missing}", file=sys.stderr)
        return 1

    names = check_shape()
    common = check_column_stability(names)
    check_design_vars(names, common)
    check_design_quality()
    outcomes = check_outcomes(common)
    check_missingness(common)
    if outcomes:
        plot_trends(outcomes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
