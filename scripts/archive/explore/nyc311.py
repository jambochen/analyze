"""候選 K（NYC 311 服務通報）的篩選用檢視。

僅供判斷此候選是否適合作為選題，非分析結果。
範圍限於 CLAUDE.md 允許的篩選檢視：筆數、欄位型別、缺失率、唯一值數、
描述統計、單變數分布、時間序列折線圖。

處理時間的分位數與設限比例屬描述統計，用於確認生存分析素材是否存在。
不配適任何存活模型，不做檢定。
"""

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import polars as pl

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "raw" / "nyc311"
ROWS = SRC / "requests_2025.parquet"
FIGDIR = ROOT / "outputs" / "figures"

for family in ("Noto Sans CJK TC", "Noto Sans CJK JP", "WenQuanYi Zen Hei"):
    if any(family in f.name for f in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = family
        break
plt.rcParams["axes.unicode_minus"] = False

FMT = "%Y-%m-%dT%H:%M:%S%.f"


def section(title: str) -> None:
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def load() -> pl.DataFrame:
    df = pl.read_parquet(ROWS)
    return df.with_columns(
        *[
            pl.col(c).str.strptime(pl.Datetime, FMT, strict=False)
            for c in ("created_date", "closed_date", "due_date")
        ],
        *[
            pl.col(c).cast(pl.Float64, strict=False)
            for c in ("latitude", "longitude")
        ],
    )


def show(df: pl.DataFrame, rows: int = -1) -> None:
    with pl.Config(tbl_rows=rows, tbl_cols=-1, tbl_width_chars=200):
        print(df)


def main() -> int:
    section("0. 聚合摘要（Socrata SoQL，涵蓋 2020 至今全量 2,250 萬列）")
    for path in sorted(SRC.glob("agg_*.csv")):
        agg = pl.read_csv(path)
        print(f"\n--- {path.stem} （{len(agg)} 組）---")
        show(agg.head(20), rows=20)
        if path.stem == "agg_by_month":
            ym = agg.with_columns(pl.col("ym").str.strptime(pl.Datetime, FMT, strict=False))
            fig, ax = plt.subplots(figsize=(11, 4))
            ax.plot(ym["ym"], ym["n"], label="通報量")
            ax.plot(ym["ym"], ym["n_closed"], label="已結案")
            ax.legend()
            ax.set_title("候選 K：每月 311 通報量（僅供篩選，非分析結果）")
            fig.tight_layout()
            fig.savefig(FIGDIR / "explore_nyc311_monthly.png", dpi=120)
            plt.close(fig)

    df = load()

    section("1. 列級樣本的規模與欄位型別（2025 全年）")
    print(f"列數：{len(df):,}")
    print(f"欄數：{df.width}")
    for c, dt in df.schema.items():
        print(f"  {c:28s} {dt}")

    section("2. 缺失率與唯一值數")
    print(f"{'欄位':28s} {'缺失率':>9s} {'唯一值':>12s}")
    for c in df.columns:
        print(f"{c:28s} {df[c].null_count() / len(df):9.2%} {df[c].n_unique():12,}")

    section("3. 生存分析素材：設限與處理時間")
    df = df.with_columns(
        (pl.col("closed_date") - pl.col("created_date"))
        .dt.total_seconds().truediv(3600).alias("hours_to_close"),
        (pl.col("due_date") - pl.col("created_date"))
        .dt.total_seconds().truediv(3600).alias("hours_allowed"),
    )
    n_censored = df["closed_date"].null_count()
    print(f"右設限（未結案）：{n_censored:,} 列，{n_censored / len(df):.2%}")
    print(f"有官方期限 due_date：{df['due_date'].is_not_null().sum():,} 列，"
          f"{df['due_date'].is_not_null().mean():.2%}")
    # 結案早於通報必為資料錯誤，先看數量再決定是否構成排除理由
    n_neg = (df["hours_to_close"] < 0).sum()
    print(f"結案時間早於通報時間：{n_neg:,} 列")

    ht = df.filter(pl.col("hours_to_close") >= 0)["hours_to_close"]
    print("\n處理時間（小時）分位數：")
    for q in (0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99):
        print(f"  q{q:>5.2f}  {ht.quantile(q):10.2f}")
    print(f"  平均    {ht.mean():10.2f}   標準差 {ht.std():10.2f}   最大 {ht.max():10.2f}")

    fig, ax = plt.subplots(figsize=(9, 4))
    # 處理時間右尾極長，取對數才看得出主體形狀
    ax.hist(ht.filter(ht > 0).log10(), bins=120)
    ax.set_xlabel("log10(處理小時數)")
    ax.set_title("候選 K：處理時間分布（僅供篩選，非分析結果）")
    fig.tight_layout()
    fig.savefig(FIGDIR / "explore_nyc311_duration_hist.png", dpi=120)
    plt.close(fig)

    section("4. 逾期情形（與官方 due_date 相比）")
    dued = df.filter(pl.col("hours_allowed").is_not_null())
    print(f"有期限的案件：{len(dued):,} 列")
    if len(dued):
        overdue = (dued["hours_to_close"] > dued["hours_allowed"]).sum()
        print(f"其中已結案且超過期限：{overdue:,}（{overdue / len(dued):.2%}）")

    section("5. 多層次結構：各層級的群組數與每群樣本量")
    for c in ("agency", "complaint_type", "descriptor", "borough",
              "community_board", "council_district", "incident_zip"):
        g = df.group_by(c).len()["len"]
        print(f"{c:20s} 群組數 {len(g):>7,}   每群中位數 {g.median():>9,.0f}   "
              f"最小 {g.min():>7,}   最大 {g.max():>9,}")

    section("6. 交叉分類的檢查：community_board × council_district")
    cross = df.select("community_board", "council_district").drop_nulls().unique()
    print(f"實際出現的組合數：{len(cross):,}")
    print("若組合數明顯多於任一層級的群組數，代表兩層交錯而非巢狀，"
          "適用交叉分類隨機效果模型。")

    section("7. 各機關的處理時間（僅供篩選，非分析結果）")
    by_agency = (
        df.filter(pl.col("hours_to_close") >= 0)
        .group_by("agency")
        .agg(
            pl.len().alias("n"),
            pl.col("hours_to_close").median().alias("中位處理小時"),
            pl.col("hours_to_close").quantile(0.9).alias("q90"),
        )
        .sort("n", descending=True).head(15)
    )
    show(by_agency, rows=15)
    print("此表僅確認機關間是否存在可建模的變異，不構成機關績效比較。")

    section("8. 通報管道與案件狀態")
    show(df.group_by("open_data_channel_type").len().sort("len", descending=True))
    show(df.group_by("status").len().sort("len", descending=True), rows=20)

    section("9. 每日通報量時間序列")
    daily = (
        df.select(pl.col("created_date").dt.date().alias("d"))
        .group_by("d").len().sort("d")
    )
    print(f"日期範圍：{daily['d'].min()} 至 {daily['d'].max()}，共 {len(daily)} 天")
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(daily["d"], daily["len"], linewidth=0.8)
    ax.set_title("候選 K：2025 年每日通報量（僅供篩選，非分析結果）")
    fig.tight_layout()
    fig.savefig(FIGDIR / "explore_nyc311_daily.png", dpi=120)
    plt.close(fig)

    print("\n以上全部為篩選用檢視，非分析結果。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
