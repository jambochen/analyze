"""候選 I（美國航班 + 實測氣象）的篩選用檢視。

僅供判斷此候選是否適合作為選題，非分析結果。
範圍限於 CLAUDE.md 允許的篩選檢視：筆數、欄位型別、缺失率、唯一值數、
描述統計、單變數分布、時間序列折線圖。相關係數矩陣另行標註。

不配適任何模型。分組平均只用於確認資料結構是否存在可用的變異，
不構成處理效應估計。
"""

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import polars as pl

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "raw" / "flights-weather" / "flights_weather_recent_24mo.parquet"
AIRPORTS = ROOT / "data" / "raw" / "flights-weather" / "airports_reference.parquet"
FIGDIR = ROOT / "outputs" / "figures"

for family in ("Noto Sans CJK TC", "Noto Sans CJK JP", "WenQuanYi Zen Hei"):
    if any(family in f.name for f in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = family
        break
plt.rcParams["axes.unicode_minus"] = False


def section(title: str) -> None:
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def prefix_of(col: str) -> str:
    for p in ("wx_", "oracle_", "reported_"):
        if col.startswith(p):
            return p
    return "(其他)"


def main() -> int:
    lf = pl.scan_parquet(SRC)
    schema = lf.collect_schema()
    cols = schema.names()

    section("1. 規模與欄位")
    n_rows = lf.select(pl.len()).collect().item()
    print(f"列數：{n_rows:,}")
    print(f"欄數：{len(cols)}")
    print(f"檔案大小：{SRC.stat().st_size / 1e9:.2f} GB")

    section("2. 欄位前綴分布（資料集的洩漏防護設計）")
    groups: dict[str, list[str]] = {}
    for c in cols:
        groups.setdefault(prefix_of(c), []).append(c)
    for p, members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        print(f"\n{p}（{len(members)} 欄）")
        for c in members:
            print(f"  {c:46s} {schema[c]}")

    section("3. 缺失率與唯一值數")
    nulls = lf.select(
        [pl.col(c).null_count().alias(c) for c in cols]
    ).collect().row(0)
    # 唯一值數對 4,280 萬列成本過高，僅取類別型與低基數候選欄位
    cat_cols = [c for c in cols if schema[c] in (pl.Utf8, pl.Categorical, pl.Boolean)]
    uniques = lf.select(
        [pl.col(c).n_unique().alias(c) for c in cat_cols]
    ).collect().row(0)
    uniq_map = dict(zip(cat_cols, uniques))
    print(f"{'欄位':46s} {'缺失率':>9s} {'唯一值':>12s}")
    for c, nn in sorted(zip(cols, nulls), key=lambda kv: -kv[1]):
        u = uniq_map.get(c)
        print(f"{c:46s} {nn / n_rows:9.2%} {('' if u is None else f'{u:,}'):>12s}")

    section("4. 主要數值欄位的描述統計")
    num_cols = [
        c for c in cols
        if schema[c].is_numeric() and not c.endswith("_id")
    ]
    # 211 欄橫排會爆版，改為每欄一列
    stats = lf.select(
        [
            pl.struct(
                pl.col(c).mean().alias("mean"),
                pl.col(c).std().alias("std"),
                pl.col(c).min().alias("min"),
                pl.col(c).quantile(0.25).alias("q25"),
                pl.col(c).median().alias("median"),
                pl.col(c).quantile(0.75).alias("q75"),
                pl.col(c).max().alias("max"),
            ).alias(c)
            for c in num_cols
        ]
    ).collect().row(0, named=True)
    header = f"{'欄位':44s} {'平均':>12s}{'標準差':>12s}{'最小':>12s}{'q25':>12s}{'中位':>12s}{'q75':>12s}{'最大':>12s}"
    print(header)
    for c, st in stats.items():
        vals = "".join(
            f"{v:>12.2f}" if isinstance(v, (int, float)) else f"{'':>12s}"
            for v in (st["mean"], st["std"], st["min"], st["q25"],
                      st["median"], st["q75"], st["max"])
        )
        print(f"{c:44s} {vals}")

    section("5. 時間跨度與每月航班量")
    date_col = next(
        (c for c in ("flight_date_local_origin", "sched_dep_utc") if c in cols),
        None,
    )
    if date_col is None:
        print("找不到日期欄位，請檢查 data_dictionary.md")
    else:
        monthly = (
            lf.select(pl.col(date_col).cast(pl.Date).dt.truncate("1mo").alias("ym"))
            .group_by("ym").len().sort("ym").collect()
        )
        print(f"日期欄位：{date_col}")
        print(f"起訖：{monthly['ym'].min()} 至 {monthly['ym'].max()}")
        with pl.Config(tbl_rows=-1):
            print(monthly)
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(monthly["ym"], monthly["len"])
        ax.set_title("候選 I：每月航班量（僅供篩選，非分析結果）")
        ax.set_ylabel("航班數")
        fig.tight_layout()
        fig.savefig(FIGDIR / "explore_flights_monthly.png", dpi=120)
        plt.close(fig)

    section("6. 結構可衍生性檢查")
    for label, col in (("機尾號（追蹤結構）", "tail_number"),
                       ("航空公司", "operating_carrier"),
                       ("起點機場", "origin_iata"),
                       ("終點機場", "dest_iata")):
        if col in cols:
            print(f"{label:20s} {col:24s} 唯一值 {uniq_map.get(col, '需另計'):,}"
                  if isinstance(uniq_map.get(col), int)
                  else f"{label:20s} {col:24s}")
    if {"origin_iata", "dest_iata"} <= set(cols):
        n_routes = lf.select(
            pl.concat_str([pl.col("origin_iata"), pl.col("dest_iata")], separator="-").n_unique()
        ).collect().item()
        print(f"航線（origin-dest 組合）唯一值：{n_routes:,}")
    if "tail_number" in cols:
        legs = (
            lf.filter(pl.col("tail_number").is_not_null())
            .group_by("tail_number").len().collect()["len"]
        )
        print(f"每架機尾號的航段數：中位數 {legs.median():,.0f}、"
              f"平均 {legs.mean():,.1f}、最大 {legs.max():,}")

    section("7. 延誤分布（單變數）")
    delay_col = next(
        (c for c in ("arr_delay_min", "dep_delay_min") if c in cols), None
    )
    if delay_col:
        s = lf.select(pl.col(delay_col)).collect()[delay_col].drop_nulls()
        qs = [0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        print(f"欄位：{delay_col}")
        for q in qs:
            print(f"  q{q:>5.2f}  {s.quantile(q):8.1f} 分")
        fig, ax = plt.subplots(figsize=(9, 4))
        # 長尾極端值會壓扁圖形，截在 1% 與 99% 分位之間看主體形狀
        ax.hist(s.filter(s.is_between(s.quantile(0.01), s.quantile(0.99))),
                bins=120)
        ax.set_title(f"候選 I：{delay_col} 分布（僅供篩選，非分析結果）")
        ax.set_xlabel("分鐘")
        fig.tight_layout()
        fig.savefig(FIGDIR / "explore_flights_delay_hist.png", dpi=120)
        plt.close(fig)

    section("8. 飛航天氣等級的樣本量（僅供篩選，非分析結果）")
    cat_col = next((c for c in cols if "flight_category" in c or "flight_cat" in c), None)
    if cat_col and delay_col:
        tab = (
            lf.group_by(cat_col)
            .agg(pl.len().alias("n"), pl.col(delay_col).mean().alias("平均延誤分"))
            .sort("n", descending=True).collect()
        )
        print(f"分組欄位：{cat_col}")
        with pl.Config(tbl_rows=-1):
            print(tab)
        print("此表僅確認各等級樣本量是否足夠，不構成處理效應估計。")

    section("9. 回報偏誤素材的可用性")
    rep_cols = [c for c in cols if c.startswith("reported_")]
    if rep_cols and delay_col:
        print(f"reported_ 欄位共 {len(rep_cols)} 個：{rep_cols}")
        nonzero = lf.select(
            [(pl.col(c) > 0).sum().alias(c) for c in rep_cols
             if schema[c].is_numeric()]
        ).collect()
        print("各 reported_ 欄位大於零的列數：")
        with pl.Config(tbl_cols=-1, tbl_width_chars=200):
            print(nonzero)

    section("10. 機場對照表")
    if AIRPORTS.exists():
        ar = pl.read_parquet(AIRPORTS)
        print(f"列數 {len(ar):,}、欄數 {ar.width}")
        with pl.Config(tbl_cols=-1, tbl_width_chars=200):
            print(ar.head(5))

    print("\n以上全部為篩選用檢視，非分析結果。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
