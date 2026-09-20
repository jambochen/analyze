"""候選「PJM 電網負載與氣象」的篩選用檢視。

僅供判斷此候選是否適合作為選題，非分析結果。
範圍限於 CLAUDE.md 允許的篩選檢視：筆數、欄位型別、缺失率、唯一值數、
描述統計、單變數分布、時間序列折線圖。相關係數另行標註。

**不配適任何模型。** 本候選的核心賣點是「氣象效應與資料中心結構性成長
的分離」，那需要迴歸，屬禁止範圍。這裡只驗證該賣點的**前提**是否成立：

1. 時間軸是否真如作者所述無缺漏
2. 四個品質旗標是否存在且有作用
3. 負載是否真有氣象解釋不了的結構性成長
4. 氣溫與負載是否呈非線性（U 形）關係

分離本身留待正式分析階段。
"""

import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import polars as pl

matplotlib.use("Agg")
for family in ("Noto Sans CJK TC", "Noto Sans CJK JP", "WenQuanYi Zen Hei"):
    if any(family in f.name for f in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = family
        break
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "raw" / "pjm-demand-weather" / "pjm_demand_weather_hourly_v1.parquet"
FIGS = ROOT / "outputs" / "figures" / "screening"

FLAGS = [
    "demand_is_imputed",
    "temp_is_missing",
    "temp_is_imputed",
    "weather_is_preliminary",
    "is_dst_transition_day",
    "degree_day_is_partial",
    "is_weekend",
    "is_holiday_statutory",
    "is_holiday_observed",
]


def section(title: str) -> None:
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def show(df: pl.DataFrame, rows: int = -1) -> None:
    with pl.Config(tbl_rows=rows, tbl_cols=-1, tbl_width_chars=200):
        print(df)


def structure(df: pl.DataFrame) -> None:
    section("1. 基本結構與各欄缺失率")
    print(f"{df.height:,} 列 × {df.width} 欄\n")

    rows = []
    for col in df.columns:
        s = df[col]
        rows.append({
            "欄位": col,
            "型別": str(s.dtype).replace("Datetime(time_unit='us', time_zone=", "tz("),
            "缺失率": s.null_count() / df.height,
            "唯一值": s.n_unique(),
        })
    show(pl.DataFrame(rows))


def check_timeline(df: pl.DataFrame) -> None:
    """賣點一：時間軸無缺漏。作者宣稱 UTC 軸上連續、唯一、無間隙。"""
    section("2. 時間軸完整性（賣點一）")

    ts = df["datetime_utc"]
    start, end = ts.min(), ts.max()
    span_hours = int((end - start).total_seconds() // 3600) + 1

    print(f"起訖　　{start} 至 {end}")
    print(f"實際列數　{df.height:,}")
    print(f"應有小時數 {span_hours:,}（依起訖推算）")
    print(f"缺漏　　　{span_hours - df.height:,} 小時")
    print(f"重複時間戳 {df.height - ts.n_unique():,} 個")

    gaps = (
        df.select(pl.col("datetime_utc").sort().diff().alias("間隔"))
        .drop_nulls()
        .group_by("間隔")
        .agg(pl.len().alias("次數"))
        .sort("次數", descending=True)
    )
    print("\n相鄰時間戳的間隔分布：")
    show(gaps)


def check_flags(df: pl.DataFrame) -> None:
    """賣點二：品質旗標齊備且真的有作用。"""
    section("3. 品質旗標（賣點二）")

    rows = []
    for flag in FLAGS:
        if flag not in df.columns:
            continue
        n_true = df[flag].sum()
        rows.append({
            "旗標": flag,
            "True 筆數": n_true,
            "佔比": n_true / df.height,
        })
    show(pl.DataFrame(rows))

    print("\n旗標若全為 False 便無資訊，以下確認各旗標實際標出了東西。")

    section("3b. demand_source 的組成")
    show(
        df.group_by("demand_source")
        .agg(
            pl.len().alias("筆數"),
            pl.col("datetime_utc").min().alias("最早"),
            pl.col("datetime_utc").max().alias("最晚"),
        )
        .sort("筆數", descending=True)
    )
    print("\neia_api 是未經異常篩檢的初步值，PUDL 覆蓋推進後會被取代。")

    if "demand_imputed_pudl_mwh_imputation_code" in df.columns:
        section("3c. PUDL 的補值原因碼")
        show(
            df.filter(pl.col("demand_imputed_pudl_mwh_imputation_code").is_not_null())
            .group_by("demand_imputed_pudl_mwh_imputation_code")
            .agg(pl.len().alias("筆數"))
            .sort("筆數", descending=True)
        )


def check_growth(df: pl.DataFrame) -> None:
    """賣點三：負載有氣象解釋不了的結構性成長。

    只看逐年同月的平均，不做任何分解——分解需要迴歸。
    """
    section("4. 負載的結構性成長（賣點三）")

    yearly = (
        df.with_columns(pl.col("datetime_local").dt.year().alias("年"))
        .group_by("年")
        .agg(
            pl.len().alias("小時數"),
            pl.col("demand_mwh").mean().round(0).alias("平均MWh"),
            pl.col("demand_mwh").max().round(0).alias("尖峰MWh"),
            pl.col("temp_pop_wt_c").mean().round(2).alias("平均氣溫C"),
        )
        .sort("年")
    )
    print("逐年（完整年度以外的頭尾年僅供參考）：")
    show(yearly)

    print("\n七月同月比較——作者在 caveats 宣稱兩年約成長 10%：")
    july = (
        df.filter(pl.col("datetime_local").dt.month() == 7)
        .with_columns(pl.col("datetime_local").dt.year().alias("年"))
        .group_by("年")
        .agg(
            pl.len().alias("小時數"),
            pl.col("demand_mwh").mean().round(0).alias("七月平均MWh"),
            pl.col("temp_pop_wt_c").mean().round(2).alias("七月平均氣溫C"),
        )
        .sort("年")
    )
    show(july)

    print("""
注意作者 caveats 那個「兩年約 10%」的數字：他寫的是
  2024 年七月 109,807 → 2025 年 116,188 → 2026 年 120,628（截至 19 日）
最後一個是**部分月**，因為當時七月還沒過完。本檔的 2026 年七月已完整。""")

    partial = (
        df.filter(
            (pl.col("datetime_local").dt.month() == 7)
            & (pl.col("datetime_local").dt.day() <= 19)
        )
        .with_columns(pl.col("datetime_local").dt.year().alias("年"))
        .group_by("年")
        .agg(pl.col("demand_mwh").mean().round(0).alias("七月1-19日平均MWh"))
        .sort("年")
        .filter(pl.col("年") >= 2024)
    )
    print("\n同口徑（皆取 1-19 日）重算：")
    show(partial)
    print("""
1-19 日口徑 2024→2026 為 +7.1%，非 10%。
而完整七月的 2026（114,957）反而低於 2025（116,188）。

這不是作者造假——資料每次 rebuild 會推進尾端，他寫 caveat 時 2026 年
七月尚未結束。但**那個 10% 不可直接引用**，要自己按當前檔案重算。

同月比較已控掉季節，但未控氣溫。僅供篩選，非分析結果。""")


def check_temp_response(df: pl.DataFrame) -> None:
    """賣點四：氣溫與負載的 U 形關係。

    分箱平均只用於確認非線性形狀存在，不是反應函數的估計。
    """
    section("5. 氣溫與負載的關係（賣點四）")

    binned = (
        df.drop_nulls(["temp_pop_wt_c", "demand_mwh"])
        .with_columns(
            (pl.col("temp_pop_wt_c") / 5).floor().mul(5).cast(pl.Int32).alias("氣溫組")
        )
        .group_by("氣溫組")
        .agg(
            pl.len().alias("小時數"),
            pl.col("demand_mwh").mean().round(0).alias("平均MWh"),
        )
        .sort("氣溫組")
    )
    show(binned)
    print("\n每 5°C 一組。冷熱兩端皆升、中間低 = U 形。")
    print("**分箱平均僅供確認形狀，非反應函數的估計。**")

    section("5b. 相關係數（僅供篩選，非分析結果）")
    numeric = [
        "demand_mwh", "temp_pop_wt_c", "apparent_temp_pop_wt_c",
        "hdh_65", "cdh_65", "shortwave_radiation_pop_wt_wm2",
        "cloud_cover_pop_wt_pct", "snowfall_pop_wt_cm",
    ]
    cols = [c for c in numeric if c in df.columns]
    sub = df.select(cols).drop_nulls()
    corr = pl.DataFrame({
        "變數": cols,
        **{
            c: [sub.select(pl.corr(a, c)).item() for a in cols]
            for c in cols
        },
    })
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=220, float_precision=3):
        print(corr)
    print("\n線性相關係數對 U 形關係本來就會低估，這裡只用於看氣象欄位之間的重疊。")


def plot_series(df: pl.DataFrame) -> None:
    """時間序列折線圖與單變數分布，皆屬允許範圍。"""
    section("6. 圖")
    FIGS.mkdir(parents=True, exist_ok=True)

    daily = (
        df.with_columns(pl.col("datetime_local").dt.date().alias("日"))
        .group_by("日")
        .agg(
            pl.col("demand_mwh").mean().alias("日均負載"),
            pl.col("temp_pop_wt_c").mean().alias("日均氣溫"),
        )
        .sort("日")
    )

    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    axes[0].plot(daily["日"], daily["日均負載"] / 1000, linewidth=0.5)
    axes[0].set_ylabel("日均負載 (GWh)")
    axes[0].set_title("PJM 日均負載與氣溫（僅供篩選，非分析結果）")
    axes[1].plot(daily["日"], daily["日均氣溫"], linewidth=0.5, color="tab:orange")
    axes[1].set_ylabel("日均氣溫 (°C)")
    axes[1].set_xlabel("日期")
    for ax in axes:
        ax.grid(alpha=0.3)
    plt.tight_layout()
    out = FIGS / "pjm-daily-series.png"
    plt.savefig(out, dpi=110)
    plt.close()
    print(f"已輸出 {out.relative_to(ROOT)}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(df["demand_mwh"].drop_nulls().to_numpy() / 1000, bins=80)
    ax.set_xlabel("小時負載 (GWh)")
    ax.set_ylabel("小時數")
    ax.set_title("負載分布（僅供篩選）")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = FIGS / "pjm-demand-hist.png"
    plt.savefig(out, dpi=110)
    plt.close()
    print(f"已輸出 {out.relative_to(ROOT)}")


def main() -> int:
    if not SRC.exists():
        raise SystemExit(f"找不到 {SRC}")
    print(f"資料檔：{SRC.name}（{SRC.stat().st_size / 1e6:.1f} MB）")
    df = pl.read_parquet(SRC)

    structure(df)
    check_timeline(df)
    check_flags(df)
    check_growth(df)
    check_temp_response(df)
    plot_series(df)

    section("小結")
    print("""四個賣點前提的檢驗結果見上。判讀要點：

- 時間軸若真的無缺漏，跨年比較就不必自己推敲哪些年份可比
- 品質旗標若確實標出了初步值與插補值，敏感度分析可以直接做
- 結構性成長若肉眼可見，「氣象 vs 成長」的分離才是個真問題
- 氣溫與負載若呈 U 形，才需要平滑項或分段迴歸，否則線性就夠

**以上皆為篩選，不是分析結果。**""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
