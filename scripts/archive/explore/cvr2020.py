"""候選 J（2020 美國大選逐張選票紀錄）的篩選用檢視。

僅供判斷此候選是否適合作為選題，非分析結果。
範圍限於 CLAUDE.md 允許的篩選檢視：筆數、欄位型別、缺失率、唯一值數、
描述統計、單變數分布、時間序列折線圖。

本候選的核心賣點是「個體層級 × 多重選擇」結構，故檢視重點為該結構是否
真實存在、分裂投票是否可觀測、20 州涵蓋範圍的偏斜程度。
交叉表屬列聯表的描述統計，用於確認結構存在，不配適對數線性模型、
不做卡方檢定、不做潛在類別分析。
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

RAW = Path("data/raw/cvr2020/release")
FIGS = Path("outputs/figures/explore")

# 官方 usage-notes.R 用的 5 個搖擺州，沿用以便與資料描述論文對照。
SWING = ["ARIZONA", "GEORGIA", "MICHIGAN", "NEVADA", "WISCONSIN"]


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def scan() -> pl.LazyFrame:
    # TEXAS/OLDHAM 多出 ballot_style_raw 與 ballot_style 兩欄，全庫 362 檔中僅此一檔。
    # 不指定 extra_columns 會讓 scan 直接拋 SchemaError。
    return pl.scan_parquet(
        RAW / "**/*.parquet", hive_partitioning=True, extra_columns="ignore"
    )


def check_schema() -> list[str]:
    """欄位與型別。分區鍵 state/county_name 由目錄名推得，須確認確實出現。"""
    section("1. 欄位與型別")
    lf = scan()
    schema = lf.collect_schema()
    for name, dtype in schema.items():
        print(f"  {name:16s} {dtype}")
    return schema.names()


def check_size() -> int:
    """總列數。論文宣稱 166,470,734 列，此處驗證。"""
    section("2. 總列數（論文宣稱 166,470,734）")
    n = scan().select(pl.len()).collect(engine="streaming").item()
    print(f"實測列數：{n:,}")
    print(f"與論文差異：{n - 166_470_734:,}")
    return n


def check_coverage() -> pl.DataFrame:
    """州與郡的涵蓋範圍。20 州非隨機，此為已知的外部效度限制。"""
    section("3. 州與郡涵蓋範圍")
    per_state = (
        scan()
        .group_by("state")
        .agg(
            pl.len().alias("列數"),
            pl.col("county_name").n_unique().alias("郡數"),
        )
        .sort("列數", descending=True)
        .collect(engine="streaming")
    )
    print(per_state)
    print(f"\n州數：{per_state.height}，郡數合計：{per_state['郡數'].sum()}")
    return per_state


def check_voters() -> None:
    """選民數與每票列數。

    cvr_id 僅郡內唯一，故識別選民須用 (state, county_name, cvr_id) 三欄組合。
    若誤用單一 cvr_id 會把不同郡的選民混為一人。

    逐郡累計而非一次 group_by 全庫。瓶頸是**分組數**而非列數：此 key 產生
    4,620 萬個分組，group_by 的 hash table 須容納全部唯一 key 才能輸出，
    這一步無法串流（pipeline breaker），`sink_parquet` 同樣失敗。
    實測 16 GB 上限下爆掉。對照 `candidate` 僅 4,502 組，串流峰值 0.3 GB。

    分郡在此不是近似：cvr_id 僅郡內唯一，跨郡 key 本就不會相遇，
    分郡聚合與全庫聚合結果完全相同。
    """
    section("4. 選民數與「個體 × 多重選擇」結構")

    wrong = 0
    right = 0
    n_choices_hist: dict[int, int] = {}
    # 目錄名為 URL 編碼（NEW%20JERSEY），polars 解碼後才是 state 欄的值，
    # 故不能用欄位值直接組路徑，改為逐目錄走檔案系統。
    for state_dir in sorted(RAW.iterdir()):
        for county_dir in sorted(state_dir.iterdir()):
            counts = (
                pl.scan_parquet(
                    county_dir / "*.parquet", extra_columns="ignore"
                )
                .group_by("cvr_id")
                .agg(pl.len().alias("n"))
                .collect(engine="streaming")
            )
            right += counts.height
            for n, c in (
                counts.group_by("n").agg(pl.len().alias("c")).iter_rows()
            ):
                n_choices_hist[n] = n_choices_hist.get(n, 0) + c

    # 全庫僅用 cvr_id 去重的實際值，單欄 unique 記憶體可負擔
    wrong = (
        scan().select(pl.col("cvr_id").n_unique()).collect(engine="streaming").item()
    )

    print(f"僅用 cvr_id 去重（錯誤做法）：{wrong:,}")
    print(f"用 (state, county_name, cvr_id) 去重：{right:,}")
    print("論文宣稱選民數：42,700,000（約）")
    print(f"→ cvr_id 郡內唯一已確認，錯誤做法低估 {right / wrong:.1f}x")

    hist = pl.DataFrame(
        {
            "n_choices": list(n_choices_hist.keys()),
            "ballots": list(n_choices_hist.values()),
        }
    ).sort("n_choices")
    total = hist["ballots"].sum()
    weighted = (hist["n_choices"] * hist["ballots"]).sum()
    print("\n每張選票的選擇筆數（多重選擇結構的規模）：")
    print(f"  最少 {hist['n_choices'].min()}，最多 {hist['n_choices'].max()}，"
          f"平均 {weighted / total:.2f}")
    print(hist.head(15))


def check_offices() -> pl.DataFrame:
    """選舉項目分布。多層級是分裂投票分析的前提。"""
    section("5. 選舉項目（office）分布")
    by_office = (
        scan()
        .group_by("office")
        .agg(
            pl.len().alias("列數"),
            pl.col("state").n_unique().alias("州數"),
        )
        .sort("列數", descending=True)
        .collect(engine="streaming")
    )
    print(by_office)
    return by_office


def check_candidate_codes() -> None:
    """candidate 的特殊值。UNDERVOTE/OVERVOTE 為有效碼而非缺失，須先確認占比。"""
    section("6. candidate 特殊值與 party 分布")
    lf = scan()

    # candidate 僅 4,502 個唯一值，分組數小，串流 group_by 峰值 0.3 GB。
    cands = (
        lf.filter(pl.col("office") == "US PRESIDENT")
        .group_by("candidate", "party")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .collect(engine="streaming")
    )
    print("總統選舉的候選人（前 12）：")
    print(cands.head(12))

    total = cands["n"].sum()
    special = (
        cands.filter(
            pl.col("candidate").str.contains("(?i)undervote|overvote|write")
        )
        .group_by("candidate")
        .agg(pl.col("n").sum())
        .sort("n", descending=True)
        .with_columns((100 * pl.col("n") / total).round(3).alias("占比%"))
    )
    print("\n特殊值（非真實候選人，不可當缺失值丟棄）：")
    print(special)

    parties = (
        lf.group_by("party")
        .agg(pl.len().alias("n"))
        .sort("n", descending=True)
        .collect(engine="streaming")
    )
    print("\nparty 分布（前 10）：")
    print(parties.head(10))


def check_nulls(cols: list[str]) -> None:
    """缺失率。"""
    section("7. 各欄缺失率")
    # 分區鍵 state/county_name 來自目錄名，永不為 null。
    data_cols = [c for c in cols if c not in ("state", "county_name")]
    lf = scan()
    n = lf.select(pl.len()).collect(engine="streaming").item()
    nulls = lf.select(
        [pl.col(c).null_count().alias(c) for c in data_cols]
    ).collect(engine="streaming")
    print(
        pl.DataFrame(
            {
                "欄位": data_cols,
                "缺失率%": [round(100 * nulls[c][0] / n, 3) for c in data_cols],
            }
        )
    )


def check_split_ticket() -> None:
    """分裂投票的可觀測性。

    交叉表為列聯表描述統計，用於確認「總統選擇 × 下游選擇」結構存在。
    不做檢定、不配適模型。僅取搖擺州以控制記憶體。
    """
    section("8. 分裂投票結構是否可觀測（僅搖擺州）")
    print("僅供篩選，非分析結果")

    # 逐郡 join 再累加列聯表。跨郡的 join 沒有意義（cvr_id 郡內唯一），
    # 故分郡處理不損失任何配對，且中間結果維持在單郡規模。
    parts = []
    for state_dir in sorted(RAW.iterdir()):
        if state_dir.name.removeprefix("state=") not in SWING:
            continue
        for county_dir in sorted(state_dir.iterdir()):
            lf = pl.scan_parquet(
                county_dir / "*.parquet", extra_columns="ignore"
            )
            pres = (
                lf.filter(pl.col("office") == "US PRESIDENT")
                .select("cvr_id", pres_party="party")
            )
            down = (
                lf.filter(pl.col("office") != "US PRESIDENT")
                .filter(pl.col("party").is_in(["DEM", "REP"]))
                .group_by("cvr_id")
                .agg(
                    pl.col("party").n_unique().alias("n_parties"),
                    pl.col("party").first().alias("first_party"),
                )
            )
            parts.append(
                pres.join(down, on="cvr_id", how="inner")
                .with_columns(
                    pl.when(pl.col("n_parties") > 1)
                    .then(pl.lit("Mixed"))
                    .when(pl.col("first_party") == "DEM")
                    .then(pl.lit("All-DEM"))
                    .otherwise(pl.lit("All-REP"))
                    .alias("down_ballot"),
                    pl.when(pl.col("pres_party").is_in(["DEM", "REP"]))
                    .then(pl.col("pres_party"))
                    .otherwise(pl.lit("Other/Under"))
                    .alias("pres"),
                )
                .group_by("pres", "down_ballot")
                .agg(pl.len().alias("n"))
                .collect(engine="streaming")
            )

    joined = (
        pl.concat(parts)
        .group_by("pres", "down_ballot")
        .agg(pl.col("n").sum())
    )

    table = joined.pivot(on="down_ballot", index="pres", values="n").fill_null(0)
    print("\n總統選擇 × 下游政黨組合（列聯表，人數）：")
    print(table)

    tot = joined["n"].sum()
    split = joined.filter(
        ((pl.col("pres") == "DEM") & (pl.col("down_ballot") == "All-REP"))
        | ((pl.col("pres") == "REP") & (pl.col("down_ballot") == "All-DEM"))
    )["n"].sum()
    print(f"\n可比對的選票數：{tot:,}")
    print(f"明確反向分裂投票：{split:,}（{100 * split / tot:.2f}%）")


def plot_state_sizes(per_state: pl.DataFrame) -> None:
    """各州列數長條圖。單變數分布屬允許範圍。"""
    section("9. 州規模分布圖")
    FIGS.mkdir(parents=True, exist_ok=True)
    d = per_state.sort("列數", descending=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(d["state"], d["列數"] / 1e6)
    ax.set_ylabel("rows (millions)")
    ax.set_title("2020 CVR rows by state (screening only)")
    plt.xticks(rotation=75, ha="right", fontsize=7)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = FIGS / "cvr2020-state-sizes.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"圖已存：{out}")
    print("僅供篩選，非分析結果")


def main() -> int:
    if not RAW.exists():
        print(f"缺少資料目錄：{RAW}", file=sys.stderr)
        return 1

    cols = check_schema()
    check_size()
    per_state = check_coverage()
    check_voters()
    check_offices()
    check_candidate_codes()
    check_nulls(cols)
    check_split_ticket()
    plot_state_sizes(per_state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
