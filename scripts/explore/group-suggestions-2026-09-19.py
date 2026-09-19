"""組員提出的四個 Kaggle 候選資料集的篩選用檢視。

僅產出筆數、欄位型別、缺失率、唯一值數與描述統計，不做任何建模。
"""

import pathlib

import polars as pl

RAW = pathlib.Path(__file__).resolve().parents[2] / "data" / "raw"

DATASETS = {
    "EV 電池健康": RAW / "ev-battery-health-prediction-dataset-20k" / "ev battery_failure prediction Dataset.csv",
    "學生考試表現": RAW / "students-performance-in-exams" / "StudentsPerformance.csv",
    "臺北市實價登錄": RAW / "taiwan-taipei-city-real-estate-transaction-records" / "taipei_city_real_estate_transaction_v2.csv",
    "AI 對就業薪資影響": RAW / "ai-impact-on-jobs-and-salaries-2020-2026" / "ai_jobs_salaries_clean.csv",
}


def inspect(label: str, path: pathlib.Path) -> None:
    df = pl.read_csv(path, infer_schema_length=10000)
    print(f"\n{'=' * 70}\n{label}  筆數={df.height}  欄位數={df.width}\n{'=' * 70}")
    for col in df.columns:
        s = df[col]
        null_rate = s.null_count() / df.height
        n_unique = s.n_unique()
        line = f"  {col:<38} {str(s.dtype):<10} 缺失={null_rate:6.2%} 唯一={n_unique:>6}"
        if s.dtype.is_numeric():
            line += f" min={s.min()} med={s.median()} max={s.max()}"
        elif n_unique <= 12:
            line += f" 值={sorted(str(v) for v in s.unique() if v is not None)}"
        print(line)


def inspect_ai_jobs() -> None:
    """候選 4 的深入檢視。

    一般欄位掃描看不出這份資料的問題——它幾乎沒有 null。問題在於欄位清單
    裡缺了資料集標題所承諾的變數，以及手工對照欄的覆蓋率。
    """
    df = pl.read_csv(DATASETS["AI 對就業薪資影響"], infer_schema_length=10000)
    print(f"\n{'=' * 70}\nAI 對就業薪資影響：深入檢視\n{'=' * 70}")

    print("\n[欄位清單] 確認是否含 AI 曝險度")
    print(" ", df.columns)

    print("\n[isco_group_hint 覆蓋率]")
    n_null = df["isco_group_hint"].null_count()
    print(f"  缺失 {n_null} / {df.height} = {n_null / df.height:.1%}")
    print("  可接合的組別：")
    for row in (
        df.drop_nulls("isco_group_hint")
        .group_by("isco_group_hint")
        .agg(pl.len())
        .sort("len", descending=True)
        .to_dicts()
    ):
        print(f"    {row['isco_group_hint']:<60} {row['len']:>6}")

    print("\n[被歸為 Other / Unclassified 的最大職稱]")
    unclassified = df.filter(pl.col("role_family") == "Other / Unclassified")
    for row in unclassified["job_title"].value_counts(sort=True).head(10).to_dicts():
        print(f"    {row['job_title']:<30} {row['count']:>6}")

    print("\n[地理集中度]")
    top = df["company_location"].value_counts(sort=True).head(3).to_dicts()
    print(f"  前三國 {[r['company_location'] for r in top]} 佔 "
          f"{sum(r['count'] for r in top) / df.height:.1%}")
    same = (df["employee_residence"] == df["company_location"]).mean()
    print(f"  employee_residence == company_location：{same:.2%}")

    print("\n[年度分布與 remote_ratio 年均值]")
    for row in (
        df.group_by("work_year")
        .agg(pl.len(), pl.col("remote_ratio").mean())
        .sort("work_year")
        .to_dicts()
    ):
        print(f"    {row['work_year']}  n={row['len']:>6}  "
              f"remote_ratio 均值={row['remote_ratio']:.1f}")

    print("\n[薪資粒度：估算值的跡象]")
    for step in (1000, 5000):
        print(f"    {step} 的整數倍：{(df['salary_in_usd'] % step == 0).mean():.1%}")

    print("\n[可追蹤性]")
    print("    個人 ID：無    公司 ID：無    對照組：無（全為 AI／資料類職缺）")


if __name__ == "__main__":
    for label, path in DATASETS.items():
        inspect(label, path)
    inspect_ai_jobs()
