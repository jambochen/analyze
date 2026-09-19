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


if __name__ == "__main__":
    for label, path in DATASETS.items():
        inspect(label, path)
