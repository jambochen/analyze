"""下載 arimancy/observed-weather-us-flights 的完整資料。

七個年度檔（2019-2025）構成 4,280 萬筆航班的完整面板，機場逐時氣象檔
提供年度檔以外的氣象脈絡。低報稽核要按航空公司與機場分層，小型業者與
小機場在 24 個月入門檔裡樣本數不足，故取完整版。

入門檔 flights_weather_recent_24mo.parquet 與年度檔內容重疊，保留它是
為了跑得快的檢視，不納入正式分析。
"""

import subprocess
import sys
from pathlib import Path

DATASET = "arimancy/observed-weather-us-flights"
FILES = [
    *[f"flights_weather_{year}.parquet" for year in range(2019, 2026)],
    "airport_hourly_weather.parquet",
    "airports_reference.parquet",
    "airports_reference.csv",
    "data_dictionary.md",
    "LICENSE.txt",
]

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "data" / "raw" / "flights-weather"


def fetch(name: str) -> None:
    target = DEST / name
    if target.exists():
        print(f"[skip] {name} 已存在")
        return
    print(f"[get ] {name}")
    subprocess.run(
        [
            "uvx", "--from", "kaggle", "kaggle", "datasets", "download",
            "-d", DATASET, "-f", name, "-p", str(DEST), "--unzip",
        ],
        check=True,
    )


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        fetch(name)
    total = 0.0
    for path in sorted(DEST.iterdir()):
        size = path.stat().st_size / 1e6
        total += size
        print(f"{path.name:40s} {size:10.1f} MB")
    print(f"{'總計':40s} {total:10.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
