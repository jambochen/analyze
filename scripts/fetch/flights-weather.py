"""下載候選 I（arimancy/observed-weather-us-flights）的篩選用檔案。

只取 24 個月入門檔與兩份小檔（機場對照表、資料字典）。
完整版 6.1 GB 在篩選階段沒有必要——入門檔 schema 與完整版相同。
"""

import subprocess
import sys
from pathlib import Path

DATASET = "arimancy/observed-weather-us-flights"
FILES = [
    "flights_weather_recent_24mo.parquet",
    "airports_reference.parquet",
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
    for path in sorted(DEST.iterdir()):
        print(f"{path.name:40s} {path.stat().st_size / 1e6:10.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
