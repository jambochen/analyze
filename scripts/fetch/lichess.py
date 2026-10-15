"""下載 Lichess 標準棋 rated 對局 2014 全年月檔，並以官方 sha256 驗證。

取 2014 年：每月約 70 萬到 135 萬局，全年壓縮檔約 2 GB，只解析表頭即可分析，
規模足以切出連續對局與流失。連續十二個月讓跨月的玩家序列不在月底被截斷。
2014 年的棋步尚無時鐘與引擎評估標註，研究問題只用表頭，不受影響。
"""

import subprocess
from pathlib import Path

BASE = "https://database.lichess.org/standard"
MONTHS = [f"2014-{m:02d}" for m in range(1, 13)]

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "data" / "raw" / "lichess"
DEST.mkdir(parents=True, exist_ok=True)

for name in ["sha256sums.txt", "counts.txt"]:
    subprocess.run(["curl", "-sSfL", "-o", DEST / name, f"{BASE}/{name}"], check=True)

files = [f"lichess_db_standard_rated_{m}.pgn.zst" for m in MONTHS]
for f in files:
    subprocess.run(["curl", "-sSfL", "-C", "-", "-o", DEST / f, f"{BASE}/{f}"], check=True)

sums = [l for l in (DEST / "sha256sums.txt").read_text().splitlines() if l.split()[-1] in files]
subprocess.run(["sha256sum", "-c", "-"], input="\n".join(sums) + "\n", text=True, cwd=DEST, check=True)
