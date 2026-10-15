"""下載 Lichess 標準棋 rated 對局月檔，並以官方 sha256 驗證。

2014 全年：arena 錦標賽整年隨機配對，提供匹配品質的自然實驗。每月約 70 萬到
135 萬局，只解析表頭即可分析。連續十二個月讓跨月的玩家序列不在月底被截斷。

2015 年 1 至 6 月：arena 於 2015 年 4 月改依等級分配對，取改制前後各約三個月，
比較隨機配對與等級分配對下的留存。
"""

import subprocess
from pathlib import Path

BASE = "https://database.lichess.org/standard"
MONTHS = [f"2014-{m:02d}" for m in range(1, 13)] + [f"2015-{m:02d}" for m in range(1, 7)]

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
