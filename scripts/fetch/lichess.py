"""下載 Lichess 標準棋 rated 對局月檔，並以官方 sha256 驗證。

2013 全年：只作為 2014 年玩家的對局史，用於界定新手期（帳號累計局數）與
重算等級分時的預熱，不進主分析。

2014 全年：arena 錦標賽整年隨機配對，提供匹配品質的自然實驗。每月約 70 萬到
135 萬局，只解析表頭即可分析。連續十二個月讓跨月的玩家序列不在月底被截斷。

2015 年 1 至 6 月：arena 於 2015 年 4 月改為參考等級分配對，取改制前後各約
三個月；也讓 2014 年底的對局有足夠的後續期間判定是否流失。

已存在的檔案不重下載，最後統一以 sha256 驗證。
"""

import asyncio
import subprocess
from pathlib import Path

import aiohttp

BASE = "https://database.lichess.org/standard"
MONTHS = (
    [f"2013-{m:02d}" for m in range(1, 13)]
    + [f"2014-{m:02d}" for m in range(1, 13)]
    + [f"2015-{m:02d}" for m in range(1, 7)]
)

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "data" / "raw" / "lichess"
DEST.mkdir(parents=True, exist_ok=True)

files = [f"lichess_db_standard_rated_{m}.pgn.zst" for m in MONTHS]


async def fetch(session, sem, name):
    async with sem, session.get(f"{BASE}/{name}") as r:
        r.raise_for_status()
        with open(DEST / name, "wb") as f:
            async for chunk in r.content.iter_chunked(1 << 20):
                f.write(chunk)
    print(name)


async def main():
    # 靜態檔案伺服器，少量並行即可；不是 API，不受一次一個請求的限制
    sem = asyncio.Semaphore(4)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as session:
        await asyncio.gather(*(fetch(session, sem, n) for n in ["sha256sums.txt", "counts.txt"]))
        todo = [f for f in files if not (DEST / f).exists()]
        await asyncio.gather(*(fetch(session, sem, f) for f in todo))


asyncio.run(main())

sums = [l for l in (DEST / "sha256sums.txt").read_text().splitlines() if l.split()[-1] in files]
subprocess.run(["sha256sum", "-c", "-"], input="\n".join(sums) + "\n", text=True, cwd=DEST, check=True)
