"""以 Lichess API 查每場錦標賽的賽制（arena 或瑞士制）。

月檔表頭的 `Event` 對 arena 與瑞士制都標為 tournament，只有 arena 是隨機配對。
配對來源只在 API 對局資料的 `source` 欄，批量檔案沒有。同一場錦標賽只有一種
賽制，每場取一局查詢即可。

API 規格要求同一時間只發一個請求，故依序發出；收到 429 時等一分鐘再試。
原始回應原樣存成 ndjson，與每場錦標賽所取的對局 ID 對照表一併存放。
"""

import asyncio
import csv
import subprocess
from pathlib import Path

import aiohttp

MONTHS = [f"2014-{m:02d}" for m in range(1, 13)] + [f"2015-{m:02d}" for m in range(1, 7)]

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "raw" / "lichess"
DEST = ROOT / "data" / "raw" / "lichess-tournament-source"
DEST.mkdir(parents=True, exist_ok=True)

# 跨月的錦標賽只保留第一次出現的對局
first = {}
for m in MONTHS:
    cmd = f"zstdcat {SRC}/lichess_db_standard_rated_{m}.pgn.zst | grep -E '^\\[(Event|Site) '"
    lines = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True).stdout.splitlines()
    for event, site in zip(lines[0::2], lines[1::2]):
        if "tournament" in event:
            tid = event.split("/tournament/")[1].rstrip('"]')
            first.setdefault(tid, site.split("lichess.org/")[1].rstrip('"]'))
    print(m, len(first))

with open(DEST / "tournament_game.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["tournament_id", "game_id"])
    w.writerows(first.items())

ids = list(first.values())
URL = "https://lichess.org/api/games/export/_ids?moves=false"


async def main():
    async with aiohttp.ClientSession(headers={"Accept": "application/x-ndjson"}) as session:
        with open(DEST / "games.ndjson", "w") as out:
            for i in range(0, len(ids), 300):
                while True:
                    async with session.post(URL, data=",".join(ids[i : i + 300])) as r:
                        if r.status == 429:
                            await asyncio.sleep(60)
                            continue
                        r.raise_for_status()
                        out.write(await r.text())
                        break
                print(i + 300, len(ids))


asyncio.run(main())
