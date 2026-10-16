"""以 Lichess API 逐局查 2014 年對局的配對來源與對局時間。

月檔沒有 `source`（大廳、指定邀戰、pool）也沒有對局結束時間。前者用於依 CLAUDE.md
的紀律排除大廳中的指定邀戰；後者（`lastMoveAt` 減 `createdAt`）用於以對局結束起算
到下一局的時間，扣除懸殊對局下得較快造成的機械效果。

arena 對局先查，存活分析可先開始；大廳對局接著查。實測帶 token 與匿名的速度相同
（每次 300 局約 10 秒，限制在伺服器端），故匿名查詢。API 規格要求同一時間只發一個
請求，收到 429 時等一分鐘再試。已取得的對局不重查，中斷後可續傳。原始回應原樣以
gzip 壓縮存放，未壓縮約 4 GB，SSD 空間不足以存未壓縮檔。
"""

import asyncio
import gzip
import json
from pathlib import Path

import aiohttp
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "data" / "raw" / "lichess-game-meta"
DEST.mkdir(parents=True, exist_ok=True)
OUT = DEST / "games.ndjson.gz"

g = (
    pl.scan_parquet(ROOT / "data" / "processed" / "lichess_games.parquet")
    .filter(pl.col("start").dt.year() == 2014)
    .select("game_id", "start", arena=pl.col("tournament_id").is_not_null())
    # arena 先、各自依時間排序，存活分析需要的 arena 部分最先完成
    .sort(pl.col("arena").not_(), "start")
    .collect()
)
g.select("game_id", "arena").write_csv(DEST / "requested.csv")

done = set()
if OUT.exists():
    with gzip.open(OUT, "rt") as f:
        done = {json.loads(l)["id"] for l in f}
ids = [i for i in g["game_id"].to_list() if i not in done]
print(len(done), len(ids), flush=True)

URL = "https://lichess.org/api/games/export/_ids?moves=false"


async def main():
    async with aiohttp.ClientSession(headers={"Accept": "application/x-ndjson"}, timeout=aiohttp.ClientTimeout(total=600)) as session:
        for i in range(0, len(ids), 300):
            while True:
                # 連續數日的抓取必然遇到伺服器中斷連線；這類錯誤重試即可，不該讓整個抓取停下
                try:
                    async with session.post(URL, data=",".join(ids[i : i + 300])) as r:
                        if r.status == 429:
                            await asyncio.sleep(60)
                            continue
                        r.raise_for_status()
                        text = await r.text()
                        break
                except (aiohttp.ClientPayloadError, aiohttp.ClientConnectionError, asyncio.TimeoutError) as e:
                    print("retry", i, type(e).__name__, flush=True)
                    await asyncio.sleep(30)
            # 每批各寫一個 gzip member 再關檔，中斷時最多損失一批
            with gzip.open(OUT, "at") as out:
                out.write(text if text.endswith("\n") else text + "\n")
            if i // 300 % 100 == 0:
                print(i + 300, len(ids), flush=True)


asyncio.run(main())
