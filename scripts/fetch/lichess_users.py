"""以 Lichess API 查 2014 年 arena 錦標賽玩家的帳號資料，取帳號建立時間。

用途：界定新手期。月檔從 2013 年起，更早建立的帳號看不到對局史；帳號建立時間
不隨時間改變，可直接界定，也用來驗證以 2013 年起累計局數界定新手期的代理。

只查 2014 年參加過 arena 的帳號，不查全部帳號：API 規格明言此端點不可用於
匯出全站使用者，且限制每 10 分鐘 8,000 個、每天 120,000 個。每次 300 個，
請求間隔 23 秒，維持在每 10 分鐘 8,000 個以下；同一時間只發一個請求。
收到 429 時等 10 分鐘，讓速率視窗重置。原始回應原樣存成 ndjson。
"""

import asyncio
import json
from pathlib import Path

import aiohttp
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "data" / "raw" / "lichess-users"
DEST.mkdir(parents=True, exist_ok=True)

g = pl.scan_parquet(ROOT / "data" / "processed" / "lichess_games.parquet").filter(
    pl.col("month").str.starts_with("2014"), pl.col("tournament_system") == "arena"
)
# API 的使用者 ID 是小寫的帳號名稱
ids = (
    pl.concat([g.select(user="white"), g.select(user="black")])
    .unique()
    .sort("user")
    .collect()
    .with_columns(id=pl.col("user").str.to_lowercase())
)
ids.write_csv(DEST / "requested.csv")
ids = ids["id"].to_list()
print(len(ids))

URL = "https://lichess.org/api/users?profile=false"


async def main():
    async with aiohttp.ClientSession() as session:
        with open(DEST / "users.ndjson", "w") as out:
            for i in range(0, len(ids), 300):
                while True:
                    async with session.post(URL, data=",".join(ids[i : i + 300])) as r:
                        if r.status == 429:
                            await asyncio.sleep(600)
                            continue
                        r.raise_for_status()
                        for u in await r.json():
                            out.write(json.dumps(u) + "\n")
                        break
                out.flush()
                print(i + 300, len(ids), flush=True)
                await asyncio.sleep(23)


asyncio.run(main())
