"""下載候選 K（NYC 311，Socrata 資料集 erm2-nwe9）的篩選用資料。

取兩層：
1. 聚合摘要（SoQL $group）——機關、案件類型、月份、borough 的總量與結案率，
   不必下載全量即可判斷結構。
2. 一整年（2025）的列級資料——生存分析的存活時間與右設限必須看列級，
   聚合查詢看不出來。2,250 萬列全量在篩選階段沒有必要。

列級分頁以 aiohttp 併發抓取。

**速率限制**：Socrata 對匿名請求以 IP 為單位節流，實測可低至 4 KB/s。帶
application token 者官方聲明不節流（除非判定為濫用）。token 無 secret、不算
認證，僅用於將請求歸屬到特定應用程式。

設定方式：於 https://evergreen.data.socrata.com/signup 申請後，設環境變數
`SOCRATA_APP_TOKEN`。未設定則走匿名路徑，仍可運作但慢得多。

註：SODA 3.0（2025 年釋出）將強制要求 token，且端點改為
`/api/v3/<id>/query.json`。本腳本目前用的 2.1 `/resource/<id>.csv` 仍可用。
"""

import asyncio
import io
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import aiohttp
import polars as pl

BASE = "https://data.cityofnewyork.us/resource/erm2-nwe9"
ROW_YEAR = 2025
# 日期切片的寬度。365 萬列 / 365 天約每天 1 萬列，7 天約 7 萬列
SLICE_DAYS = 7
# 單次查詢的上限，設得比預期切片大小寬鬆，觸及即代表切片太寬需調小
SLICE_LIMIT = 200_000
# 匿名請求受 IP 節流，併發開大只會互相排擠；帶 token 時官方不節流，可放寬
CONCURRENCY_ANON = 6
CONCURRENCY_TOKEN = 16
RETRIES = 5

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "data" / "raw" / "nyc311"

# 列級只取生存分析與多層次模型用得到的欄位，避免拖進 48 欄中大量結構性全空者
ROW_COLS = [
    "unique_key", "created_date", "closed_date", "due_date", "status",
    "agency", "complaint_type", "descriptor", "location_type",
    "borough", "community_board", "council_district", "incident_zip",
    "open_data_channel_type", "resolution_description",
    "latitude", "longitude",
]

AGGREGATES = {
    "by_agency": {
        "$select": "agency, count(*) as n, "
                   "sum(case(closed_date is not null, 1, true, 0)) as n_closed",
        "$group": "agency",
        "$order": "n DESC",
    },
    "by_complaint_type": {
        "$select": "complaint_type, count(*) as n, "
                   "sum(case(closed_date is not null, 1, true, 0)) as n_closed",
        "$group": "complaint_type",
        "$order": "n DESC",
        "$limit": "300",
    },
    "by_month": {
        "$select": "date_trunc_ym(created_date) as ym, count(*) as n, "
                   "sum(case(closed_date is not null, 1, true, 0)) as n_closed",
        "$group": "ym",
        "$order": "ym",
        "$limit": "200",
    },
    "by_borough_agency": {
        "$select": "borough, agency, count(*) as n",
        "$group": "borough, agency",
        "$order": "n DESC",
        "$limit": "500",
    },
    "by_channel": {
        "$select": "open_data_channel_type, count(*) as n",
        "$group": "open_data_channel_type",
        "$order": "n DESC",
    },
}

WHERE = (f"created_date >= '{ROW_YEAR}-01-01T00:00:00' "
         f"and created_date < '{ROW_YEAR + 1}-01-01T00:00:00'")


async def get(session: aiohttp.ClientSession, params: dict) -> bytes:
    for attempt in range(RETRIES):
        try:
            async with session.get(f"{BASE}.csv", params=params) as resp:
                if resp.status in (429, 500, 502, 503, 504):
                    raise aiohttp.ClientResponseError(
                        resp.request_info, resp.history, status=resp.status
                    )
                resp.raise_for_status()
                return await resp.read()
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            wait = 2 ** attempt
            print(f"  重試 {attempt + 1}/{RETRIES}（{exc}），等待 {wait}s")
            await asyncio.sleep(wait)
    raise RuntimeError(f"查詢失敗：{params}")


async def fetch_aggregates(session: aiohttp.ClientSession) -> None:
    pending = {
        name: params for name, params in AGGREGATES.items()
        if not (DEST / f"agg_{name}.csv").exists()
    }
    for name in AGGREGATES:
        if name not in pending:
            print(f"[skip] agg_{name}.csv")
    results = await asyncio.gather(
        *(get(session, params) for params in pending.values())
    )
    for name, raw in zip(pending, results):
        (DEST / f"agg_{name}.csv").write_bytes(raw)
        print(f"[agg ] {name}")


async def fetch_rows(session: aiohttp.ClientSession, concurrency: int) -> None:
    """以日期切片取代 $offset 分頁。

    實測：`$order=unique_key` + `$limit=50000` 單頁逾時（>300s），同樣查詢
    拿掉 `$order` 只需 2.9s。`unique_key` 無索引，Socrata 每頁都對 365 萬列
    重做全表排序。但拿掉 `$order` 後 `$offset` 分頁不保證穩定，會漏列或重複。

    改以 `created_date` 切片：每片是獨立的範圍查詢，不需排序，且切片彼此
    互斥又週延，天然保證不重不漏。
    """
    target = DEST / f"requests_{ROW_YEAR}.parquet"
    if target.exists():
        print(f"[skip] {target.name}")
        return

    raw = await get(session, {"$select": "count(*)", "$where": WHERE})
    total = int(raw.decode().splitlines()[-1].strip().strip('"'))

    start = date(ROW_YEAR, 1, 1)
    end = date(ROW_YEAR + 1, 1, 1)
    slices = [
        (s, min(s + timedelta(days=SLICE_DAYS), end))
        for s in (start + timedelta(days=i)
                  for i in range(0, (end - start).days, SLICE_DAYS))
    ]
    print(f"[rows] {ROW_YEAR} 年共 {total:,} 列，分 {len(slices)} 個日期切片")

    sem = asyncio.Semaphore(concurrency)
    done = 0

    async def fetch_slice(lo: date, hi: date) -> tuple[date, pl.DataFrame]:
        nonlocal done
        where = (f"created_date >= '{lo}T00:00:00' "
                 f"and created_date < '{hi}T00:00:00'")
        async with sem:
            raw = await get(session, {
                "$select": ", ".join(ROW_COLS),
                "$where": where,
                "$limit": str(SLICE_LIMIT),
            })
        # 全欄位讀為字串，型別轉換留給檢視腳本，避免切片間 schema 不一致
        df = pl.read_csv(io.BytesIO(raw), infer_schema_length=0)
        done += 1
        print(f"  {done:>3}/{len(slices)} 切片 {lo} 至 {hi}：{len(df):,} 列")
        if len(df) >= SLICE_LIMIT:
            raise RuntimeError(
                f"切片 {lo} 至 {hi} 觸及上限 {SLICE_LIMIT:,}，"
                f"資料可能被截斷。請調小 SLICE_DAYS")
        return lo, df

    results = await asyncio.gather(*(fetch_slice(lo, hi) for lo, hi in slices))
    frames = [df for _, df in sorted(results, key=lambda kv: kv[0])]
    combined = pl.concat(frames, how="vertical_relaxed")

    if len(combined) != total:
        print(f"[warn] 取得 {len(combined):,} 列，與 count(*) 的 {total:,} 不符")
    n_dup = len(combined) - combined["unique_key"].n_unique()
    if n_dup:
        print(f"[warn] 有 {n_dup:,} 筆重複的 unique_key")

    combined.write_parquet(target)
    print(f"[done] {target.name} {len(combined):,} 列、"
          f"{target.stat().st_size / 1e6:.1f} MB")


async def main_async() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    timeout = aiohttp.ClientTimeout(total=300)
    headers = {"User-Agent": "analyze/0.1"}
    token = os.environ.get("SOCRATA_APP_TOKEN", "").strip()
    if token:
        headers["X-App-Token"] = token
        concurrency = CONCURRENCY_TOKEN
        print(f"[auth] 使用 app token，不受 IP 節流（併發 {concurrency}）")
    else:
        concurrency = CONCURRENCY_ANON
        print(f"[auth] 未設定 SOCRATA_APP_TOKEN，走匿名路徑（受 IP 節流，併發 {concurrency}）")
    async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
        await fetch_aggregates(session)
        await fetch_rows(session, concurrency)


def main() -> int:
    asyncio.run(main_async())
    return 0


if __name__ == "__main__":
    sys.exit(main())
