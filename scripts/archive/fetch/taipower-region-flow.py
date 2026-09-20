"""下載候選 L-3（台電各區域間過去電力潮流量，d006009）。

**滾動視窗警告**：台電此類資料集雖上架多年，實際僅保留最近約 3 個月。
實測 2026-08-24 與 2026-09-19 兩次抓取內容相同（`Last-Modified` 均為
2026-08-24），代表視窗未必每日推進，但舊資料終將消失。

故本腳本**不覆寫既有檔案**，而是依下載日期另存，使歷次快照都能保留。
若要延長時間跨度，須定期執行本腳本累積快照。
"""

import sys
from datetime import date
from pathlib import Path
from urllib.request import Request, urlopen

DATASET = "d006009"
URL = f"https://service.taipower.com.tw/data/opendata/apply/file/{DATASET}/001.json"

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "data" / "raw" / "taipower-region-flow"


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    target = DEST / f"001_{date.today():%Y%m%d}.json"
    if target.exists():
        print(f"[skip] {target.name} 今日已下載")
        return 0

    req = Request(URL, headers={"User-Agent": "analyze/0.1"})
    with urlopen(req, timeout=180) as resp:
        # 來源的 Last-Modified 是判斷視窗是否推進的唯一線索，必須記錄
        last_modified = resp.headers.get("Last-Modified", "(無)")
        body = resp.read()

    target.write_bytes(body)
    print(f"[done] {target.name} {len(body) / 1e6:.1f} MB")
    print(f"       來源 Last-Modified：{last_modified}")

    snapshots = sorted(DEST.glob("001_*.json"))
    if len(snapshots) > 1:
        print(f"\n目前共 {len(snapshots)} 份快照：")
        for s in snapshots:
            print(f"  {s.name}  {s.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
