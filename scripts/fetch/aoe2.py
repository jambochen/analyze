"""下載 AoE2 DE ranked 對局的 parquet 快照（AOE2 Complete Guide，aoe2-guide.vercel.app/data）。

網站的 parquet 是隨版本累積、新版本上線即重置的快照，同一網址日後內容不同，所以依下載日期另存一個目錄，
並印出大小與 sha256 供 SOURCE.md 記錄。
"""

import datetime as dt
import hashlib
import subprocess
from pathlib import Path

BASE = "https://aoe2-guide.vercel.app/data"
DEST = Path(__file__).resolve().parents[2] / "data" / "raw" / "aoe2" / dt.date.today().isoformat()
DEST.mkdir(parents=True, exist_ok=True)

for f in ["matches.parquet", "players.parquet"]:
    subprocess.run(["curl", "-fsSL", "-D", str(DEST / f"{f}.headers"), "-o", str(DEST / f), f"{BASE}/{f}"], check=True)
    print(f, (DEST / f).stat().st_size, hashlib.sha256((DEST / f).read_bytes()).hexdigest())
