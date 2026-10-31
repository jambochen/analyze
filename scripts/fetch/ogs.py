"""下載 OGS 圍棋對局資料庫 ogs-data.db（online-go/goratings，git LFS）。

repo 的檔案是 LFS 指標，實體由 media.githubusercontent.com 提供。大小與 sha256 取自
指標檔，下載後驗證；curl -C - 可在中斷後續傳。
"""

import hashlib
import re
import subprocess
from pathlib import Path

DEST = Path(__file__).resolve().parents[2] / "data" / "raw" / "ogs"
RAW = "https://raw.githubusercontent.com/online-go/goratings/master/data/ogs-data.db"
MEDIA = "https://media.githubusercontent.com/media/online-go/goratings/master/data/ogs-data.db"

pointer = subprocess.run(["curl", "-sfL", RAW], capture_output=True, text=True, check=True).stdout
(DEST / "ogs-data.db.lfs-pointer").write_text(pointer)
oid = re.search(r"oid sha256:(\w+)", pointer).group(1)
size = int(re.search(r"size (\d+)", pointer).group(1))

subprocess.run(["curl", "-fL", "-C", "-", "-o", str(DEST / "ogs-data.db"), MEDIA], check=True)

h = hashlib.sha256()
with open(DEST / "ogs-data.db", "rb") as f:
    while chunk := f.read(1 << 24):
        h.update(chunk)
assert (DEST / "ogs-data.db").stat().st_size == size
assert h.hexdigest() == oid
print("sha256 ok", oid)
