#!/usr/bin/env bash
set -eu

# 固定 arXiv 版本，讓方法核對對應到同一份原文。
mkdir -p ref/competitive-games
curl -L --fail --silent --show-error --max-time 90 https://arxiv.org/src/2503.21713v2 -o ref/competitive-games/gee-source.tar
curl -L --fail --silent --show-error --max-time 90 https://arxiv.org/src/2207.07780v1 -o ref/competitive-games/chowdhary-source.tar
curl -L --fail --silent --show-error --max-time 90 https://arxiv.org/src/1702.06820v1 -o ref/competitive-games/eomm-source.tar
curl -L --fail --silent --show-error --max-time 90 https://arxiv.org/pdf/2503.21713v2 -o ref/competitive-games/gee.pdf
curl -L --fail --silent --show-error --max-time 90 https://arxiv.org/pdf/2207.07780v1 -o ref/competitive-games/chowdhary.pdf
curl -L --fail --silent --show-error --max-time 90 https://arxiv.org/pdf/2502.10985v1 -o ref/competitive-games/tang.pdf
# 這兩篇作者頁未提供 TeX；EOMM 源碼包則只有包裝 PDF 的 TeX。
curl -L --fail --silent --show-error --max-time 90 https://www.cemyuksel.com/research/matchmaking/i3d2024-matchmaking.pdf -o ref/competitive-games/yuksel.pdf
curl -L --fail --silent --show-error --max-time 90 https://www.cemyuksel.com/research/matchmaking/i3d2024-matchmaking-supplemental.pdf -o ref/competitive-games/yuksel-supplemental.pdf
curl -L --fail --silent --show-error --max-time 90 https://www.cs.cornell.edu/~tj/publications/chen_joachims_16b.pdf -o ref/competitive-games/chen-joachims.pdf

python3 - <<'PY'
from pathlib import Path
import tarfile

root = Path('ref/competitive-games')
for name in ['gee', 'chowdhary', 'eomm']:
    with tarfile.open(root / f'{name}-source.tar') as archive:
        archive.extractall(root / f'{name}-source', filter='data')
PY

for paper in gee chowdhary tang yuksel yuksel-supplemental chen-joachims; do
    pdftotext -layout "ref/competitive-games/$paper.pdf" "ref/competitive-games/$paper.txt"
done
pdftotext -layout ref/competitive-games/eomm-source/camera_ready_final_final_v7.pdf ref/competitive-games/eomm.txt

python3 - <<'PY'
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

root = Path('ref/competitive-games')
sources = [
    ('tang.pdf', 'https://arxiv.org/pdf/2502.10985v1', 'arXiv nonexclusive-distrib/1.0'),
    ('gee-source.tar', 'https://arxiv.org/src/2503.21713v2', 'CC BY 4.0'),
    ('gee.pdf', 'https://arxiv.org/pdf/2503.21713v2', 'CC BY 4.0'),
    ('chowdhary-source.tar', 'https://arxiv.org/src/2207.07780v1', 'arXiv nonexclusive-distrib/1.0'),
    ('chowdhary.pdf', 'https://arxiv.org/pdf/2207.07780v1', 'arXiv nonexclusive-distrib/1.0'),
    ('eomm-source.tar', 'https://arxiv.org/src/1702.06820v1', 'arXiv nonexclusive-distrib/1.0; PDF wrapper only'),
    ('yuksel.pdf', 'https://www.cemyuksel.com/research/matchmaking/i3d2024-matchmaking.pdf', 'Copyright author; see PDF notice'),
    ('yuksel-supplemental.pdf', 'https://www.cemyuksel.com/research/matchmaking/i3d2024-matchmaking-supplemental.pdf', 'See author supplement and main PDF notice'),
    ('chen-joachims.pdf', 'https://www.cs.cornell.edu/~tj/publications/chen_joachims_16b.pdf', 'Copyright authors; publication rights ACM; see PDF notice'),
]
records = []
for name, url, rights in sources:
    path = root / name
    records.append({
        'file': name,
        'url': url,
        'downloaded_at_utc': datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
        'bytes': path.stat().st_size,
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'rights': rights,
    })
(root / 'manifest.json').write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n')
PY
