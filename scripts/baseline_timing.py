"""核對第二次報告的口稿順序，依每秒兩字計算講述與停頓時間。"""

from pathlib import Path
import csv
import math
import re

source = Path("report/baseline.typ").read_text()
script_path = Path("report/baseline_script.md")
script = script_path.read_text()
slides = ["封面", *re.findall(r"^== (.+)$", source, re.M)]
parts = re.split(r"^## (.+)\n", script, flags=re.M)
titles = parts[1::2]
assert titles == slides, "投影片與口稿的標題或順序不一致"

rows = []
elapsed = 0
for i, title in enumerate(titles):
    body = parts[2 * i + 2]
    pause = int(re.search(r"^停頓與指圖：(\d+) 秒。$", body, re.M).group(1))
    spoken = re.sub(r"^(?:講述時間：|停頓與指圖：).*$", "", body, flags=re.M)
    characters = sum(c.isalnum() for c in spoken)
    speech = math.ceil(characters / 2)
    end = elapsed + speech + pause
    start_label = f"{elapsed // 60:02d}:{elapsed % 60:02d}"
    end_label = f"{end // 60:02d}:{end % 60:02d}"
    timing = f"講述時間：{start_label}–{end_label}（口述 {speech} 秒）。"
    body = re.sub(r"^講述時間：.*$", timing, body, flags=re.M)
    parts[2 * i + 2] = body
    rows.append([title, characters, speech, pause, elapsed, end])
    elapsed = end

with Path("outputs/tables/baseline_timing.csv").open("w") as f:
    writer = csv.writer(f)
    writer.writerow(["slide", "characters", "speech_seconds", "pause_seconds", "start_seconds", "end_seconds"])
    writer.writerows(rows)

rebuilt = parts[0] + "".join(f"## {title}\n{body}" for title, body in zip(parts[1::2], parts[2::2]))
script_path.write_text(rebuilt.rstrip() + "\n")
print(f"{len(rows)} slides; {sum(row[1] for row in rows)} characters; {elapsed // 60}:{elapsed % 60:02d}; buffer {1200 - elapsed} seconds")
assert elapsed <= 1200, "按每秒兩字與逐頁停頓估算，超過20分鐘"
