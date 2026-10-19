"""檢查簡報是否有投影片的內容溢出到下一頁。

touying 遇到放不下的內容會自動多開一頁，編譯不報錯；
因此比對 PDF 頁數與「標題頁加二級標題數」，不等即有溢出。
"""

from pathlib import Path
import json
import re
import subprocess
import tempfile

bad = []
for path in sorted(Path("report").glob("*.typ")):
    if "deck.with" not in path.read_text():
        continue
    heads = json.loads(subprocess.run(["typst", "query", "--root", ".", str(path), "heading.where(level: 2)"], capture_output=True, text=True, check=True).stdout)
    with tempfile.TemporaryDirectory() as tmp:
        pdf = Path(tmp) / "out.pdf"
        subprocess.run(["typst", "compile", "--root", ".", str(path), str(pdf)], check=True)
        info = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True, check=True).stdout
    pages = int(re.search(r"^Pages:\s+(\d+)", info, re.M).group(1))
    print(path, "pages", pages, "slides", len(heads) + 1)
    if pages != len(heads) + 1:
        bad.append(path)

if bad:
    raise SystemExit(f"內容溢出：{', '.join(map(str, bad))}")
