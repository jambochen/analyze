"""檢查 Typst 報告中非 CeTZ 示意圖的圖說是否含漢字。"""

from pathlib import Path
import re


han = re.compile(r"[\u3400-\u9fff]")
caption = re.compile(r"caption:\s*\[")
violations = []

for path in sorted(Path("report").glob("*.typ")):
    source = path.read_text()
    for match in caption.finditer(source):
        figure_start = source.rfind("#figure(", 0, match.start())
        if figure_start >= 0 and "cetz.canvas" in source[figure_start:match.start()]:
            continue

        start = match.end()
        depth = 1
        end = start
        quoted = False
        while depth and end < len(source):
            char = source[end]
            if char == '"' and (end == 0 or source[end - 1] != "\\"):
                quoted = not quoted
            elif not quoted and char == "[":
                depth += 1
            elif not quoted and char == "]":
                depth -= 1
            end += 1

        body = source[start:end - 1]
        visible = []
        i = 0
        while i < len(body):
            if body[i] != "#":
                visible.append(body[i])
                i += 1
                continue

            # 略過 Typst 插值及其參數；參數中可能有中文 CSV 欄名。
            i += 1
            while i < len(body) and (body[i].isalnum() or body[i] in "_.-"):
                i += 1
            while i < len(body) and body[i] == "(":
                nesting = 1
                i += 1
                quoted = False
                while nesting and i < len(body):
                    char = body[i]
                    if char == '"' and body[i - 1] != "\\":
                        quoted = not quoted
                    elif not quoted and char == "(":
                        nesting += 1
                    elif not quoted and char == ")":
                        nesting -= 1
                    i += 1
                while i < len(body) and (body[i].isalnum() or body[i] in "_.-"):
                    i += 1

        if han.search("".join(visible)):
            line = source.count("\n", 0, match.start()) + 1
            violations.append(f"{path}:{line}: 圖說含中文可見文字")

if violations:
    raise SystemExit("\n".join(violations))
