# CLAUDE.md 中能機械判斷的規則：文件不手寫章節序號、腳本與輸出檔名不編號、腳本不寫型別註記、類別、
# if __name__ == "__main__" 與 argparse。需要判斷的規則（不寫估計量、改動即重寫）不在此，仍由 CLAUDE.md 規範。
# docs/internal/blog/ 是覆盤紀錄，不檢查；archive 目錄不檢查。

import ast
import re
import sys
from pathlib import Path

bad = []
section = re.compile(r"(見|如|參見|參考)?第[一二三四五六七八九十百\d]+[章節]")
for f in [*Path("docs").rglob("*.md"), *Path("report").glob("*.typ"), Path("README.md")]:
    if "archive" in f.parts or "blog" in f.parts:
        continue
    for i, line in enumerate(f.read_text().splitlines(), 1):
        if section.search(line):
            bad.append(f"{f}:{i}: 手寫的章節序號，改用標籤引用")

numbered = re.compile(r"^(\d+|step\d+|s\d+)[_-]")
for d in ["scripts", "outputs/tables", "outputs/figures"]:
    for f in Path(d).rglob("*"):
        if "archive" not in f.parts and numbered.match(f.name):
            bad.append(f"{f}: 檔名編號，改取自它回答的問題")

for f in Path("scripts").rglob("*.py"):
    if "archive" in f.parts:
        continue
    for node in ast.walk(ast.parse(f.read_text())):
        where = f"{f}:{getattr(node, 'lineno', 0)}"
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            a = node.args
            if node.returns or any(x.annotation for x in [*a.posonlyargs, *a.args, *a.kwonlyargs, a.vararg, a.kwarg] if x):
                bad.append(f"{where}: 型別註記")
        elif isinstance(node, ast.AnnAssign):
            bad.append(f"{where}: 型別註記")
        elif isinstance(node, ast.ClassDef):
            bad.append(f"{where}: 類別")
        elif isinstance(node, ast.Compare) and isinstance(node.left, ast.Name) and node.left.id == "__name__":
            bad.append(f"{where}: if __name__ == \"__main__\"")
        elif isinstance(node, ast.Attribute) and node.attr == "ArgumentParser":
            bad.append(f"{where}: argparse")

print("\n".join(bad))
sys.exit(1 if bad else 0)
