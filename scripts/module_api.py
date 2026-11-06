# 列出被其他腳本 import 的共用 module：每個 module 頂端註解的第一行，與其公開函式的簽名、docstring 首行。
# 找介面時讀這份輸出即可，不必讀完整支分析腳本；資訊直接取自程式碼，不會過時。

import ast
from pathlib import Path

files = sorted(Path("scripts").rglob("*.py"))
names = {f.stem for f in files if f.parent == Path("scripts")}
used = set()
for f in files:
    for node in ast.walk(ast.parse(f.read_text())):
        if isinstance(node, ast.Import):
            used |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            used.add(node.module.split(".")[0])

for name in sorted(names & used):
    src = Path(f"scripts/{name}.py").read_text()
    head = next((line.lstrip("# ").strip() for line in src.splitlines() if line.startswith("#") or line.startswith('"""')), "")
    print(f"{name}: {head.strip(chr(34))}")
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
            doc = (ast.get_docstring(node) or "").split("\n")[0]
            print(f"  {node.name}({ast.unparse(node.args)})  {doc}")
        elif isinstance(node, ast.Assign):
            ids = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if len(ids) == len(node.targets) and all(i.isupper() for i in ids):
                print(f"  {', '.join(ids)} = {ast.unparse(node.value)[:80]}")
