"""把 `main()` 結構的篩選腳本機械轉成 notebook 分格版。

轉換規則：
- 模組層級的 import、常數、helper 函式 → 第一個儲存格
- `main()` 裡每個 `section("...")` 開一個新儲存格，標題轉成 markdown 格
- `main()` 以外的函式（繪圖 helper 等）原樣保留在 setup 格

**只做機械轉換，不刪節任何檢視步驟。** 這是刻意的——合併或摘要會讓
分析過程流失，而這些腳本的價值正在於完整的中間步驟。

用法：
    uv run python scripts/explore/_to_notebook.py taipower-unit-netp.py
"""

import ast
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def split_main(source: str) -> tuple[str, list[tuple[str, str]]]:
    """回傳 (setup 程式碼, [(section 標題, 該段程式碼)])。"""
    tree = ast.parse(source)
    lines = source.splitlines()

    main_node = next(
        (n for n in tree.body
         if isinstance(n, ast.FunctionDef) and n.name == "main"),
        None,
    )
    if main_node is None:
        raise SystemExit("找不到 main()，此腳本可能已是分格版")

    setup_parts = []
    for node in tree.body:
        if node is main_node:
            continue
        if isinstance(node, ast.If) and getattr(node.test, "left", None) is not None:
            continue  # if __name__ == "__main__"
        seg = lines[node.lineno - 1: node.end_lineno]
        setup_parts.append("\n".join(seg))

    body_lines = lines[main_node.body[0].lineno - 1: main_node.end_lineno]
    dedented = [ln[4:] if ln.startswith("    ") else ln for ln in body_lines]

    cells: list[tuple[str, str]] = []
    title, buf = "", []
    for ln in dedented:
        # main() 的 return 在模組層級無意義，剝掉
        if re.fullmatch(r"\s*return(\s+\d+)?\s*", ln):
            continue
        m = re.match(r'section\("(.+?)"\)', ln.strip())
        if m:
            if buf or title:
                cells.append((title, "\n".join(buf).strip()))
            title, buf = m.group(1), []
        else:
            buf.append(ln)
    if buf or title:
        cells.append((title, "\n".join(buf).strip()))

    return "\n\n".join(setup_parts), cells


def render(src_name: str, setup: str, cells: list[tuple[str, str]], doc: str) -> str:
    out = [
        f'"""{doc.strip()}\n\n本檔由 `{src_name}` 機械轉換而來（見 _to_notebook.py），',
        "內容完全相同，只改為 `# %%` 分格以便轉 notebook：",
        "",
        f"    jupytext --to notebook scripts/explore/nb_{src_name.replace('-', '_')}",
        '"""',
        "",
        "# %%",
        setup,
    ]
    for title, code in cells:
        if not code.strip():
            continue
        if title:
            out += ["", "# %% [markdown]", f"# ## {title}"]
        out += ["", "# %%", code]
    return "\n".join(out) + "\n"


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)

    src = HERE / sys.argv[1]
    source = src.read_text(encoding="utf-8")
    doc = ast.get_docstring(ast.parse(source)) or ""

    setup, cells = split_main(source)
    dst = HERE / f"nb_{src.stem.replace('-', '_')}.py"
    dst.write_text(render(src.name, setup, cells, doc), encoding="utf-8")

    print(f"{src.name} → {dst.name}（{len(cells)} 個儲存格）")
    for title, _ in cells:
        if title:
            print(f"    {title}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
