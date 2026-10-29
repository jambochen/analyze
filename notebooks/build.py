# 把 report/expected_score.typ 用到的分析腳本轉成 notebook 並執行，給組員看完整流程。
# notebook 由腳本自動切出，不手改：手改的複本會與腳本漂移。腳本改了就重跑這支。
# 切法：頂層的空行之後若接頂層程式，就開新儲存格；儲存格開頭的頂層註解轉成 markdown。
# 執行目錄設為專案根目錄，腳本的相對路徑才對；會覆寫 outputs/ 的表與圖，種子固定，結果與直接跑腳本相同。
# 在專案根目錄執行：.venv/bin/python notebooks/build.py [腳本名 ...]

import sys

import nbformat
from nbclient import NotebookClient

# 依賴順序：descriptive_slides 讀 calibration 的輸出，expected_score_slides 讀其餘各支的輸出
SCRIPTS = {
    "data_overview": "「資料與樣本」：各速度類別的局數、arena 局數、2014 年玩家數。",
    "calibration": "「斜率 β 與累計局數」：依累計局數分層的校準斜率（`calibration_cox`）。",
    "expected_score_baselines": "「估計的 α 與 β」「三個模型的比較」「樣本外的校準」：老手對局的 score test、三個模型的概似比檢定與 AIC、2014 年下半年的樣本外評估。",
    "threshold_rating_form": "附錄「切點隨平均等級分的形式」：白方優勢與和局帶寬依平均等級分分組各自估計。",
    "descriptive_slides": "「上一局與本局的轉移」「連輸、連贏之後的離開」：arena 內的一階 Markov 轉移與連輸、連贏後的離開率。",
    "expected_score_slides": "投影片的統計圖：只讀前面各支的輸出表，不重估模型。",
}


def cells(src):
    lines = src.rstrip("\n").split("\n")
    chunks, cur = [], []
    for i, line in enumerate(lines):
        prev_blank = i > 0 and lines[i - 1].strip() == ""
        if line.strip() and not line[0].isspace() and prev_blank and cur:
            chunks.append(cur)
            cur = []
        cur.append(line)
    chunks.append(cur)
    out = []
    for c in chunks:
        k = 0
        while k < len(c) and c[k].startswith("#"):
            k += 1
        if k:
            out.append(nbformat.v4.new_markdown_cell("\n".join(l.removeprefix("#").removeprefix(" ") for l in c[:k])))
        code = "\n".join(c[k:]).strip("\n")
        if code:
            out.append(nbformat.v4.new_code_cell(code))
    return out


for name in sys.argv[1:] or SCRIPTS:
    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_markdown_cell(f"# `scripts/{name}.py`\n\n報告 `report/expected_score.typ` 的{SCRIPTS[name]}\n\n本 notebook 由 `notebooks/build.py` 從腳本自動產生，內容以腳本為準。")]
    nb.cells += cells(open(f"scripts/{name}.py").read())
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    NotebookClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": "."}}).execute()
    nbformat.write(nb, f"notebooks/{name}.ipynb")
    print(name, "done", flush=True)
