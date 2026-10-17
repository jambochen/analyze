# 續下的選擇：被分到強對手是否改變續下，crossover 的結論能否推到所有玩家

腳本：`scripts/continuation_selection.py`。共用 module：`scripts/arena_batches.py`（`player_sample_rd`：樣本、$h$ 與主基準殘差）、`scripts/likelihood.py`（`lrt_player_dyad`：修正的概似比檢定）。表在 `outputs/tables/continuation_selection_*.csv`，圖在 `outputs/figures/continuation_selection_bounds.png`。規格見 `docs/methods.md`「續下的選擇模型」「續下的選擇：界限與對照」。本文件只寫定性結論，數值一律見對應的輸出檔。

## 問題

crossover 模型（`docs/crossover.md`）的反應變數是同一場錦標賽中之後第 $k$ 局的表現，只有續下的玩家才有這一局。被分到強對手若同時改變誰會續下，續下者在兩種對手強弱下就不是同一群人，續下者之間的差可能只是組成的差。這份分析回答兩件事：

- 被隨機分到比期望強的對手，是否改變同場再開一局的機率，方向是否單調
- 把結論限定在「兩種對手強弱下都會續下的玩家」（survivor average causal effect）時，效果的界限是否仍排除 0

## 資料與做法

- 2014 年 arena 的玩家局，風險集為本局之後錦標賽仍有 $k$ 局名目長度的時間、且不被資料終點或全站停擺截斷者；跨年的錦標賽排除。各速度類別、層與局數的列數、玩家數與續下率見 `continuation_selection_sample.csv`
- 續下模型：logistic 迴歸，以局加權的修正概似比檢定（`continuation_selection_tests.csv`、`continuation_selection_coef.csv`）
- 兩組：$\bar e$ 的五等分格內，$h$ 最大與最小的五分之一。$\bar e$ 接近 0.5 的玩家 $|h|$ 較大、$\bar e$ 極端的玩家集中在 $h$ 的中段，全體五等分的組成因此不同；先依 $\bar e$ 分格，格內的分組只來自配對的隨機
- 反應變數：之後第 $k$ 局的主基準殘差，即總效果的尺度。效果以「最強五分之一減最弱五分之一」表示，與 crossover 以 $h$ 的斜率表示的總效果不是同一個量，方向可以對照、大小不可直接比較
- 界限與對照：Lee 界限為主；不假設單調性的界限、inverse probability weighting、選擇上的有參數敏感度分析為對照。以玩家加權為主，以局加權為次；區間為玩家 block bootstrap 的 Imbens–Manski 區間

## 結論

### 被分到強對手的人較少續下，且單調

- **總效果**：$h$ 越大，同場再開一局的機率越低。Bullet 與 Blitz 的兩層在 Holm 校正後都拒絕「$h$ 不影響續下」（`continuation_selection_tests.csv` 的 `model = total`）。升級規則的診斷不通過，界限必須使用
- **單調性**：$\bar e$ 格內的五個 $h$ 等分，續下率隨對手變強依序下降；所有速度類別、層、局數與加權下，相鄰差都沒有與整體趨勢相反而顯著者（`continuation_selection_rates.csv`、`continuation_selection_monotone.csv`）。資料支持單調性的方向，Lee 界限可用
- **給定本局結果**：輸棋與和棋時，對手越強反而越常續下；贏棋時對手越強越少續下（`continuation_selection_coef.csv` 的 `model = given_result`）。總效果的方向主要經由「對手強就較常輸、輸了就較少續下」。給定結果時方向反轉，與 `docs/eomm-assumption.md`「爆冷輸後離開最多」一致：輸給比期望強的對手，不如輸給弱的對手那樣令人離開

### 界限都含 0：crossover 的總效果只在續下者上成立

- **只用續下者**：Blitz 的第一局，以玩家加權時最強五分之一的下一局表現低於最弱五分之一，區間不含 0，方向與 crossover 的總效果一致。Bullet 在兩局以上的批沒有差異；一局一批以玩家加權時有負的差異，與 crossover 中 Bullet 一局一批的零星結果同樣不作解讀
- **Lee 界限**：所有速度類別、層、局數與加權下，界限都含 0（`continuation_selection_bounds.csv`、`continuation_selection_bounds.png`）。Blitz 第一局的界限下端偏負、上端略正：續下率在兩組間的差雖只有幾個百分點，殘差的分布很寬，修剪掉這個比例就足以讓效果的正負不定
- **依事先寫定的判準**（區間不含 0 才說效果在續下的選擇之下仍成立），Blitz 剛被分到強對手之後的 carryover，對「兩種對手強弱下都會續下的玩家」未能確立。它可能是 tilt，也可能只是較弱的玩家在被分到強對手後較不會離開、留下來的人組成不同
- **不假設單調性的界限**：比 Lee 界限寬得多，第二局之後以玩家加權時寬到沒有資訊。單調性收窄了大部分的寬度

### 對照

- **inverse probability weighting**：結論與只用續下者相同，Blitz 第一局的差異不含 0。這是預期中的：它假設給定控制變數後續下與表現無關，與只用續下者的假設只差在控制變數。續下機率的倒數沒有極端值，未截斷（`ess_min`、`w_max`）
- **有參數的敏感度分析**：在續下率較高的一組中，令續下者屬於「兩組都會續下」的機率依下一局殘差以 logistic 變化。Blitz 第一局以玩家加權時，只要「因對手弱才多留下來的人」下一局表現比一定會留下的人稍好，效果的點估計就變號；變號所需的偏離不大（`continuation_selection_sensitivity.csv`、`continuation_selection_flip.csv`。`flip` 的 `gamma` 為由小到大第一個與前一格點正負不同的格點）

### 多局 carryover

第二、三局的 Lee 界限同樣含 0，且隨局數變寬。第一局的界限已含 0，所以沒有任何局數的 carryover 能推到所有玩家；多局的結果只能在續下者上報告。

### 一局一批與 Classical

一局一批的層以「下完的時間與等級分無關」為前提，結論與兩局以上的批相同：續下隨對手變強單調下降，界限都含 0。Classical 的樣本小，只作描述。

## 對後續分析的意涵

- **依狀態調整的勝率**：預測的對象是「決定再下一局的人」的下一局，正是續下者。預測不需要推到不續下的人，選擇不影響預測的用途；Blitz 仍照 `docs/crossover.md` 加入上一局一項
- **停止規則**：回測只刪局，估計的是續下者照規則停下時的變化，不需要界限（`docs/methods.md`「停止規則」）。「剛輸使下一局變差」的因果解讀未能確立，建議的措辭須是「剛被分到強對手、仍選擇續下的人，下一局低於系統的期望」，不說「輸棋使人變差」
- **類推到電競**：排隊配對中的玩家同樣是選擇續下的人，預測上的結論可以類推；因果上的 tilt 在 Lichess 上就未能確立
- **連敗的 bootstrap**：停止的時點須以續下模型一併模擬，且停止機率應隨上一局對手強弱與結果變化，不只隨結果

## 限制

- 界限針對總效果的尺度（主基準殘差）；直接效果的殘差未做界限，直接效果在 crossover 中本身就未能確立
- 兩組只用 $h$ 的兩端五分之一，丟掉中間三組的資訊；它換來的是不需要對 $h$ 的劑量反應設模型
- 不假設單調性的界限，各格以「列數 × 較低的續下率」加權平均，不是在所有可能的權重上取極值；它在本題已經寬到沒有資訊，這個近似不影響結論
- bootstrap 以玩家為群，未處理同一局兩位玩家的相依；inverse probability weighting 的區間未計入續下模型的估計誤差
- 從對局結束起算的 hazard 需要 API 的對局結束時間，尚未做
