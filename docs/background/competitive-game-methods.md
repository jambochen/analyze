# 核心對照論文的方法與可採用部分

核心對照為 Gee、Chowdhary、Yuksel、Chen–Joachims 與 EOMM；研究目標見[文獻定位](competitive-game-research-goals.md)。本筆記已閱讀下載原文的方法與驗證段落，區分原方法、本專案已有工作及可考慮的延伸。列為候選不表示已執行、已有效或已列入主模型；既有前瞻登記不更動。

## 本地原文與版本

下載腳本為 [`scripts/fetch/competitive_papers.sh`](../../scripts/fetch/competitive_papers.sh)，從專案根目錄執行。原檔、解開的源碼與文字抽取放在 `ref/competitive-games/`，沿用 `ref/` 不入版控的慣例。來源網址、內容雜湊與下載紀錄見同目錄 `manifest.json`。解壓使用 Python 的 data filter，沒有編譯或執行論文附帶程式。

| 論文 | 實際取得 | 閱讀入口 |
|---|---|---|
| Gee 等，Investigating Experiential Effects in Online Chess using a Hierarchical Bayesian Analysis | arXiv 2503.21713v2 正文 TeX、圖、書目及 PDF | [正文 TeX](../../ref/competitive-games/gee-source/journal-article.tex)、[PDF](../../ref/competitive-games/gee.pdf) |
| Chowdhary 等，Quantifying human performance in chess | arXiv 2207.07780v1 正文與補充 TeX、圖、書目及 PDF；這是預印本，不當成 Scientific Reports 正式版源碼 | [正文 TeX](../../ref/competitive-games/chowdhary-source/main.tex)、[補充 TeX](../../ref/competitive-games/chowdhary-source/supplementary.tex)、[PDF](../../ref/competitive-games/chowdhary.pdf) |
| Yuksel，Skill-Based Matchmaking for Competitive Two-Player Games | 作者提供的正文及補充 PDF；作者頁未找到公開 TeX | [正文](../../ref/competitive-games/yuksel.pdf)、[補充](../../ref/competitive-games/yuksel-supplemental.pdf) |
| Chen、Joachims，Predicting Matchups and Preferences in Context | 作者提供的 PDF；作者頁未找到公開 TeX | [正文](../../ref/competitive-games/chen-joachims.pdf) |
| Chen 等，EOMM | arXiv 1702.06820v1 source 包；`main.tex` 只有 `pdfpages` 載入已排好的 PDF，沒有可編輯正文 | [PDF](../../ref/competitive-games/eomm-source/camera_ready_final_final_v7.pdf)、[包裝 TeX](../../ref/competitive-games/eomm-source/main.tex) |

arXiv 的可下載源碼不自動等於可任意再授權。Gee 頁面標 CC BY 4.0；Chowdhary 與 EOMM 頁面標 arXiv 非專屬散布授權。作者 PDF 依各自版權聲明保存供研究閱讀；本筆記不重新發布全文。

## Gee：玩家差異與序列模型檢查

原文的 Modeling Winner-Loser Effects 使用階層貝氏 logistic。每位玩家有自己的截距與近況斜率，兩者採二變量常態分布作部分合併；固定項為分差與執子顏色。近況是同一 session 過去勝率相對累積勝率的偏離，另比較不同歷史長度。尺度採半常態先驗，相關矩陣採 LKJ 先驗，以 Stan 估計並檢查收斂。這個全體效應與個人效應模型不會自動解決續下選擇，也沒有使用 RD。

Model Validation 的主要做法是後驗預測檢查：以前段對局估計參數，在後段逐局模擬賽果，將模擬出的上一局寫回下一局的近況，再計算 Glicko-2 評分軌跡與觀察軌跡比較。附錄另做玩家內排列與已知效應、不同局數的模擬，檢查偵測能力。[原文 v2](https://arxiv.org/html/2503.21713v2)。

**可借用：以生成資料檢查現有模型漏掉的序列結構。** 不一定要先改成貝氏模型；可以用固定估計參數的參數式模擬，清楚標示未納入訓練不確定性。候選檢查量為連敗長度、累積表現殘差及月份偏差，依速度與 RD 分層，避免只看平均預測損失。

這裡有兩種不同設計，不能混用：

- 固定已觀察的局前特徵，抽樣勝和負，是條件於該組特徵的模型診斷。它不會重現近況與評分的回饋，不能宣稱完整生成了玩家歷史。
- 逐局遞迴生成則須同步更新雙方近況與評分，一局只生成一個一致結果。若仍固定實際對手、開局時間與續下序列，結論也只條件於該賽程；不是替代配對政策或休息介入的模擬。

採用前要先決定要檢查哪一種失配。Gee 的累積勝率文字使用「到目前這局」，若實作局前預測須明定只到上一局，不能把本局結果納入中心化。TeX 中留有註解與編修痕跡，方法以可見正文核對。

玩家 random intercept/slope 的方法復現與預測比較已完成，見[比較規格](../hierarchical-replication.md)與[結果](../results/hierarchical-prediction.md)。原式以模擬資料核對，本資料則保留和局、納入雙方玩家效果，與現行模型在相同測試對局比較，並檢查訓練期未見玩家的表現。既有[玩家異質性檢查](../results/player-heterogeneity.md)問的是對手強弱的 random slope，與 Gee 的近況係數不同；既有 random intercept GLMM 的比較流程也不同，兩者都不能代替這次驗證。報告統一介紹三種用時的論文模型，與共用係數的上一局模型比較；中用時改善，短用時 score MSE 增加，長用時不明確。相對完整候選 BIC 模型的另項比較保留於結果文件，不混用兩種對照的結論。

## Chowdhary：連勝虛無模型與開局多樣性

Materials and Methods 的連勝比較，對每位玩家重排整段生涯賽果，保留總勝、和、負數，再比較各種連勝連敗長度。開局多樣性用 Shannon entropy，另比較開局頻率與勝率，以及生涯不同區段的多樣性。玩家能力分組依生涯平均評分，屬回溯描述。[預印本](https://arxiv.org/abs/2207.07780v1)、[正式論文](https://doi.org/10.1038/s41598-023-27735-9)。

**可借用：清楚展示不同虛無模型回答不同問題。** 全序列打亂回答「與順序完全可交換相比，序列是否特別」；本專案的[連勝分析](../results/streaks.md)回答「逐局條件調整後，連敗是否仍有額外資訊」。可將前者作說明性對照，不能將它當成已控制實力變動的主檢定。

若追加連敗長度模擬，應與 Gee 的序列檢查合併規劃，避免重複分析。固定模型但實力與對手隨時間變化時，不能任意重排原始勝負。開局熵只有在研究目標擴展到學習或策略時才有用途；必須另解析棋步／開局欄位、處理每人局數差異，且當局開局不能作為局前模型輸入。生涯平均能力也不能拿來界定局前新手。

## Yuksel：自適應評分與目標勝率配對

Rating Estimation 將加權的得分殘差除以機率函數導數和，形成 Newton 式評分更新；近期賽果權重較高，遺忘速度由玩家評分變異調整。增量版本只維護少數累積量，另提供限制單步變動及雙方零和更新。Matchmaking 由近期勝績和目標勝率反推理想對手評分，再按理想評分排序配對，是近似法而非一般受限最優解。[作者正文與補充](https://www.cemyuksel.com/research/matchmaking/)。

**可借用：作為「評分追蹤不足」的替代預測基準。** 這比直接用它改配對更容易對應目前的問題。若納入候選，須按真實時間逐局更新，用同一測試集合比較現有評分基準、現有近況修正與新評分的增量；真實潛在實力不可觀察，所以評估局前勝和負機率，不把平台評分當實力真值。

成本是要重建完整評分狀態、規定暖身與左截斷處理，並獨立處理和局及雙方更新；不能把更新公式直接當三類結果機率。其評分變異也不能直接稱為本專案 Glicko-2 的 RD。

**目前列為較大延伸，非優先替換。** 原文實證全部是合成模擬；主文的 Glicko 對照不讓局間時間增加 RD，補充才比較其他設定，所以不能由該比較斷言實際 Lichess Glicko-2 較差。其近期目標勝率配對會主動調整對手難度，目標與「每局勢均力敵」不同。若作政策候選，必須保留本專案的合法對手、顏色、完整分母與個人代價檢查；靜態批次無法證明長期連敗減少。

## Chen–Joachims：情境交互作用與成分比較

模型為每位玩家形成 blade 與 chest 向量，利用雙方交叉內積之差建立反對稱優勢，再轉成勝率。向量可由玩家特徵線性或 tanh 映射而來；CONCAT 合併玩家與情境特徵，SPLIT 先分別映射再逐元素相乘。以對數概似與矩陣正則化訓練，驗證資料選維度及正則強度，測試資料報預測表現。StarCraft II 用隨機切分；網球才按時段留出未來測試資料。[作者全文](https://www.cs.cornell.edu/~tj/publications/chen_joachims_16b.pdf)。

**可借用：分開檢查資訊來源及情境交互作用。** 對照組可依現有基準、雙方近況、情境及交互作用分開，但每項必須對應尚未回答的問題。本專案已有 RD 與分數變動交互作用、首局與間隔比較，見[時間外結果](../results/temporal-validation.md)及[情境結果](../results/entry-context.md)，這些不必換個名稱再做。

若新診斷顯示某個局前情境有穩定殘差，先以有序 logistic 的少量交互作用驗證，保留雙方交換時勝負對調、和局不變的結構；沒有必要立即採高維向量模型。未來才考慮 blade–chest，須處理和局、未見玩家、相剋資料稀疏與時間漂移。不能照搬其 StarCraft 隨機切分，也不能把當局開局或局後資料當預測特徵。

## EOMM：將結果機率轉成決策成本

Optimization Objective 將每位玩家視為圖上的節點，每條邊的權重是雙方預期離開風險的總和，再求最小權重完美匹配。風險以勝和負機率加權「給定結果後的離開機率」；後者用帶正則化的 logistic，以近期結果、間隔及遊玩量作輸入。案例以抽出的玩家狀態建立候選池，模擬賽果與離開後比較配法。[全文](https://arxiv.org/abs/1702.06820v1)。

**可借用而且已部分採用：明定邊成本、可行集合與雙方代價。** 本專案[固定批次比較](../results/matching-policy.md)用平衡及個人保護作目標；固定執棋顏色後是二分圖指派，不必為了引用 EOMM 改成一般圖匹配。

**不直接採用原流失分解。** 它依賴給定結果後，對手狀態不再影響離開；本專案的[假設檢查](../results/appendix/eomm-assumption.md)未支持這個簡化，且控制結果本身仍有中介混淆的識別限制。原文固定和局機率的設定也不適用本專案。其模擬反覆抽池比較不能替代真人政策試驗，更不能用增加模擬輪數消除模型估計與外推的不確定性。

## 候選優先序與停止條件

| 定位 | 方法 | 能增加的答案 | 何時不做 |
|---|---|---|---|
| 已執行條件參考 | 借用 Gee 的模型檢查與 Chowdhary 的連敗摘要 | 固定觀察特徵下，連敗及累積偏差如何隨時間、RD 與玩家權重改變？ | 未重建遞迴回饋，不作生成虛無檢定；見[結果](../results/sequence-diagnostics.md) |
| 有具體缺口才加 | Chen–Joachims 式成分比較與簡單情境交互作用 | 增量來自哪種局前資訊？ | 候選已在首局、間隔、RD 分析中完成，或只為增加模型數量 |
| 較大延伸 | Yuksel 的自適應評分 | 近況增量能否由另一種實力更新吸收？ | 無法公平處理歷史缺漏、和局、暖身及同一時間外測試集合 |
| 已完成，三種用時呈現 | Gee 式玩家 random intercept/slope | 允許玩家近況係數不同，能否改善相同測試對局的預測？ | 診斷與完整核對通過；相對共用係數模型僅中用時兩項誤差皆改善，短用時 score MSE 增加 |
| 沿用已有設計 | EOMM 的明確成本與受限匹配觀念 | 決策目標與誰承擔代價 | 不增加一套尚無可識別留存效益的政策模擬 |

原文已保存並完成方法可用性評讀；[序列診斷](../sequence-diagnostics.md)已用既有固定模型與已使用的測試資料執行。結果顯示權重及長視窗支持會影響戰績解讀，不能由單局預測改善直接得到個人異常門檻。沒有重估模型或使用新確認性樣本；新增確認性假設仍須另有未使用資料與登記。
