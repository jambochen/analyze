# TrueSkill 與 logistic 的預測比較

## 結論

在相同的 arena 測試紀錄與權重下，本次 TrueSkill 設定未優於含 RD 的 ordinal logistic 評分基準或共用係數的上一局模型。相對上一局模型，短用時的合併 log loss 與 score MSE 差異不明確；中用時的 log loss 較高，個別 CI 不含零，但三種用時的 Holm correction 後未達顯著；長用時兩項誤差皆較高。長用時兩段、高 RD 與雙方低 RD 組的 log loss 均較差。

共同分箱的 calibration 未呈現一致改善。這項結果支持保留現有 logistic 做法，不能推論所有 TrueSkill 設定皆不適用於西洋棋。原模型的有限候選、資料範圍與初始化固定於測試結果比較前，沒有依測試結果追加調參。

## 比較口徑

復現 Herbrich、Minka、Graepel 的一對一含和局 Gaussian filtering，並用本地資料實際更新玩家實力。主指標為三類 log loss，輔以 score MSE、RPS 與 calibration；未使用和局記半分的 binary cross entropy，也未引用 Tang 的數值作為本地預測結果。方法與候選見 [復現規格](../trueskill-replication.md)。

TrueSkill 由年初的共同 prior 開始，只用 arena 歷史對手與賽果，包含未進入評估的首局；不使用平台評分，也不另外估計白方優勢。logistic 使用平台局前評分、重建 RD、先手與近期資訊。平台評分包含 arena 以外的歷史，所以兩者比較同一預測任務，並非控制所有輸入後只比較 link function。

每個用時類別與時間切分，各以訓練期最後一月的局前 log loss 選 hyperparameters。參數固定後重放至測試段終點，測試期間允許用先前可用的賽果更新實力。validation 排除在測試開始時尚未達到結果時間代理的對局；測試資料不參與參數選擇。長用時的 draw probability 選到候選上界，因此結果也受預先限定候選範圍限制，沒有根據測試表現擴充範圍。

已使用資料的探索不改寫既有前瞻驗證。逐筆損失差與合併 CI 以觀察到的模型、hyperparameters 與更新路徑為條件，不包含整條序列重抽或重新調參的變動。表格底色依個別 CI，Holm p-value 則對三種用時的主要比較作校正。

## 時間與數值核對

以實際開局時間界定年份，PGN 月檔名稱不作切分依據。每局預測先於同時點的更新；只有時間嚴格早於開局的結果才進入當下狀態。API 的 `last_move_ms` 是結果可用時間的代理，不一定等於認輸或逾時公布時間，因此仍有時間測量限制。其他並行對局在結果可用時依序更新當時的玩家分布。

Rust 使用標準一對一 truncated Normal moments，CDF 採 `libm::erfc`，draw margin 由 SciPy inverse Normal CDF 計算。Python `trueskill` 的 SciPy backend 為獨立參考。勝負交換、機率加總、posterior variance、包含和局的逐局更新，以及未來賽果擾動與截斷序列檢查皆通過。正式重放沒有觸發機率下限裁切。

逐筆樣本、baseline 與 recent 的損失均與既有保存結果一致；同一局雙方視角的勝負機率互換，和局機率相同。共同 calibration 分箱的對局數與觀察比例亦一致。數值誤差與耗時保存在輸出，不以 CPU 使用率評斷運算完成與否。

## 重現與產物

依序執行：

- `scripts/trueskill_prepare.py`：由既有對局與 API 時間資料建構事件輸入，核對評估樣本。
- `scripts/explore/trueskill_check.py`：參考實作、因果時間順序與速度核對。
- `scripts/trueskill_prediction.py`：Rayon 平行候選，完成訓練內選擇、局前預測與比較。
- `scripts/trueskill_audit.py`：核對完整樣本、逐筆損失、機率對稱與合併結果。
- `scripts/trueskill_report.py`：輸出簡報表格與 calibration 圖。

Rust 核心在 `rust/likelihood/src/trueskill.rs`，建置方式沿用既有 `likelihood-rs`。大型工作依計算規範限制記憶體與執行緒，單一序列依序處理，獨立候選以 Rayon 並行。

`outputs/tables/trueskill_tuning.csv` 保存全部候選，`trueskill_parameters.csv` 保存實際選擇、測試樣本與耗時，`trueskill_comparison.csv` 保存分段、分組及合併比較，`trueskill_decision.csv` 保存主要 Holm correction。`trueskill_calibration.csv` 同時保留自身與基準共同分箱，`trueskill_final_audit.csv` 記錄完整核對。

`data/processed/trueskill_*_prediction.parquet` 保存三類預測與逐筆損失；`outputs/tables/trueskill_manifest.json` 保存設定與來源雜湊，可依事件輸入重放玩家狀態。輸出只含串接用代碼，不呈現帳號。
