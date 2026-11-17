# 玩家 intercept/slope 模型的預測比較

## 比較範圍

依 Gee 等人的 correlated random intercept/slope 結構實作。原文的 binary 模型及保留和局的雙方 ordinal 延伸，均通過模擬恢復、likelihood 比對與抽樣診斷。這是方法復現及本資料的延伸，並非原論文數值結果的重現。模型與估計規格見 [復現規格](../hierarchical-replication.md)。

簡報統一介紹三種用時的論文模型，與共用係數的上一局模型比較。這項呈現選擇不表示它在三類都勝出。下列段落除另標明外，差值均為 correlated random intercept/slope 減去共用係數模型；負值表示誤差較小。兩段合併依觀測筆數加權，log loss 的三種用時比較使用 Holm correction。資料已用於探索，CI 條件於已配適模型，不含重新訓練的變動。

## 整體預測

| 用時 | 前段 | 後段 | 兩段合併 |
|---|---|---|---|
| 短 | log loss 與 score MSE 平均差為正，CI 包含零 | log loss 平均差為正、CI 包含零；score MSE 的 CI 位於零以上 | log loss 無明確改善；score MSE 增加 |
| 中 | 兩項差值 CI 均包含零 | 兩項 CI 均位於零以下 | log loss 經 Holm correction 後支持改善，score MSE 亦降低 |
| 長 | 兩項平均差為正，CI 包含零 | 兩項平均差為正，CI 包含零 | 兩項均無明確改善 |

目前結果不支持全面取代共用係數模型。中用時的改善主要出現在後段，前段單獨看仍不明確。

## Random slope 的增量

只有 random intercept 的模型作為對照，不依測試表現從兩種結構中挑選。加入 random slope 後，三種用時合併的 log loss 與 score MSE 差值 CI 均位於零以下。中用時前段的增量仍不明確，後段支持改善。

整體模型和共用係數模型的差異包含 prior 與 likelihood 權重，不能把全部改善歸因於 random slope；只有相對 random intercept 的比較對應新增 slope 的增量。

## 分組與 calibration

分組只作探索性診斷，不據此調整 prior 或改成只對已見玩家套用。

| 用時與時段 | 相對共用係數模型的結果 |
|---|---|
| 短，前段 | 高 RD 與雙方低 RD 的兩項 CI 均包含零；雙方已見組的 score MSE 降低，log loss 不明確 |
| 短，後段 | 雙方低 RD 組的兩項誤差增加；雙方已見組降低；含未見玩家組增加 |
| 中，前段 | 各分組的兩項 CI 均包含零 |
| 中，後段 | 高 RD 組的 log loss 降低，score MSE 不明確；雙方低 RD、雙方已見及含未見玩家組的兩項誤差均降低 |
| 長，前段 | 含未見玩家組的兩項誤差增加，其餘分組不明確 |
| 長，後段 | 高 RD 組的兩項誤差增加，其餘分組不明確 |

報告的 calibration 圖使用評分基準機率的共同分箱，固定各箱對局組成。以下比較玩家差異模型與評分基準的加權平均絕對偏差，是分箱描述，不是正式 calibration 檢定。

| 用時與時段 | 偏差較小 | 偏差較大 |
|---|---|---|
| 短，前段 | 期望得分、勝局機率 | 和局、負局機率 |
| 短，後段 | 無 | 期望得分與勝、和、負機率 |
| 中，前段 | 勝局、和局機率 | 期望得分、負局機率 |
| 中，後段 | 期望得分與勝、和、負機率 | 無 |
| 長，兩段 | 和局機率 | 期望得分、勝局與負局機率 |

## 與完整候選 BIC 模型的研究比較

此處的對照包含 RD interaction 與 spline，與簡報的共用係數對照不同。比較依 [規格](../hierarchical-replication.md)要求兩段平均 log loss 皆下降、合併檢定經 Holm correction 後支持改善，且全體 score MSE 未明確惡化。

僅中用時符合這項判準；短用時後段兩項誤差增加，長用時無明確改善。中用時相對此 BIC 對照的合併 log loss 降低，score MSE 不明確，後段高 RD 組的兩項誤差增加。這些結論只適用於此對照，不能套到前述共用係數比較。

完整結果保存在 `hierarchical_prediction_decision.csv`、`hierarchical_prediction_pooled.csv` 及各配適的比較表。簡報不介紹 RD 分支，分析產物保留供查核。

## 診斷與限制

Rust log posterior 與 gradient 通過原模型比對、玩家係數輸出、雙方交換對稱性及完整訓練資料數值核對。Rust NUTS 的模擬參數恢復與收斂通過既定門檻。

全部配適的保留參數符合 R-hat 與 bulk ESS 門檻，沒有 divergence。未見玩家的 quadrature 通過增加節點的核對，鏈間 Monte Carlo standard error 符合精度要求。完整產物核對包含逐局損失、測試樣本、共同分箱與基準預測重現。

三種用時使用相同模型結構，各自估計參數；沒有依 RD 或是否已見另選模型。既有固定模型的前瞻驗證未涵蓋此模型，不將探索比較寫成新的確認性證據。

## 輸出與重現

- `hierarchical_<speed>_<fold>_<kind>_diagnostics.csv`、`_fit.json`：診斷、配適設定及來源雜湊，位於 `outputs/tables/`。
- 同一前綴的 `_prediction.csv`、`_calibration.csv`、`_mc.csv`：各比較基準的損失、calibration 與鏈間穩定性。
- `data/processed/hierarchical_<speed>_<fold>_<kind>.npz`、`_prediction.parquet`：posterior、轉換及逐列預測。
- `scripts/hierarchical_summary.py`：核對全部組合，產出合併比較及完整候選 BIC 對照的判定。
- `scripts/hierarchical_report.py`：整理三種用時的論文模型；`hierarchical_report_models.json` 保存來源，`hierarchical_report_comparison.csv` 與 `hierarchical_report_periods.csv` 保存共用係數對照的合併及分段結果。
