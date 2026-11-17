# 報告的研究目標與論證

## 核心目標

報告標題為「局前評分與近期戰績」，副標題為「一對一競技對局的預測模型與結論」。承接 `report/expected_score.typ` 的 ordinal logistic regression，先介紹續下選擇與上一局資訊，並與上次留下的模型比較。完成近期資訊與玩家差異模型的結果後，再介紹 TrueSkill，實際比較相同測試樣本上的預測。

| 研究問題 | 分析內容 | 交付 |
|---|---|---|
| 近期資訊是否有增量？ | 上一局資訊、歷史長度與玩家係數 | 相對評分基準及共用係數模型的預測比較 |
| 兩種模型如何預測？ | ordinal logistic 與 TrueSkill 的輸入、CDF、實力分布及更新 | 方法說明及本地 score MSE 比較 |
| 結果適用到哪裡？ | 時段、RD 與續下者 | 分組預測誤差與適用範圍 |

TrueSkill 的評估使用本地逐局預測，不以文獻排名代替實驗。簡報的測試評估僅呈現 score MSE，衡量期望得分的平方誤差；不評估完整三類機率分布的優劣。TrueSkill 沿用訓練內以 log loss 選出的參數，如實保留選參數方式。既有分析與前瞻登記的指標不變。預測改善不識別 tilt、休息效果或配對操控。

## 報告的模型選擇結論

以相同測試紀錄與權重的合併 score MSE 增量提出分用時建議：短用時使用加入上一局資訊的共用係數 ordinal logistic regression；中用時使用含玩家 random intercept 與 random slope 的 GLMM；長用時使用加入上一局結果及紀錄缺失指標的共用係數 ordinal logistic regression。三種用時的上一局模型相對評分基準都有合併改善；GLMM 只有中用時有明確額外改善。長用時上一局模型兩段平均差均為負，前段 CI 含零；中用時 GLMM 的改善主要出現在後段。分段結果描述穩定性，不以每段 CI 均排除零作為採用要求。

這是看過資料後的探索性取捨，不是事前登記的規則。CI 為未作多重比較校正的個別區間，不含重新訓練與選模變動。平均增量小，須同時呈現基準誤差與改善幅度，不將區間排除零寫成已證明具有大的實用收益。合併區間由逐筆損失差處理跨期玩家相依，不能平均分段 CI；摘要由 `scripts/baseline_report.py` 讀取已保存的預測產生。

目前不建議改用本次設定的 TrueSkill。此模型選擇使用已看過的測試資料，屬探索後的建議；整套選擇仍須以新資料驗證，不改寫既有前瞻登記與研究分析的判定。

## 方法與報告範圍

Gee 式玩家差異模型與共用係數的上一局模型比較，三種用時各自估計，統一呈現 correlated random intercept/slope。合併兩段後，中用時的 score MSE 改善；短用時的 score MSE 增加，長用時尚無明確改善。完整結果見 [玩家差異模型比較](results/hierarchical-prediction.md)。

TrueSkill 採一對一含和局的原始模型，以 arena 歷史對手與賽果更新玩家實力；hyperparameters 僅在訓練內選擇。相同樣本的本地比較未見改善，長用時 score MSE 較高；短、中用時合併 CI 含零。完整結果見 [TrueSkill 預測比較](results/trueskill-prediction.md)，設定見 [復現規格](trueskill-replication.md)。

| 已完成分析 | 報告位置及可以支持的判斷 |
|---|---|
| [評分基準](results/expected-score-baselines.md) | 回顧上次的含 RD ordinal logistic regression，作為加入近期資訊的比較基準 |
| [上一局資訊](results/temporal-validation.md) | 共用係數候選以 BIC 選擇；score MSE 與分組 CI |
| [玩家 intercept/slope](results/hierarchical-prediction.md) | 方法、整體與分組比較、預測幅度；保留不利結果 |
| [TrueSkill](results/trueskill-prediction.md) | 方法、訓練內參數選擇、相同測試紀錄的 score MSE 與分組比較 |
| [序列診斷](results/sequence-diagnostics.md) | 單局預測的用途限制；固定觀察條件的參考未重建評分回饋與續下 |

樹模型、MLP、GAM、RD interaction 與 spline 保留於研究文件，不占用方法介紹頁。各分析的基準、輸入與選模流程不同，不能以原始損失混合排名。Tang 的 Lichess 文獻結果保留於背景文件，其樣本、指標與本地實驗不同。

## 論證順序

先說明本次問題、樣本與續下選擇，再回顧 ordinal logistic regression，另頁交代三類機率與期望得分的評估，接著介紹近期資訊。簡報與口稿不介紹 dyadic covariance 的推導或修正前後的推論比較，也不討論新帳號與首次出現紀錄的預測。相依修正保留在實際分析中，BIC 與 CI 僅簡述所採用的口徑。

續下選擇另頁說明：只有實際繼續下棋者才有下一局結果，預測比較使用相同續下者與對局。結果限於實際續下者，不能據此推論所有玩家或提出休息建議。

上一局資訊先介紹模型與時間切分，再列出各次訓練實際選中的資訊，呈現分段 score MSE 差值、CI 與底色，以及合併增量、基準 MSE 與相對改善幅度。選中模型由 `aliases.recent` 取得，不使用完整候選的 `selected`。更長歷史、大廳與固定模型前瞻驗證未提供 score MSE 的既有比較，保留於研究文件，不列入這份簡報。

玩家差異模型以兩頁從共用係數引入個別 intercept/slope、共同 Normal distribution、partial pooling 與 posterior prediction。說明每局只進 likelihood 一次，測試沿用相同觀測權重；原文是 binary 模型，本題保留和局及雙方效果。另頁列出固定項、prior 與訓練權重的比較差異，並分開展示整套模型相對共用係數模型、random slope 相對 random-intercept-only 模型的結果。RD 分組頁只講該頁展示的分組；已見／未見玩家的結果保留於研究文件。機率調整幅度保留，明示比較對象改為評分基準，不視為誤差改善。

玩家差異比較之後，TrueSkill 方法分頁介紹實力與當局表現、表現差的三個機率區域、觀察賽果後的 Bayesian 更新，以及下一局前加入的實力變動。從 MLR 的平均與誤差分解切入，明示當局表現不可直接觀察；用同一個假設的 A、B 對局貫穿三類機率與爆冷更新，圖中數值由腳本計算，不能解讀為本地實驗結果。說明 posterior 的 Normal approximation 與忽略玩家間相關，再與 ordinal logistic 的輸入、機率及估計方式比較。接著呈現本地參數選擇、整體與分組結果；所有模型的 calibration 診斷保留於研究文件，不列入簡報與口稿。TrueSkill 僅使用 arena 歷史，平台評分則含更廣的歷史；時間使用 API 的最後下棋時間作結果可用時間代理，須交代測量限制。

TrueSkill 結果之前另頁說明兩套流程的歷史範圍、初始化、白方優勢與和局範圍。區分訓練內 log loss 選參數與測試 score MSE 比較；長用時的和局參數碰到候選上界，列為本次設定的限制。

後續討論單局預測與序列用途的界線。跨遊戲分析保留於研究文件，不列入簡報與口稿。結尾先並列同口徑的合併比較，再回答模型建議與適用範圍。所有頁面納入講述，不設附錄。

高 RD 對局指任一方達平台後來採用的門檻，回套於歷史資料；其餘稱雙方低 RD 對局。訓練期未見玩家與高 RD 玩家是不同分類。AoE2 依累計局數分組，不能視為同一族群。

## 證據與完成條件

支持報告結論的分析、診斷與比較須完成。不利或不明確的結果也是研究結論；不因加入模型便宣稱全面改善。玩家差異與 TrueSkill 比較使用已看過的資料，屬探索，CI 不含重新訓練與調參變動；既有前瞻登記維持原口徑。

新增確認性資料、完整序列生成、配對最佳化、停手建議與個人異常警示不列本次必要交付，相應成果保留於各自文件。

## 交付與檢查

`report/baseline.typ`、`report/baseline_script.md` 與 `report/baseline.pdf` 逐頁對應，不列時間估計。用詞依 [報告語言規範](internal/report-language.md)，統計術語使用英文，口稿直接說明資料、比較與結果。

數值由 `outputs/` 讀取。`scripts/trueskill_illustration.py` 產生教學例子的機率、更新與圖，並以直接積分核對 posterior moments。`scripts/hierarchical_report.py` 與 `scripts/trueskill_report.py` 整理本地模型的比較表與圖；原始模型、設定、逐筆預測與來源各自保存。這些產物用於重現歷史預測，未驗證現行平台的部署表現。

修改後編譯 Typst，執行 `make check-slides`、caption 檢查與 `make lint`，並核對口稿與投影片的標題、比較對象及結果一致。
