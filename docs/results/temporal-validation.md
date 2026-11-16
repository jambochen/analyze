# 時間外預測與模型選擇結果

主分析以相依修正 BIC 選模型、AIC 作對照，巢狀比較用相依修正 LRT。執行規格見 `docs/temporal-validation.md`，估計腳本為 `scripts/likelihood_selection.py`，報告圖為 `scripts/likelihood_selection_report.py`。數值以 `outputs/tables/likelihood_selection_*.csv` 為準。

## 相同訓練樣本的共同估計

Lichess 每個速度、每個向前切分，都在相同 arena 玩家局共同估計評分基準與近況／RD 項。所有基準參數自由估計，和局帶寬參數亦重估；全站係數只作數值起點。概似、資訊矩陣與 dyadic score 共變異都使用同一批訓練列，避免固定估計基準卻忽略其不確定性。

BIC 使用唯一對局數為主尺度，AIC 作對照，另列玩家數及玩家局數的 BIC。全部有限候選都參與比較，LRT 不先淘汰候選。這是依賴資料上的工作資訊準則，並非精確 Bayes factor；所有時段已使用，結果屬探索與穩健性分析。

## BIC、AIC 與巢狀 LRT

| 速度 | 前段 BIC 選擇 | 後段 BIC 選擇 |
|---|---|---|
| Bullet | 評分變動與 RD 樣條 | 結果、缺歷史、評分變動與 RD 樣條 |
| Blitz | 評分變動與 RD 樣條 | 結果、缺歷史、評分變動與 RD 樣條 |
| Classical | 結果與缺歷史 | 結果與缺歷史 |

AIC 在 Bullet、Blitz 前段保留較多近況項，選入結果、缺歷史、評分變動與樣條；後段及 Classical 與 BIC 一致。依玩家數計 BIC 時，Bullet 前段也改選含結果的樣條；依玩家局數則與對局數尺度一致。其餘切分不隨這些尺度改變選擇。因此 Bullet 前段的近況項取捨較敏感，不把單一尺度的結果當成穩定定案。

預定的巢狀比較顯示，在含結果與評分變動的模型上增加 RD 交互作用及升降不對稱，Bullet、Blitz 在 Holm 校正後仍有配適證據，Classical 則不足。這不代表交互作用在全部候選中最合適，樣條與交互作用彼此非巢狀，主模型仍依 BIC。

LRT 的原始 p、Holm p、數值計算方式及保守上界均保存。這些增項比較與選模使用相同訓練資料，不作選模後的新確認性檢定。

## 時間外結果

- 簡單近況族中的 BIC 最佳者，在 Bullet、Blitz 的全體 NLL 與得分 MSE 均有跨期增量。Bullet 前段高 RD 的 NLL 區間仍含零，不能以全體結果代替所有分組。Classical 選入結果與缺歷史，前段改善不明確，後段改善較明確。
- 全候選 BIC 選定模型在 Bullet 前段的全體與高 RD 組反而較基準差；Blitz、Classical 前段的全體差值區間含零。後段三種速度的全體損失皆改善。訓練 BIC 最小並不保證時間外改善。
- AIC 與 BIC 的前段候選不同，但都未避開 Bullet 前段的惡化；相同模型的測試差值為零，不解讀為不同模型具有相同母體風險。主準則不依測試表現更換。

這些比較中的基準及所有候選都在 arena 共同估計，不能將損失直接接到全站固定基準流程，或把變化只歸因於改了選模準則。基準估計樣本與候選比較流程也有差異。

圖：[近況增量](../../outputs/figures/likelihood_selection_report_metrics.svg)、[RD 候選](../../outputs/figures/likelihood_selection_report_branches.svg)、[BIC 模型](../../outputs/figures/likelihood_selection_report_selected.svg)、[AIC 與 BIC 的測試差值](../../outputs/figures/likelihood_selection_report_criteria_test.svg)。

## 機率、幅度與校準

相對損失改善與勝率調整分開呈現。BIC 選出的模型多數對局修正較小，少數調整較大；調整幅度不能替代準確度，也不能稱為玩家勝率增加。匿名情境沿用訓練期局前特徵選例，各模型都使用自己的共同估計基準與近況係數。

分箱偏差沒有在各結果類別同步改善。Bullet 前段的勝負偏差變大，Classical 後段也未全面改善；分箱偏差與對數損失可能給出不同訊息。共同基準分箱只比較相同對局的條件平均，自身預測分箱則描述各模型校準，兩者皆受切點與有限樣本影響。

圖：[相對改善](../../outputs/figures/likelihood_selection_report_gain.svg)、[勝率調整](../../outputs/figures/likelihood_selection_report_changes.svg)、[得分條件偏差](../../outputs/figures/likelihood_selection_report_score.svg)、[三類校準](../../outputs/figures/likelihood_selection_report_classes.svg)。

## 固定基準的參照實驗

`temporal_prediction_*`、`rd_model_comparison_*` 與 `prediction_value_*` 保留固定基準、訓練內驗證 NLL 選模的成果。該流程的 Bullet、Blitz 簡單近況在兩段皆有全體增量，Classical 保留基準；RD 共同比較前段選樣條、後段選交互作用，Bullet 前段也出現測試惡化。這是不同流程的結果，與本文件主分析的 BIC 模型分開解讀。

既有 HMM、樹模型、MLP、GAM 與玩家隨機截距只作模型取捨參照，不以不同基準或選模流程的損失直接排名。2015 年前瞻登記仍只驗證當時固定的近況模型，不替新的 BIC 選擇提供確認性背書。

## 跨遊戲結果

- **OGS**：blitz 與 live 的校準基準均比系統期望好，近況修正再改善全體與新手的預測；新手延伸在兩個速度類別都有額外增量。老手的改善不一致：近況在 blitz 有改善、live 的區間含零；新手延伸在 live 的老手也有小幅改善，blitz 的區間含零。這些是結束時間排序下的結果，不能直接當成開局前部署驗證。
- **AoE2**：校準基準相對 Elo 代理的整體差異區間含零，新手組反而較差。加入近況後，全體與新手的預測改善，老手組點估計較差、區間含零，不寫成確立的傷害。內部選擇保留上一局結果與缺少歷史的差，未保留等級分變動。沒有 RD，不執行 RD 延伸；其與近況模型的零差值代表同一模型。
- **適用範圍**：近況的增量並非在每個速度、每種玩家都一致。新手的改善較集中，老手需分遊戲評估；不能將同一修正套用所有群體。各遊戲的新手定義、歷史可用性與評分尺度不同，不跨遊戲排列原始損失的高低。

## 模型交付與核對

`outputs/tables/prediction_models.json` 保存每個候選各自的基準係數、近況欄位、係數、樣條轉換及 BIC／AIC 選擇。`models` 的鍵是固定候選名稱；報告摘要的 `recent`、`interaction`、`spline` 等標籤先經 `aliases` 解析，不能直接按同名鍵重建。各模型的 `baseline_coef` 不共用。這是歷史切分下的模型，不宣稱已驗證現行平台部署。

| 輸出 | 內容 |
|---|---|
| `likelihood_selection_fits` | 各候選訓練 NLL、參數數與有效參數數、普通及修正資訊準則、配適診斷 |
| `likelihood_selection_selection` | 所有候選相對各準則最小值的差、各準則選擇 |
| `likelihood_selection_lrt` | 巢狀 LR、受檢維度、相依修正 p、尾端上界及 Holm 校正 |
| `likelihood_selection_sample` | 訓練／測試時間範圍、實際對局及玩家局數，核對既有樣本 |
| `likelihood_selection_prediction` | 固定模型的分組測試 NLL、RPS、MSE 與差值區間 |
| `likelihood_selection_calibration`、`calibration_bins` | 各分組、機率目標、自身及共同分箱偏差 |
| `likelihood_selection_changes`、`scenarios` | 機率調整與匿名訓練情境；dr 以百分為單位 |
| `likelihood_selection_report_models` | 主模型、AIC 對照及 BIC 樣本數尺度的報告摘要 |

正式估計通過配適一階條件、H 與 sandwich 正定、概似與預測機率一致、樣本與時間順序、JSON 重建核對。測試區間條件於已估模型，不包含重新訓練或選模變異；相關時段已使用，不作新的確認性宣稱。
