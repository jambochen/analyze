# 玩家差異模型的復現與預測比較

## 問題與來源

依 [Gee 等人的 hierarchical logistic regression](https://doi.org/10.1515/jqas-2025-0035)，檢查允許玩家各自具有 intercept 與近期戰績 coefficient，能否改善目前的局前預測。復現的是模型結構與估計方法，不宣稱重現原論文不同玩家群的數值結果。原模型的 binary likelihood、相關的 bivariate Normal 玩家係數、Half-Normal scale prior 與 LKJ(2) correlation prior 先以模擬驗證。

## 對本資料的必要調整

- 保留勝、和、負，沿用現有 ordinal logistic 的評分、RD 與隨平均評分變動的和局帶寬。固定效果共同估計，不把全年基準當成已知 offset。
- 使用本場錦標賽的上一局；近期變數為上一局是否獲勝減去局前累積勝率。累積勝率只用資料可觀察期間、同用時類別、嚴格早於本局開局的 rated 對局；同時開局的紀錄不互相提供結果。對手缺上一局時該近期變數設零，另保留缺失指標。
- 固定效果保留前局和局、勝局、缺失指標及評分變動。玩家偏差為 `a_i - a_j + b_i h_i - b_j h_j`；`(a,b)` 的共同平均為零，固定近況項估計整體關係。這是雙方共享玩家參數的延伸，不是原論文單一 focal player 的模型。
- 同局雙方視角只進一次 Bayesian likelihood，避免把相同結果當成兩個獨立觀察；評估則保留現行完整玩家局與權重。這項估計權重差異必須揭露。
- 訓練期未出現的玩家，預測時積分其 population distribution，不以其測試結果估計個別係數。已見玩家也使用 posterior predictive probability，不僅代入 posterior mean。
- 固定效果的 design matrix 以訓練共變數作可逆 SVD 白化，白化後優勢係數使用 Normal(0, 2.5) prior；log-width 係數使用 Normal prior，中心對應固定帶寬 exp(-2)、standard deviation 為一。這些固定效果 prior 是配合本研究新增變數的設定，並非原論文原尺度的 prior。

## 比較與判定

沿用現有兩段時間切分與三種用時類別。固定比較 correlated random intercept/slope、只有 random intercept 的對照，以及現行 BIC 選定模型與簡單近況模型。新候選不加入 spline，不根據測試結果調整 prior、近期視窗或模型形式。這是已使用資料的探索，不改寫既有前瞻登記。

主要指標為 log loss，輔以 score MSE、RPS 與 calibration。逐局 paired loss difference 使用既有 dyadic standard error；另分雙方已見與含未見玩家、高 RD 與雙方低 RD。兩段合併以玩家局加權，區間條件於已配適模型，不含重估變動；訓練期間重疊的限制保留。

校準同時呈現各模型自身的機率分箱，以及現有評分基準機率的共同分箱；後者固定對局組成，核對各模型的局數與觀察比例一致。分箱偏差只作描述，不以各自不同分箱的摘要值直接排名。已保存的預測可由 `scripts/hierarchical_calibration.py` 重整分箱，不重新配適或抽樣。

主要新候選固定為 correlated random intercept/slope；random intercept 僅作機制對照，不依測試結果從兩者中挑一個。完整候選 BIC 對照的改善以用時類別分別判定：相對現行選定模型，兩段 log loss 的差都須為負，合併差的雙尾檢定經三種用時類別 Holm 校正後須支持改善，且不能有明確的 score MSE 惡化。若只有 random intercept 有改善，不能把改善歸因於近期戰績的玩家差異。沒有達到判準則不宣稱優於此對照，完整記錄失敗結果。簡報統一介紹三種用時的論文模型，使用共用係數的上一局模型作比較；呈現範圍不由這項 BIC 對照判定，兩種對照的結果分開記錄，見 [結果](results/hierarchical-prediction.md)。

## 計算與驗證

先檢查 binary 與 ordinal likelihood、交換雙方後的機率、模擬參數恢復，再實測一個配適的耗時。正式估計使用四條 NUTS chains，檢查 divergence、R-hat、ESS 與 posterior predictive calculation 的 Monte Carlo 穩定性；診斷不合格時先解決估計問題，不解讀預測排名。資料與結果另存，不覆寫既有模型產物。

`scripts/explore/hierarchical_prepare.py` 建立資料，`scripts/explore/hierarchical_check.py` 驗證模型，`scripts/explore/hierarchical_benchmark.py` 實測耗時。`scripts/explore/hierarchical_run.py` 對同一用時類別、同一切分的兩種模型平行配適，各自完成診斷與評估後，才開始下一組；任一工作失敗即不再送出後續組合。`HIERARCHICAL_WORKERS=1` 可改成逐一執行。整批分析須依 `docs/internal/computing.md` 共用記憶體上限，並限制每個工作的執行緒。

若某個配適未通過診斷，單獨處理後以 `HIERARCHICAL_START` 指定下一個 `speed,fold,kind` 組合，避免重跑已驗證的配適。posterior predictive calculation 依鏈平行；同一對局先核對雙方的設計矩陣與近期資訊互相對應，再計算一次機率，交換勝、負機率還原另一方。評估仍使用全部玩家局及原權重。`scripts/explore/hierarchical_predictive_check.py` 與保存的逐列計算結果核對兩種模型、新玩家及已見玩家的機率。

正式配適由 `scripts/hierarchical_prediction.py` 執行，`HIERARCHICAL_SPEED`、`HIERARCHICAL_FOLD` 與 `HIERARCHICAL_KIND` 指定工作。`scripts/hierarchical_evaluate.py` 使用同一組設定讀取 posterior，檢查零 divergence、全部保留參數的 R-hat 小於 1.01 與 bulk ESS 大於 400，才計算預測。新玩家的積分使用 Gauss–Hermite quadrature，並核對增加節點後的數值差異；四條鏈分別輸出預測損失供 Monte Carlo 穩定性檢查。既有模型的預測須能重現原表的損失差，才能將新模型與其比較。

完整 log posterior 與 gradient 由 `rust/likelihood/src/hierarchical.rs` 計算，包含 ordinal likelihood、全部 prior 及參數變換的 Jacobian。Rayon 依固定資料區塊平行累加，再按固定順序合併。`scripts/hierarchical_rust.py` 將 Rust 函式接到 nutpie 的 Rust NUTS sampler，保留相同模型、prior 與診斷門檻。每個配適預設六個 Rayon threads，同時執行四條 chains；兩個配適共用既定記憶體上限。

`scripts/explore/hierarchical_rust_check.py` 核對兩種模型在不同參數位置的 log posterior 差、全部 gradient、玩家係數輸出與交換雙方後的對稱性。PyTensor 對固定 prior scale 的常數摺疊可產生與參數無關的微小 log posterior 差，因此另核對扣除常數後的差值與梯度。`scripts/explore/hierarchical_rust_benchmark.py` 使用完整訓練資料核對數值及單次計算耗時；`scripts/explore/hierarchical_rust_recovery.py` 驗證 Rust 抽樣的模擬參數恢復與收斂。實際配適的 metadata 記錄 backend 與來源雜湊，已完成且通過檢查的配適保留，不因更換等價計算後端重跑。

模型與逐列預測在 `data/processed/hierarchical_*`，診斷與比較表在 `outputs/tables/hierarchical_*`。`hierarchical_prediction_decision.csv` 僅表示預先寫定的損失判準是否通過；解讀時仍須閱讀分組、校準與 Monte Carlo 穩定性的結果。

`scripts/hierarchical_audit.py` 核對實際測試對局數、雙方視角的機率對稱、儲存損失與比較表、共同分箱及收斂診斷。四條鏈的 log loss 差異所估的 Monte Carlo standard error，須小於對應 dyadic standard error 的十分之一，避免數值波動左右比較。合併判定前強制跑全部用時類別的核對；指定單一用時類別的檢查只表示該部分通過，不能作整體結案依據。
