# TrueSkill 的本地預測比較

## 模型與問題

依 Herbrich、Minka、Graepel 的 [TrueSkill 原始模型](https://www.microsoft.com/en-us/research/wp-content/uploads/2007/01/NIPS2006_0688.pdf)，復現一對一、含和局的 Gaussian skill filtering。每位玩家具有 Normal skill distribution，當局表現加入 Normal noise，以正負 draw margin 分出負、和、勝；賽果出現後作 moment matching，更新雙方的近似 Normal posterior。不是 TrueSkill Through Time，不使用未來對局作 smoothing。

比較問題是：只由 arena 歷史對手與賽果更新的 TrueSkill，是否比現有含 RD 的 ordinal logistic 評分基準與共用係數的上一局模型預測得準。兩者輸入與更新方式不同；平台評分包含 arena 以外的歷史，不能將整體差異只歸因於 Normal 與 logistic CDF。TrueSkill 不加入平台評分或白方優勢參數，保留原模型結構。

## 資料與時間

使用既有 2014 年 arena 對局，按短、中、長用時獨立維護玩家狀態。初始實力平均為零，所有玩家採相同 prior，不用測試期表現初始化。以完整 arena 對局序列更新，包括不符合續下者評估條件的對局；測試則嚴格使用目前報告的相同紀錄與權重，同局只更新一次。

開局時計算三類機率，僅使用嚴格早於開局、已完成的對局結果。完成時間取既有 API 的 `last_move_ms`，同時完成時依 game ID 排序；同一時間的開局先預測再更新。資料年份與測試切分依開局時間，不依 PGN 月檔名稱。保存時間與樣本核對結果；不得將尚在進行的對局結果提前寫入狀態。`last_move_ms` 為 API 提供的時間代理，不保證等於認輸或逾時結果實際公布的毫秒。

沿用兩段時間切分：前段測試為七至九月，後段為十至十二月。分別用訓練期間最後一個月（六月、九月）的局前預測 log loss 選 hyperparameters；此前對局用於累積 skill posterior。選定後從年初重放至測試段終點，hyperparameters 固定，測試期間仍按已完成賽果更新 skill。這與 logistic 固定 regression coefficients、局前評分與近期資訊持續更新的流程對照。既有資料已使用過，結果屬探索，不改動前瞻登記。

## 選擇與評估

固定 beta 為一以識別尺度，初始 sigma 候選為一、二、四，tau 候選為零、0.02、0.1；draw probability 參數候選為 0.01、0.03、0.1，以 inverse Normal CDF 換算 draw margin。這些是方法設定，實際選中值與樣本量寫入輸出。每次重放各玩家每次更新前 variance 增加 tau 平方，預測分母也使用相同的 prior variance。有限候選在讀取測試結果前固定；不因測試表現增添候選。

主要指標為勝、和、負三類 log loss，即實際類別機率的負對數；輔以 score MSE、RPS 與 calibration。不使用和局記半分的 binary cross entropy，不將 Tang 的文獻損失當成本地數值。內部選擇也只用三類 log loss。

保存逐筆三類機率、測試期與玩家代碼，和現有 baseline、recent 模型作 paired comparison。按用時、時段、高 RD 與雙方低 RD 分組；合併兩段仍處理共享玩家。CI 條件於實際已配適模型與觀察到的更新路徑，不包含重新調參或整條序列重放的變動。三種用時對 recent 的合併 log loss 檢定採 Holm correction，其餘為診斷。呈現不利結果，不以測試段重新挑 hyperparameters。

## 計算與核對

逐局 filtering 無法向量化。先量測 Python 套件的逐局更新，若大量候選重放成為瓶頸，以 Rust 執行；Rayon 僅並行獨立候選，各序列內按事件順序更新。Python 保留資料整理、參考實作、評估與產圖。

核對勝負交換對稱、三類機率和為一、posterior variance 為正、含和局的更新與 Python `trueskill` 一致。另以未來結果擾動和截斷序列檢查局前預測不變。先跑代表性工作估計耗時，再在記憶體與執行緒上限內完成全部候選及測試；保存程式版本、hyperparameters、輸入來源與結果。
