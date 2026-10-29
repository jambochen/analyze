# 期望勝率的三個基準

腳本：`scripts/expected_score_baselines.py`，切點形式的診斷在 `scripts/threshold_rating_form.py`；修正的概似比檢定在共用的 `scripts/likelihood.py`，其檢定水準以 `scripts/adjusted_lrt_size.py` 的模擬核對。表在 `outputs/tables/expected_score_baselines_*.csv`、`outputs/tables/threshold_rating_form_*.csv` 與 `outputs/tables/adjusted_lrt_size.csv`。本文件只寫定性結論，數值一律見對應的輸出檔。

## 問題

主問題以「系統的期望勝率」為基準，量剛輸之後的表現是否低於期望。基準本身若在某些對局條件下系統性地錯，後續的殘差會混入基準的錯。`docs/results/calibration.md` 已確認 Elo 形式高估強方、白方有固定優勢、和局機率隨等級變化；`docs/results/novice-period.md` 發現校準斜率隨 rating deviation（RD）連續變化，新手的偏差不宜靠門檻排除。這裡比較三個基準，決定主問題用哪一個：

- **有序 logistic**：`docs/results/calibration.md` 的主模型，分差、平均等級分、先手
- **Glicko 形式**：分差依雙方合成的 RD 往 0 收縮，即 Glickman 對一局結果的預測式
- **加入 RD 的有序 logistic**：在有序 logistic 上加入 Glicko 的收縮項，讓分差的斜率隨 RD 變化

三者的定義、為什麼用合成的 RD、RD 的尺度，見 `docs/methods.md`「期望勝率的三個基準」。

## 資料與樣本

- 2014 年 Bullet、Blitz、Classical 的有鐘對局，類別依 `speed`，含新手對局；RD 取自重放（`docs/data/glicko-replay.md`），只用重放的速度類別與對局相同者
- 概似比檢定在各類別抽一個固定大小的簡單隨機樣本上估計，與 `docs/results/calibration.md` 相同；另以排除含左截斷一方的對局重做，因為左截斷帳號前段的 RD 是設定值
- 樣本外評估：1 至 6 月估計，7 至 12 月全部對局評估，依新手（任一方 RD 達 provisional 門檻）與否分組
- Elo 與 Glicko 期望得分本身的校準迴歸用該類別全部的對局

## 做法

### 巢狀關係與檢定

有序 logistic 與 Glicko 形式都是加入 RD 的有序 logistic 的特例：前者令收縮項的係數為 0，後者令分差、等級分與和局帶寬的等級分項為 0、只留收縮項與切點。兩者各以概似比檢定與完整模型比較，不需要非巢狀的檢定。

同一玩家下很多局，同一局又牽涉兩位玩家，觀察值不獨立，一般概似比統計量的虛無分布不是卡方。統計量照常計算，虛無分布改為加權卡方 $\sum_j \lambda_j \chi^2_1$，$\lambda_j$ 由完整模型的 Hessian 與 dyadic 的 meat 求得（`docs/methods.md`「檢定與區間」）。df 為 1 時有精確解；df 較大時以特徵函數的數值積分求尾機率，尾機率小到數值積分不可靠時，改報 $P(\lambda_{\max}\chi^2_{df} > x)$ 這個嚴格的上界。各檢定以 Holm 法控制族錯誤率。

### 修正檢定的檢定水準

在模擬資料上核對：對局結果受未觀測的玩家實力影響，受檢的共變數是玩家層的，虛無假設為真。玩家數與活躍度的集中程度分四種情境，以「各玩家局數占比平方和的倒數」量有效玩家數，與真實樣本的有效玩家數對照。

### 期望得分本身的校準

Glicko 只給期望得分，有序 logistic 的寫法需要自由估計尺度（有和局時累積 logit 的斜率較陡）。為了直接檢驗 Glicko 原式，另以得分為反應的 logistic 迴歸估 Elo 與 Glicko 期望得分的校準迴歸的截距與斜率，依新手與否分層。

### 切點對平均等級分的形式

有序模型的切點由白方優勢 $a$ 與和局帶寬 $w$ 組成，兩者都設為平均等級分的二次式（$w$ 經指數）。為檢查這個形式，把平均等級分分成 20 個等量組，每組各自估 $a$ 與 $\log w$，不預設形式，其餘項與加入 RD 的有序 logistic 相同，再與線性、二次的擬合對照。檢定兩件事：二次項是否需要（二次對線性），二次是否足夠（二次加上分組虛擬變數對二次）。四個檢定 × 三個速度類別為一族，以修正的概似比檢定與 Holm 法判定。

### 樣本外評估

三類結果的對數分數與 ranked probability score，與有序 logistic 的差值以 dyadic sandwich 求標準誤。

## 結論

### 修正的概似比檢定：有效玩家數夠多時接近名目水準

一般的概似比檢定在所有情境都嚴重過度拒絕。修正後的拒絕率隨有效玩家數增加而接近名目水準：有效玩家數少、少數玩家占大部分對局時仍偏寬鬆，有效玩家數到數千時與名目水準一致（`adjusted_lrt_size.csv`）。偏寬鬆的方向與已知的性質一致：群數少時 sandwich 低估變異數，修正所用的權重 $\lambda_j$ 也就偏小。

真實樣本中，Blitz 與 Classical 的有效玩家數落在接近名目水準的範圍；Bullet 的有效玩家數較少、$\lambda$ 較大，落在略為寬鬆的範圍。檢定在 Bullet 上若接近顯著邊緣，結論須保留。

### 兩個受限模型都被拒絕

三個速度類別、全部對局與排除左截斷兩種樣本下，有序 logistic 與 Glicko 形式對完整模型的概似比檢定都拒絕，修正後的 p 值仍遠低於任何常用的水準（`expected_score_baselines_tests.csv`）。Bullet 的檢定水準略為寬鬆，不影響這個結論。

- **有序 logistic 被拒絕**：RD 帶有分差、等級分與先手之外的資訊。期望勝率在雙方 RD 大時應往 0.5 收縮，有序 logistic 不收縮
- **Glicko 形式被拒絕**：只靠 RD 收縮不夠，Elo 式的分差斜率偏差、和局帶寬隨等級的變化與先手優勢都還在

### RD 的收縮比 Glicko 的式子更強

完整模型中，分差的斜率隨合成的 RD 增大而下降，下降得比 Glicko 的 $g(\sigma)$ 所隱含的更快（`expected_score_baselines_coef.csv` 的 `slope_diff` 與 `slope_g_diff`）。校準迴歸給出同一個方向：Glicko 的期望得分比 Elo 的更接近校準，新手對局尤其明顯，但兩者的斜率都仍低於 1，仍高估強方（`expected_score_baselines_cox.csv`）。也就是說，Glicko 承認的不確定性方向對，幅度不足。

### 樣本外：加入 RD 的模型在每一組都最好

7 至 12 月的評估中，加入 RD 的有序 logistic 在全部、新手、老手三組的對數分數與 ranked probability score 都優於有序 logistic（`expected_score_baselines_prediction.csv`）。改善主要來自新手對局，老手對局的改善小但仍在標準誤之外。Glicko 形式在新手對局上優於有序 logistic，在老手對局上反而較差，全體與有序 logistic 相近：它的收縮幫了新手，但沒有 Elo 式的斜率修正與等級分項，老手吃虧。

### 和局帶寬需要二次，白方優勢不需要

和局帶寬的對數在 Classical 低等級段持平、之後上升，彎曲明顯，Blitz 也有，二次項校正後拒絕為 0；Bullet 近乎直線，二次項不需要。二次加分組的檢定在 Bullet 與 Classical 也拒絕，但分組估計與二次曲線的偏離相對於整體趨勢很小，百萬局的樣本下小偏離也會被偵測（`threshold_rating_form_tests.csv`）。白方優勢的分組估計大致沿直線，Classical 隨等級下降；二次項在三個速度類別校正後都不拒絕為 0，Bullet 的彎曲只來自最高兩組。二次式在兩端外推時翹起，那裡資料少，不作解讀。

## 對後續分析的意涵

- **主基準**：加入 RD 的有序 logistic。檢定與樣本外評估指向同一個方向，沒有互相衝突。在估計前局效果模型之前決定，寫進登記文件
- **另兩者為敏感度分析**：主問題的結論在三個基準下若不一致，如實寫出結論依賴基準
- **新手對局可以留在主樣本**：完整模型讓期望勝率隨 RD 收縮，新手的偏差由基準吸收，不必以門檻排除；門檻只用於描述與敏感度分析（`docs/results/novice-period.md`）
- **Bullet 上的檢定**：修正的檢定略為寬鬆，接近顯著邊緣的結果須保留，或改用小樣本修正的 sandwich
- **電競的對應**：系統自己算的不確定性（RD）是有用的資訊，但系統用它收縮期望勝率的幅度不夠；定級期結束後，期望勝率仍高估強方

## 限制

- **收縮項的形式固定為 $g(\sigma)$**：完整模型只多一個係數，不同的收縮形狀（例如依雙方 RD 各自收縮）未比較
- **RD 取自重放**：重放與表頭不相符的範圍（左截斷帳號的前段、2014 年 pool 試行期、線上期間的 Classical）中，RD 的品質較差；排除左截斷的敏感度分析結論相同，pool 期間未另外排除
- **檢定水準的模擬只是一種相依結構**：未觀測的玩家實力加上活躍度的集中；真實資料的相依可能更複雜，例如同一晚的狀態
- **樣本外評估只有一次切分**：1 至 6 月對 7 至 12 月，期間 Lichess 於 7 月底重算等級分（`docs/results/calibration.md`）
