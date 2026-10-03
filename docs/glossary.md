# 術語

文件、報告、口稿一律用此表的「用詞」欄。

## 原則

- **只用數理統計的標準術語**，不自造詞，也不用計量經濟或應用領域的慣用語
- **中文譯名不統一者用英文原文**，例如 simplex、cluster。譯名已定型者用中文，
  例如固定效應、信賴集合
- **不以術語帶價值判斷**。模型的成分、組別以編號或參數符號描述，解讀另外寫
- **領域詞不在此限**。監理優先清單、天氣歸因、申報、執飛者描述研究對象，不描述
  方法，照常使用

## 模型與估計

| 用詞 | 本題指什麼 | 不用 |
|---|---|---|
| stratum（層） | 機場 × 年月的分組，固定效應的單位 | 格子 |
| within transformation；within estimator | 層內減去層平均後以 OLS 估計，消去層的固定效應 | 格子內去均值 |
| binomial GLM with identity link | extensive 層的二元模型，係數即機率差 | 線性機率模型、LPM |
| 公司的主效應 | 公司因子各水準的係數 | 公司效應 |
| reference level（基準水準） | 係數比較的基準公司 DL；單體模型的基準成因 carrier | 對照組 |
| 因子水準間的比較 | 公司之間的係數比較 | 同儕、同儕比較 |
| two-part model：binary part、conditional part | 是否為正；給定為正時的條件分布 | extensive margin、intensive margin |
| simplex；compositional data | 四個成因分鐘數加總等於到達延誤的限制 | 單體、歸因單體 |
| multinomial logit；Poisson log-linear model | 成因分配的模型與其等價的估計形式 | — |
| contrast（對比） | 兩個成因係數之差，如 weather 對 NAS 的對比 | 歸因邊界 |
| 交互作用 | 公司 × 時段、公司 × 共變量的係數 | 漂移、行為型態 |
| mixture component 1、2 | 混合模型的兩個成分，以延誤長度係數的正負描述 | 可疑成分、機械成分 |
| complete separation；quasi-complete separation | 某層內結果全同，MLE 不存在 | — |

## 推論

| 用詞 | 本題指什麼 | 不用 |
|---|---|---|
| cluster-robust variance estimator（sandwich） | 標準誤的估計方式 | 聚類、叢集 |
| two-way cluster-robust | 同時對 stratum 與機尾號 cluster | 雙向聚類 |
| multiple comparisons；false discovery rate | 多家公司多個係數同時檢定 | — |
| shrinkage；empirical Bayes | 公司係數向整體平均收縮 | — |
| rank（秩）；confidence set for ranks | 公司在清單上的位置與其不確定性 | 名次、排序的區間 |
| test of restrictions | 由模型導出一個限制，檢驗資料是否違反 | 內生錨點、錨點 |
| monotonicity restriction | 延誤越長，整段歸因天氣的機率應遞減 | 延誤長度錨點 |
| 共變量 × 公司的交互作用檢定 | 前一航段剩餘緩衝對天氣歸因的影響是否因公司而異 | 外生衝擊、尾機錨點 |
| 低於第 q 分位數者 | 剩餘緩衝最小的航班 | 壓力組 |
| negative control | 預期無效應的情境；本題指兩組氣象皆非惡劣的航班 | 安慰劑、對照樣本 |
| empirical null | 由資料估計的虛無分布，取代理論上的零 | 經驗基準 |
| robustness（穩健性） | 換設定後結論是否改變 | 設定不敏感 |

## 敘述與估計的區分

| 用詞 | 本題指什麼 | 不用 |
|---|---|---|
| 敘述統計量 | Y1、Y2、Y3 三個比例的邊際分布 | 描述層 |
| 模型參數的估計 | 兩部模型與 multinomial 模型的係數 | 推論層 |
