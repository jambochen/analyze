# 術語

本表供統計術語查閱；本研究的正式用語以根目錄 `GLOSSARY.md` 為準。兩表若有衝突，以根目錄的定義為準。

## 原則

- **數理統計與迴歸的基礎術語可直接使用**；引用文獻中的其他方法名首次出現時，先從已知概念說明用途，不自造詞
- **中文譯名不統一或難懂者保留英文原文**，例如 simplex、cluster；固定效應等熟悉的譯名可用中文。譯名已存在，也要檢查讀者是否熟悉其意思
- **不以術語帶價值判斷**。模型的成分、組別以編號或參數符號描述，解讀另外寫
- **領域詞不在此限**。監理優先清單、天氣歸因、申報、執飛者描述研究對象，不描述
  方法，照常使用

## 模型與估計

| 用詞 | 本題指什麼 | 不用 |
|---|---|---|
| stratum（層） | 機場 × 年月的分組，固定效應的單位 | 格子 |
| within transformation；within estimator | 層內減去層平均後以 OLS 估計，消去層的固定效應 | 格子內去均值 |
| 線性機率模型 | 逐班天氣欄是否填入正分鐘的二元結果，以最小平方法估計線性條件平均數 | binomial GLM with identity link |
| 公司的主效應 | 公司因子各水準的係數 | 公司效應 |
| reference level（基準水準） | 係數比較的基準公司 DL；單體模型的基準成因 carrier | 對照組 |
| 因子水準間的比較 | 公司之間的係數比較 | 同儕、同儕比較 |
| two-part model：binary part、conditional part | 是否為正；給定為正時的條件分布 | extensive margin、intensive margin |
| simplex；compositional data | 四個成因分鐘數加總等於到達延誤的限制 | 單體、歸因單體 |
| multinomial logistic regression；Poisson log-linear model | 成因分配的模型與其等價的估計形式 | — |
| contrast（對比） | 兩個成因係數之差，如 weather 對 NAS 的對比 | 歸因邊界 |
| 交互作用 | 公司 × 時段、公司 × 共變量的係數 | 漂移、行為型態 |
| mixture component 1、2 | 混合模型的兩個成分，以延誤長度係數的正負描述 | 可疑成分、機械成分 |
| complete separation；quasi-complete separation | 某層內結果全同，MLE 不存在 | — |

## 推論

| 用詞 | 本題指什麼 | 不用 |
|---|---|---|
| cluster-robust variance estimator（sandwich） | 標準誤的估計方式 | 聚類、叢集 |
| two-way cluster-robust 共變異數估計 | 同時按起點機場月份與目的地機場月份容許群內誤差相關 | 雙向、雙向聚類 |
| 多重檢定；FDR（錯誤發現率） | 同時檢定多家公司；FDR 是被標記者中真實虛無假設成立者所占比例的期望，首次使用時說明 | — |
| shrinkage；empirical Bayes | 公司係數向整體平均收縮 | — |
| 名次；confidence set for ranks | 公司在清單上的位置；首次使用 confidence set for ranks 時，先說明估計誤差可能使名次互換，再解釋它給出哪些名次仍有可能 | 名次信賴集合、排序的區間 |
| test of restrictions | 由模型導出一個限制，檢驗資料是否違反 | 內生錨點、錨點 |
| monotonicity restriction | 延誤越長，整段歸因天氣的機率應遞減 | 延誤長度錨點 |
| 共變量 × 公司的交互作用檢定 | 前一航段剩餘緩衝對天氣歸因的影響是否因公司而異 | 外生衝擊、尾機錨點 |
| 低於第 q 分位數者 | 剩餘緩衝最小的航班 | 壓力組 |
| negative control | 預期無效應的情境；本題指兩組氣象皆非惡劣的航班 | 安慰劑、對照樣本 |
| empirical null | 由資料估計的虛無分布，取代理論的虛無分布（theoretical null） | 經驗基準 |
| robustness（穩健性） | 換設定後結論是否改變 | 設定不敏感 |

## 敘述與估計的區分

| 用詞 | 本題指什麼 | 不用 |
|---|---|---|
| 敘述統計量 | Y1、Y2、Y3 三個比例的邊際分布 | 描述層 |
| 模型參數的估計 | two-part model 與 multinomial 模型的係數 | 推論層 |
