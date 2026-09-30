# 方法文獻

每條寫出處與它支撐的決定。依方法組織，不依題目。作者與期刊取自 arXiv metadata，
已逐條查證；未查證者列於文末，不得引用。

決定本身與其理由見 `decisions.md`，此處只記「引誰」。

## 固定效應與估計

| 文獻 | arXiv | 支撐什麼 |
|---|---|---|
| Chen, Martin, Wooldridge. Another Look at the Linear Probability Model and Nonlinear Index Models | 2308.15338 | extensive 層用 LPM 而非 logit：LPM 何時能近似非線性模型的平均偏效果 |
| Correia, Guimarães, Zylkin. ppmlhdfe: Fast Poisson Estimation with High-Dimensional Fixed Effects. *Stata Journal* | 1903.01690 | 單體以 Poisson 加兩組高維固定效應估計 |
| Correia, Guimarães, Zylkin. Verifying the existence of maximum likelihood estimates for generalized linear models | 1903.01633 | separation 的觀測可安全移除、不帶資訊；單體不走懲罰似然 |
| Lee, Green, Ryan. On the "Poisson Trick" and its Extensions for Fitting Multinomial Regression Models | 1707.08538 | 多項 logit 與 Poisson 對數線性的等價，公司 × 成因交互即多項係數 |
| Weidner, Zylkin. Bias and Consistency in Three-way Gravity Models | 1909.01327 | Poisson 多向固定效應下估計量一致但有 incidental parameter 偏誤，區間可能失準。單體的格子 × 成因與單元兩組固定效應屬同類結構，須正面交代 |
| Egger, Ji, Wang. PPML and Heavy-Tailed Trade and Factor Flows: Why Standard Inference Fails | 2609.18750 | 結果重尾時 PPML 的常規區間失效，提出 bootstrap 修正。申報分鐘數強右偏，單體區間的有效性須依此檢查 |

## 結果變數的分布

| 文獻 | arXiv | 支撐什麼 |
|---|---|---|
| Ospina, Ferrari. A general class of zero-or-one inflated beta regression models. *CSDA* | 1103.2372 | Y2 在 1.0 的質量點以 one-inflated 兩部處理；beta 迴歸在端點無定義 |

## 組合資料的結構性零

| 文獻 | arXiv | 支撐什麼 |
|---|---|---|
| Alzeley, Tsagris. Modelling compositional data with structural zero values | 2608.29954 | 結構性零不可插補，須以條件分布處理。與否決 ALR/ILR 的理由同向，是計數模型之外的對照路線 |
| Koslovsky. A Bayesian Zero-Inflated Dirichlet-Multinomial Regression Model for Multivariate Compositional Count Data | 2302.12283 | 單體的另一種建模：零膨脹 Dirichlet-multinomial，可分離結構性零與抽樣零 |

## 貝氏混合模型與抽樣診斷

| 文獻 | arXiv | 支撐什麼 |
|---|---|---|
| Hoffman, Gelman. The No-U-Turn Sampler | 1111.4246 | 混合模型的抽樣器 |
| Betancourt. A Conceptual Introduction to Hamiltonian Monte Carlo | 1701.02434 | divergence 是後驗幾何困難的徵兆，K=3 弱可識別的判讀依據 |
| Vehtari, Gelman, Simpson, Carpenter, Bürkner. Rank-normalization, folding, and localization: An improved R̂. *Bayesian Analysis* | 1903.08008 | 以 `r_hat` 與 ESS 否決 K=3 |
| Vehtari, Gelman, Gabry. Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC. *Statistics and Computing* | 1507.04544 | LOO 以有效抽樣為前提，抽樣無效時不採信 |
| Margossian et al. Nested R̂. *Bayesian Analysis* | 2110.13017 | 多短鏈下的收斂診斷，抽樣後端比較的補充 |
| Grün, Malsiner-Walli. Bayesian Finite Mixture Models | 2407.05470 | 貝氏有限混合的綜述：成分數未知時的處理、標籤轉換、MCMC |

## 標準誤

| 文獻 | arXiv | 支撐什麼 |
|---|---|---|
| Abadie, Athey, Imbens, Wooldridge. When Should You Adjust Standard Errors for Clustering? | 1710.02926 | 聚類層級的選擇依抽樣與指派機制，格子與機尾號雙向聚類的理由 |
| MacKinnon, Nielsen, Webb. Cluster-Robust Inference: A Guide to Empirical Practice. *Journal of Econometrics* | 2205.03285 | 聚類標準誤的實務準則，含叢集數與叢集大小不均時的失準 |
| MacKinnon, Nielsen, Webb. Jackknife inference with two-way clustering | 2406.08880 | 雙向聚類的 jackknife 變異數估計，現行雙向聚類的穩健版本 |
| MacKinnon, Nielsen, Webb. Cluster-robust jackknife and bootstrap inference for logistic regression models | 2406.00650 | 若 extensive 層改用 logit，對應的聚類推論 |

## 排序的不確定性

| 文獻 | arXiv | 支撐什麼 |
|---|---|---|
| Chetverikov, Mogstad, Morgen, Romano, Shaikh, Wilhelm. csranks: An R Package for Estimation and Inference Involving Ranks | 2401.15205 | 對名次而非係數做推論，邊際與同時的名次信賴集合 |
| Al Mohamad, van Zwet, Goeman, Solari. Simultaneous confidence sets for ranks using the partitioning principle | 1708.02729 | 整張清單同時成立的名次集合 |
| Gu, Koenker. Invidious Comparisons: Ranking and Selection as Compound Decisions | 2012.12550 | 以經驗貝氏把排序與選擇寫成複合決策，控制誤選率。收縮與名次推論的統一框架 |
| Kline, Rose, Walters. A Discrimination Report Card | 2306.13005 | 以經驗貝氏給雇主評級，權衡評級的資訊量與排序錯誤。與監理優先清單同構：對一批機構的行為偏離給出可公開的分級 |
| Bazylik, Mogstad, Romano, Shaikh, Wilhelm. Finite- and Large-Sample Inference for Ranks using Multinomial Data | 2402.00192 | 多項資料下的名次信賴集合，對應單體的成因分配 |
| Chandra, Xie. Finite-Sample Valid Rank Confidence Sets for a Broad Class of Statistical and Machine Learning Models | 2512.00316 | 有限樣本有效的名次集合，適用範圍較 csranks 廣 |

## 未觀測混淆的敏感度

| 文獻 | arXiv | 支撐什麼 |
|---|---|---|
| Strezhnev. Omitted variable bias sensitivity analysis with clustered treatment assignment | 2607.13334 | 處理變數在聚合層變動、分析在個體層時，partial R² 敏感度分析須修正。公司效應在格子內識別，屬此情形 |
| Wainstein, Hazlett. Sensitivity of weighted least squares estimators to omitted variables | 2508.02954 | 加權迴歸的遺漏變數敏感度 |
| Bach, Chernozhukov, Cinelli et al. Sensitivity Analysis for Causal ML: A Use Case at Booking.com | 2510.09109 | 敏感度分析在實務資料上的完整流程示範 |

## 模擬研究

| 文獻 | arXiv | 支撐什麼 |
|---|---|---|
| Morris, White, Crowther. Using simulation studies to evaluate statistical methods. *Statistics in Medicine* | 1712.03198 | ADEMP 框架，模擬研究的設計與報告方式 |
| Schreck, Slynko, Saadati, Benner. Statistical Plasmode Simulations: Potentials, Challenges and Recommendations | 2305.06028 | 保留真實共變量、只模擬結果的 plasmode 設計 |
| Shaw et al. A cautionary note for plasmode simulation studies in the setting of causal inference | 2504.11740 | 生成模型與分析模型相同時表現被高估，生成模型須更有彈性 |
| Karim. A Plasmode Benchmark Can Refute but Cannot Certify | 2609.26186 | 模擬只能否證方法，不能證實，結論的界限 |

## 待查證

arXiv 上沒有，須以 DOI 查證後才可引用：

- 多項與 Poisson 等價的原始出處：Baker, The Statistician
- Firth 懲罰似然：Firth, Biometrika
- 經驗貝氏收縮：Efron & Morris, JASA
- FDR：Benjamini & Hochberg, JRSS-B
- 標籤轉換：Stephens, JRSS-B
- 固定效應：Mundlak, Econometrica；Chamberlain, REStud
- 分數型結果：Papke & Wooldridge, Journal of Applied Econometrics
- 多向聚類：Cameron, Gelbach & Miller, JBES
- 高維固定效應的交替投影：Guimarães & Portugal, Stata Journal
- 機構排名的統計問題：Goldstein & Spiegelhalter, JRSS-A

Cinelli & Hazlett 的遺漏變數敏感度分析（JRSS-B）在 arXiv 以標題查無，同樣須以 DOI 查證。
