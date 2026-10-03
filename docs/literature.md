# 方法文獻

每條寫出處與它支撐的決定。依方法組織，不依題目。作者與期刊取自 arXiv 或 Crossref
的 metadata，已逐條查證；不在此表者不得引用。報告的書目 `report/refs.yml` 由
`scripts/fetch/refs.py` 依同一批識別碼產生，新增文獻時兩處一起改。

決定本身與其理由見 `decisions.md`，此處只記「引誰」。

## 固定效應與估計

| 文獻 | arXiv／DOI | 支撐什麼 |
|---|---|---|
| Chen, Martin, Wooldridge. Another Look at the Linear Probability Model and Nonlinear Index Models | 2308.15338 | extensive 層用 LPM 而非 logit：LPM 何時能近似非線性模型的平均偏效果 |
| Correia, Guimarães, Zylkin. ppmlhdfe: Fast Poisson Estimation with High-Dimensional Fixed Effects. *Stata Journal* | 1903.01690 | 單體以 Poisson 加兩組高維固定效應估計 |
| Correia, Guimarães, Zylkin. Verifying the existence of maximum likelihood estimates for generalized linear models | 1903.01633 | separation 的觀測可安全移除、不帶資訊；單體不走懲罰似然 |
| Lee, Green, Ryan. On the "Poisson Trick" and its Extensions for Fitting Multinomial Regression Models | 1707.08538 | 多項 logit 與 Poisson 對數線性的等價，公司 × 成因交互即多項係數 |
| Weidner, Zylkin. Bias and Consistency in Three-way Gravity Models | 1909.01327 | Poisson 多向固定效應下估計量一致但有 incidental parameter 偏誤，區間可能失準。單體的格子 × 成因與單元兩組固定效應屬同類結構，須正面交代 |
| Mundlak. On the Pooling of Time Series and Cross Section Data. *Econometrica* | 10.2307/1913646 | 層的固定效應以 within transformation 消去 |
| Chamberlain. Analysis of Covariance with Qualitative Data. *Review of Economic Studies* | 10.2307/2297110 | 二元結果下固定效應的條件概似；logit link 不可行的理由 |
| Baker. The Multinomial-Poisson Transformation. *The Statistician* | 10.2307/2348134 | multinomial 與 Poisson 等價的原始出處 |
| Guimarães, Portugal. A Simple Feasible Procedure to Fit Models with High-Dimensional Fixed Effects. *Stata Journal* | 10.1177/1536867X1001000406 | 高維固定效應的交替投影 |
| Firth. Bias Reduction of Maximum Likelihood Estimates. *Biometrika* | 10.1093/biomet/80.1.27 | 機場層級清單的 separation 以懲罰概似處理 |
| Fernández-Val, Weidner. Individual and Time Effects in Nonlinear Panel Models with Large N, T | 1311.7065 | 雙向固定效應的非線性模型（含 Poisson）的解析與 jackknife 偏誤校正；Weidner–Zylkin 指出的 incidental parameter 偏誤的校正方法 |
| Egger, Ji, Wang. PPML and Heavy-Tailed Trade and Factor Flows: Why Standard Inference Fails | 2609.18750 | 結果重尾時 PPML 的常規區間失效，提出 bootstrap 修正。申報分鐘數強右偏，單體區間的有效性須依此檢查 |

## 結果變數的分布

| 文獻 | arXiv／DOI | 支撐什麼 |
|---|---|---|
| Ospina, Ferrari. A general class of zero-or-one inflated beta regression models. *CSDA* | 1103.2372 | Y2 在 1.0 的質量點以 one-inflated 兩部處理；beta 迴歸在端點無定義 |
| Papke, Wooldridge. Econometric Methods for Fractional Response Variables. *Journal of Applied Econometrics* | 10.1002/(SICI)1099-1255(199611)11:6<619::AID-JAE418>3.0.CO;2-1 | 比例型結果只設定條件平均數，不設定分布 |

## 組合資料的結構性零

| 文獻 | arXiv／DOI | 支撐什麼 |
|---|---|---|
| Alzeley, Tsagris. Modelling compositional data with structural zero values | 2608.29954 | 結構性零不可插補，須以條件分布處理。與否決 ALR/ILR 的理由同向，是計數模型之外的對照路線 |
| Koslovsky. A Bayesian Zero-Inflated Dirichlet-Multinomial Regression Model for Multivariate Compositional Count Data | 2302.12283 | 單體的另一種建模：零膨脹 Dirichlet-multinomial，可分離結構性零與抽樣零 |

## 貝氏混合模型與抽樣診斷

| 文獻 | arXiv／DOI | 支撐什麼 |
|---|---|---|
| Hoffman, Gelman. The No-U-Turn Sampler | 1111.4246 | 混合模型的抽樣器 |
| Stephens. Dealing With Label Switching in Mixture Models. *JRSS-B* | 10.1111/1467-9868.00265 | 標籤轉換使收斂診斷失效，須以限制固定成分編號 |
| Betancourt. A Conceptual Introduction to Hamiltonian Monte Carlo | 1701.02434 | divergence 是後驗幾何困難的徵兆，K=3 弱可識別的判讀依據 |
| Vehtari, Gelman, Simpson, Carpenter, Bürkner. Rank-normalization, folding, and localization: An improved R̂. *Bayesian Analysis* | 1903.08008 | 以 `r_hat` 與 ESS 否決 K=3 |
| Vehtari, Gelman, Gabry. Practical Bayesian model evaluation using leave-one-out cross-validation and WAIC. *Statistics and Computing* | 1507.04544 | LOO 以有效抽樣為前提，抽樣無效時不採信 |
| Margossian et al. Nested R̂. *Bayesian Analysis* | 2110.13017 | 多短鏈下的收斂診斷，抽樣後端比較的補充 |
| Grün, Malsiner-Walli. Bayesian Finite Mixture Models | 2407.05470 | 貝氏有限混合的綜述：成分數未知時的處理、標籤轉換、MCMC |

## 標準誤

| 文獻 | arXiv／DOI | 支撐什麼 |
|---|---|---|
| Cameron, Gelbach, Miller. Robust Inference With Multiway Clustering. *JBES* | 10.1198/jbes.2010.07136 | two-way cluster-robust 變異數估計量 |
| Abadie, Athey, Imbens, Wooldridge. When Should You Adjust Standard Errors for Clustering? | 1710.02926 | 聚類層級的選擇依抽樣與指派機制，格子與機尾號雙向聚類的理由 |
| MacKinnon, Nielsen, Webb. Cluster-Robust Inference: A Guide to Empirical Practice. *Journal of Econometrics* | 2205.03285 | 聚類標準誤的實務準則，含叢集數與叢集大小不均時的失準 |
| MacKinnon, Nielsen, Webb. Jackknife inference with two-way clustering | 2406.08880 | 雙向聚類的 jackknife 變異數估計，現行雙向聚類的穩健版本 |
| MacKinnon, Nielsen, Webb. Cluster-robust jackknife and bootstrap inference for logistic regression models | 2406.00650 | 若 extensive 層改用 logit，對應的聚類推論 |

## 排序的不確定性

| 文獻 | arXiv／DOI | 支撐什麼 |
|---|---|---|
| Goldstein, Spiegelhalter. League Tables and Their Limitations. *JRSS-A* | 10.2307/2983325 | 機構排名的不確定性問題的原始提出 |
| Efron, Morris. Stein's Estimation Rule and Its Competitors: An Empirical Bayes Approach. *JASA* | 10.2307/2284155 | 排序前的 empirical Bayes 收縮 |
| Benjamini, Hochberg. Controlling the False Discovery Rate. *JRSS-B* | 10.1111/j.2517-6161.1995.tb02031.x | 多家公司多個係數同時檢定的 FDR 控制 |
| Chetverikov, Mogstad, Morgen, Romano, Shaikh, Wilhelm. csranks: An R Package for Estimation and Inference Involving Ranks | 2401.15205 | 對名次而非係數做推論，邊際與同時的名次信賴集合 |
| Al Mohamad, van Zwet, Goeman, Solari. Simultaneous confidence sets for ranks using the partitioning principle | 1708.02729 | 整張清單同時成立的名次集合 |
| Gu, Koenker. Invidious Comparisons: Ranking and Selection as Compound Decisions | 2012.12550 | 以經驗貝氏把排序與選擇寫成複合決策，控制誤選率。收縮與名次推論的統一框架 |
| Kline, Rose, Walters. A Discrimination Report Card | 2306.13005 | 以經驗貝氏給雇主評級，權衡評級的資訊量與排序錯誤。與監理優先清單同構：對一批機構的行為偏離給出可公開的分級 |
| Bazylik, Mogstad, Romano, Shaikh, Wilhelm. Finite- and Large-Sample Inference for Ranks using Multinomial Data | 2402.00192 | 多項資料下的名次信賴集合，對應單體的成因分配 |
| Chandra, Xie. Finite-Sample Valid Rank Confidence Sets for a Broad Class of Statistical and Machine Learning Models | 2512.00316 | 有限樣本有效的名次集合，適用範圍較 csranks 廣 |

## 未觀測混淆的敏感度

| 文獻 | arXiv／DOI | 支撐什麼 |
|---|---|---|
| Cinelli, Hazlett. Making Sense of Sensitivity: Extending Omitted Variable Bias. *JRSS-B* | 10.1111/rssb.12348 | partial R² 敏感度分析的基本框架，Strezhnev 與 Wainstein 的修正皆以此為底 |
| Strezhnev. Omitted variable bias sensitivity analysis with clustered treatment assignment | 2607.13334 | 處理變數在聚合層變動、分析在個體層時，partial R² 敏感度分析須修正。公司效應在格子內識別，屬此情形 |
| Wainstein, Hazlett. Sensitivity of weighted least squares estimators to omitted variables | 2508.02954 | 加權迴歸的遺漏變數敏感度 |
| Bach, Chernozhukov, Cinelli et al. Sensitivity Analysis for Causal ML: A Use Case at Booking.com | 2510.09109 | 敏感度分析在實務資料上的完整流程示範 |

## Negative control

| 文獻 | arXiv／DOI | 支撐什麼 |
|---|---|---|
| Lipsitch, Tchetgen Tchetgen, Cohen. Negative Controls: A Tool for Detecting Confounding and Bias in Observational Studies. *Epidemiology* | 10.1097/EDE.0b013e3181d61eeb | 兩端機場天氣良好的航班作為 negative control：預期無效應處仍有關聯，即顯示未測量的因素 |
| Shi, Miao, Tchetgen Tchetgen. A Selective Review of Negative Control Methods in Epidemiology | 2009.05641 | negative control 的假設、設計準則與偏誤的偵測、縮減、校正方法綜述 |
| Schuemie, Ryan, DuMouchel, Suchard, Madigan. Interpreting Observational Studies: Why Empirical Calibration Is Needed to Correct p-Values. *Statistics in Medicine* | 10.1002/sim.5925 | 以 negative control 上的估計值分布作 empirical null，校正主分析的 p 值 |
| Hwang, Quiroz, Gallego. Assessing the effectiveness of empirical calibration under different bias scenarios | 2111.04233 | empirical calibration 在不同偏誤情境下的表現；negative control 與主分析的偏誤來源不同時校正會失準 |

## 機構比較（profiling）

醫療機構評比與本題同構：以風險調整後的結果比較一批機構，找出偏離者。

| 文獻 | arXiv／DOI | 支撐什麼 |
|---|---|---|
| Normand, Shahian. Statistical and Clinical Aspects of Hospital Outcomes Profiling. *Statistical Science* | 0710.4622 | 機構 profiling 的統計問題總覽：風險調整、樣本量差異、階層模型 |
| Kalbfleisch, Wolfe. On Monitoring Outcomes of Medical Providers. *Statistics in Biosciences* | 10.1007/s12561-013-9093-x | 機構效應取固定效應或隨機效應的比較；公司直接作為固定效應因子的依據 |
| Xia, He, Li, Kalbfleisch. Accounting for total variation and robustness in profiling health care providers | 1907.07809 | 樣本量大時任何偏離皆顯著，大機構被不成比例地標記；以 empirical null 把風險調整不完全造成的機構間變異納入參照分布 |
| Hartman, Messana, Kang et al. Composite Scores for Transplant Center Evaluation: A New Individualized Empirical Null Method | 2207.07602 | 依機構樣本量個別化的 empirical null |
| Efron. Size, power and false discovery rates. *Annals of Statistics* | 0710.2245 | empirical null 與 local false discovery rate 的理論 |
| Spiegelhalter. Funnel Plots for Comparing Institutional Performance. *Statistics in Medicine* | 10.1002/sim.1970 | 以精確度為橫軸的管制界限取代排名表；overdispersion 下界限的調整 |

## 形狀限制的檢定

| 文獻 | arXiv／DOI | 支撐什麼 |
|---|---|---|
| Chetverikov. Testing Regression Monotonicity in Econometric Models | 1212.6757 | 迴歸函數單調性的無母數檢定；記入天氣的機率對天氣嚴重度、對延誤長度的 monotonicity restriction |
| Espinosa-Brito, Hennig. Inference for the proportional odds cumulative logit model with monotonicity constraints for ordinal predictors and ordinal response | 2107.04946 | 有序解釋變數加單調限制時的檢定與信賴區域；飛行規則類別為四級有序變數 |

## 交叉分類隨機效應

| 文獻 | arXiv／DOI | 支撐什麼 |
|---|---|---|
| Ghosh, Hastie, Owen. Backfitting for large scale crossed random effects regressions | 2007.10612 | 公司 × 機場非巢狀隨機效應在大樣本下的線性時間估計 |
| Papaspiliopoulos, Stumpf-Fétizon, Zanella. Scalable Bayesian computation for crossed and nested hierarchical models | 2103.10875 | 同一結構的貝氏抽樣，計算量隨觀測數線性成長 |

## 領域

| 文獻 | arXiv／DOI | 支撐什麼 |
|---|---|---|
| Forbes, Lederman, Tombe. Quality Disclosure Programs and Internal Organizational Practices: Evidence from Airline Flight Delays. *AEJ: Microeconomics* | 10.1257/mic.20130164 | 準點率排名以到達延誤 15 分鐘為界，航空公司對門檻附近航班的反應因公司而異。申報行為受制度誘因影響、且公司間異質的先例 |

## 模擬研究

| 文獻 | arXiv／DOI | 支撐什麼 |
|---|---|---|
| Morris, White, Crowther. Using simulation studies to evaluate statistical methods. *Statistics in Medicine* | 1712.03198 | ADEMP 框架，模擬研究的設計與報告方式 |
| Schreck, Slynko, Saadati, Benner. Statistical Plasmode Simulations: Potentials, Challenges and Recommendations | 2305.06028 | 保留真實共變量、只模擬結果的 plasmode 設計 |
| Shaw et al. A cautionary note for plasmode simulation studies in the setting of causal inference | 2504.11740 | 生成模型與分析模型相同時表現被高估，生成模型須更有彈性 |
| Karim. A Plasmode Benchmark Can Refute but Cannot Certify | 2609.26186 | 模擬只能否證方法，不能證實，結論的界限 |
