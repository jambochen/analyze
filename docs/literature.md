# 方法文獻

每條寫出處與它支撐的決定。依方法組織，不依題目。作者與期刊取自 arXiv 或 Crossref
的 metadata，已逐條查證；不在此表者不得引用。報告的書目 `report/refs.yml` 由
`scripts/fetch/refs.py` 依同一批識別碼產生，新增文獻時兩處一起改。

決定本身與其理由見 `decisions.md`，此處記引用依據與閱讀範圍。

## 全文與證據範圍

`ref/` 的 PDF 與 TeX 原始檔為本機副本，不納入版本控制。下表只列已能確認對應關係的
副本；閱讀範圍是本研究目前引用的論點，不代表已逐頁審閱全篇。
引用一篇文獻的方法或限制時，以其全文相關章節為準；只核對摘要或書目資料時，
不得寫成已核對全文。其他文獻可依表中的 DOI／arXiv 識別碼取得。

| 文獻 | 本機全文 | 本研究引用的範圍與限制 |
|---|---|---|
| Silverman, Skinner | `ref/w8133.pdf`（NBER 工作論文版） | 診斷分類邊界、所有權比較與病況組成檢查；PDF 文字擷取失真，精確頁碼待頁面核對，期刊版全文未取得 |
| Brakenhoff et al. | `ref/12874_2018_Article_510.pdf` | *Methods* 的 Simulation study、Data generation、Scenarios 與結果討論：機構異常者設定、分類指標及生成模型限制 |
| Steinbuss, Böhm | `ref/2004.06947v1.pdf` | §3.2、§4.1–4.3、§8.2：局部／全域異常設計、正常樣本標籤假設與生成法限制 |
| Crawford, Petela | `ref/2206.08093-source/ERClaimsAnomalyDetection.tex`（arXiv TeX 原始檔） | Upcoding Detection、Cost Avoidance Optimization：診斷分群後的單筆嚴重程度異常分數；查核排序模型另需已審核案件，不能直接用於本題 |
| Shekhar, Leder-Luis, Akoglu | `ref/2211.02927-source/paper.tex` 與各節 TeX（arXiv 原始檔） | Method、Peer-based detector、Aggregate ranking 與 Results：相似醫院比較、多訊號整合；公開案件僅提供部分驗證，異常排序不是違規判定 |


## 固定效應與估計

| 文獻 | arXiv／DOI | 支撐什麼 |
|---|---|---|
| Chen, Martin, Wooldridge. Another Look at the Linear Probability Model and Nonlinear Index Models | 2308.15338 | extensive 層用 LPM 而非 logit：LPM 何時能近似非線性模型的平均偏效果 |
| Correia, Guimarães, Zylkin. ppmlhdfe: Fast Poisson Estimation with High-Dimensional Fixed Effects. *Stata Journal* | 1903.01690 | 單體以 Poisson 加兩組高維固定效應估計 |
| Correia, Guimarães, Zylkin. Verifying the existence of maximum likelihood estimates for generalized linear models | 1903.01633 | separation 的觀測可安全移除、不帶資訊；單體不走懲罰似然 |
| Lee, Green, Ryan. On the "Poisson Trick" and its Extensions for Fitting Multinomial Regression Models | 1707.08538 | 多項 logit 與 Poisson 對數線性的等價，公司 × 成因交互即多項係數 |
| Weidner, Zylkin. Bias and Consistency in Three-way Gravity Models | 1909.01327 | Poisson 只含一組固定效應時無 incidental parameter 問題；多組時視各組個數相對於觀測數的成長速度而定。附錄的例子（兩組固定效應 $\alpha_{it}$、$\gamma_{ij}$，$J$ 與 $T$ 固定）依啟發式論證與未列出的模擬指出估計量不一致，未給證明；與單體的層 × 成因、單元兩組固定效應同一結構：一層內公司數有限、成因只有四個。偏誤不隨層數增加而消失，須以模擬量測。另指出 cluster 共變異矩陣的估計向下偏 |
| Mundlak. On the Pooling of Time Series and Cross Section Data. *Econometrica* | 10.2307/1913646 | 層的固定效應以 within transformation 消去 |
| Eccleston, Hedayat. On the Theory of Connected Designs: Characterization and Optimality. *Annals of Statistics* | 10.1214/aos/1176342876 | 區集設計的連通性：線性加法模型下，設計連通是所有處理對比可不偏估計的條件。本題把公司看成處理、機場 × 年月看成區集，從未同場的兩家公司經共同的第三家間接比較。只讀過摘要，定理的條件未逐一核對 |
| Weeks, Williams. A Note on the Determination of Connectedness in an N-Way Cross Classification. *Technometrics* | 10.1080/00401706.1964.10490188 | 分類超過兩個方向時連通的判定。本題同時含起點與目的地兩組機場月份效果，屬此情形。僅核對書目，全文未取得 |
| Abowd, Creecy, Kramarz. Computing Person and Firm Effects Using Linked Longitudinal Employer-Employee Data. U.S. Census Bureau Technical Paper TP-2002-06 | https://www2.census.gov/ces/tp/tp-2002-06.pdf | 人員與雇主兩組固定效應的線性模型中，連通群組內的效應才可比較。與上兩篇是同一性質；三篇皆為線性模型的結果，PPML 下連通只是必要條件 |
| Chamberlain. Analysis of Covariance with Qualitative Data. *Review of Economic Studies* | 10.2307/2297110 | 二元結果下固定效應的條件概似；logit link 不可行的理由 |
| Baker. The Multinomial-Poisson Transformation. *The Statistician* | 10.2307/2348134 | multinomial 與 Poisson 等價的原始出處 |
| Guimarães, Portugal. A Simple Feasible Procedure to Fit Models with High-Dimensional Fixed Effects. *Stata Journal* | 10.1177/1536867X1001000406 | 高維固定效應的交替投影 |
| Firth. Bias Reduction of Maximum Likelihood Estimates. *Biometrika* | 10.1093/biomet/80.1.27 | 機場層級清單的 separation 以懲罰概似處理 |
| Fernández-Val, Weidner. Individual and Time Effects in Nonlinear Panel Models with Large N, T | 1311.7065 | 雙向固定效應的非線性模型（含 Poisson）的解析與 jackknife 偏誤校正，漸近條件是兩個維度皆大；Poisson 在解釋變數 strictly exogenous 時偏誤項為零。單體的兩個維度（一層內的公司數、成因數）皆不大，不符合其條件；即使套用，Poisson 的校正量也是零，不提供修正 |
| Egger, Ji, Wang. PPML and Heavy-Tailed Trade and Factor Flows: Why Standard Inference Fails | 2609.18750 | 結果重尾時 PPML 的常規區間失效，提出 bootstrap 修正。申報分鐘數強右偏，單體區間的有效性須依此檢查 |

## Poisson pseudo-likelihood 與 overdispersion

| 文獻 | arXiv／DOI | 支撐什麼 |
|---|---|---|
| Wedderburn. Quasi-likelihood functions, generalized linear models, and the Gauss—Newton method. *Biometrika* | 10.1093/biomet/61.3.439 | quasi-likelihood 的定義：估計只需平均數與變異數的關係，不需完整分布。分鐘數不是計數仍可用 Poisson 的估計方程式 |
| McCullagh. Quasi-Likelihood Functions. *Annals of Statistics* | 10.1214/aos/1176346056 | quasi-likelihood 估計量在二階動差假設下的一致性與漸近常態。假設變異數函數正確，不涵蓋變異數設錯的情形 |
| Gourieroux, Monfort, Trognon. Pseudo Maximum Likelihood Methods: Theory. *Econometrica* | 10.2307/1913471 | pseudo-likelihood 取自 linear exponential family 時，條件平均數設定正確即得一致的估計量，變異數不必正確。overdispersion 不影響單體係數的一致性，只影響標準誤 |
| Santos Silva, Tenreyro. The Log of Gravity. *Review of Economics and Statistics* | 10.1162/rest.88.4.641 | Poisson pseudo-likelihood 用於非整數的非負結果，零是合法的觀測值；推論須用 robust 共變異矩陣 |
| Allison, Waterman. Fixed-Effects Negative Binomial Regression Models. *Sociological Methodology* | 10.1111/1467-9531.00117 | 不改用 negative binomial 的理由之一：條件概似版的固定效應 negative binomial 不是真正的固定效應估計量。該文的模擬另顯示，直接放入群組指示變數的版本係數偏誤小、但 model-based 標準誤偏低；模擬限於小規模且分布設定正確的情形 |
| Bosquet, Boulhol. Applying the GLM Variance Assumption to Overcome the Scale-Dependence of the Negative Binomial QGPML Estimator. *Econometric Reviews* | 10.1080/07474938.2013.806102 | 不改用 negative binomial 的理由之二：結果是單位任意的連續變數時，negative binomial 的 pseudo-likelihood 估計值隨單位改變。分鐘數正是此類 |
| Blackburn. The Relative Performance of Poisson and Negative Binomial Regression Estimators. *Oxford Bulletin of Economics and Statistics* | 10.1111/obes.12074 | 不改用 negative binomial 的理由之三：模擬顯示結果 overdispersed 時，除非 overdispersion 的形式已知，negative binomial 並不優於 Poisson |
| Santos Silva, Tenreyro. The Log of Gravity at 15. *Portuguese Economic Journal* | 10.1007/s10258-021-00203-w | 結果沒有天然單位時 overdispersion 沒有定義：平均數與變異數的關係隨單位而變，凡容許 overdispersion 的模型估計值皆隨單位改變，不限於 negative binomial。另整理 Poisson 在一組與兩組固定效應下的 incidental parameter 結果，以及兩組固定效應下 cluster 共變異矩陣失效 |

## 結果變數的分布

| 文獻 | arXiv／DOI | 支撐什麼 |
|---|---|---|
| Ospina, Ferrari. A general class of zero-or-one inflated beta regression models. *CSDA* | 1103.2372 | Y2 在 1.0 的質量點以 one-inflated 兩部處理；beta 迴歸在端點無定義 |
| Papke, Wooldridge. Econometric Methods for Fractional Response Variables. *Journal of Applied Econometrics* | 10.1002/(SICI)1099-1255(199611)11:6<619::AID-JAE418>3.0.CO;2-1 | 比例型結果只設定條件平均數，不設定分布 |
| Ramalho, Ramalho, Murteira. Alternative Estimating and Testing Empirical Strategies for Fractional Regression Models. *Journal of Economic Surveys* | 10.1111/j.1467-6419.2009.00602.x | 比例型結果在邊界有機率質量時的模型整理；two-part 做法先以二元模型處理是否為正，再以 fractional regression 處理正值 |

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
| Hwang, Quiroz, Gallego. Assessing the effectiveness of empirical calibration under different bias scenarios | 2111.04233 | 以模擬檢視 empirical calibration 在不同偏誤情境下的表現：多數情境下區間的涵蓋率提高，對估計偏誤的修正則不一致；negative control 合適與否對校正的幅度影響大 |
| Sofer, Richardson, Colicino, Schwartz, Tchetgen Tchetgen. On Negative Outcome Control of Unobserved Confounding as a Generalization of Difference-in-Differences. *Statistical Science* | 10.1214/16-sts558 | 把 difference-in-differences 解讀為 negative control outcome 的做法，並給出不要求兩個結果同尺度的推廣；公司 × 天氣狀態的交互作用即以天氣良好的航班扣除與天氣無關的公司間差異 |
| Tchetgen Tchetgen, Park, Richardson. Universal Difference-in-Differences for Causal Inference in Epidemiology | 2302.00840 | 結果為二元、計數或多類別時 parallel trends 不可信，改以 odds ratio 尺度上的 equi-confounding 假設識別；本題的結果為二元與成因分配，相減的尺度須交代 |

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

## 逐期監測

| 文獻 | arXiv／DOI | 支撐什麼 |
|---|---|---|
| Steiner, Cook, Farewell, Treasure. Monitoring surgical performance using risk-adjusted cumulative sum charts. *Biostatistics* | 10.1093/biostatistics/1.4.441 | risk-adjusted CUSUM：以概似比的得分逐筆累加，個案組成改變不致誤發訊號；清單由一次性排序改為逐期監測時的工具 |
| Spiegelhalter, Sherlaw-Johnson, Bardsley, Blunt, Wood, Grigg. Statistical Methods for Healthcare Regulation: Rating, Screening and Surveillance. *JRSS-A* | 10.1111/j.1467-985x.2011.01010.x | 監理機關的三項工作（評等、決定查核對象、持續監測）各自的統計工具，以可含 overdispersion 的 Z 分數為共同基礎，並處理大量指標同時監測的 multiple comparisons；與本題的交付物同構 |

## 選樣與截斷

樣本限於到達延誤達申報門檻的航班，而公司本身影響航班是否進入樣本。

| 文獻 | arXiv／DOI | 支撐什麼 |
|---|---|---|
| Lee. Training, Wages, and Sample Selection: Estimating Sharp Bounds on Treatment Effects. *Review of Economic Studies* | 10.1111/j.1467-937x.2009.00536.x | 結果只在被選入者身上觀測、而處理影響選入時，以修剪樣本得到效應的 sharp bounds，不需 exclusion restriction |
| Semenova. Generalized Lee Bounds | 2008.12720 | Lee bounds 要求處理對選入的效應方向在所有個體一致；此文放寬為給定共變量後方向一致，並以共變量收窄 bounds |
| Frangakis, Rubin. Principal Stratification in Causal Inference. *Biometrics* | 10.1111/j.0006-341x.2002.00021.x | 以處理後變數在各處理下的潛在值交叉分類個體；「無論由哪家公司執飛都會達門檻的航班」即一個 principal stratum，估計對象須定義在其內 |
| Knox, Lowe, Mummolo. Administrative Records Mask Racially Biased Policing. *American Political Science Review* | 10.1017/s0003055420000039 | 行政紀錄只含被選入者、而選入本身受所研究的因素影響時，慣用的估計有偏誤，部分量不可識別；以 principal stratification 導出偏誤的形式與 sharp bounds。申報紀錄只含達門檻的航班，結構相同 |

## 量測誤差

`wx_` 與 `oracle_` 是同一天氣在兩個時點的量測。

| 文獻 | arXiv／DOI | 支撐什麼 |
|---|---|---|
| Schennach. Recent Advances in the Measurement Error Literature. *Annual Review of Economics* | 10.1146/annurev-economics-080315-015058 | 非線性模型中量測誤差的估計與推論方法綜述，含以重複量測、工具變數等輔助資訊識別的做法，以及誤差不獨立、均值不為零時的結果 |
| Rosner, Spiegelman, Willett. Correction of Logistic Regression Relative Risk Estimates and Confidence Intervals for Random Within-Person Measurement Error. *American Journal of Epidemiology* | 10.1093/oxfordjournals.aje.a116453 | 沒有 gold standard 時以重複量測估計誤差並校正 logistic 迴歸的係數與區間，容許多個共變量的誤差相關 |

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

## 隱藏行為的偵測

申報者不願揭露的行為如何由資料偵測。本題與這批研究同構：以誠實申報下應有的
型態為基準，找出偏離者。

| 文獻 | arXiv／DOI | 支撐什麼 |
|---|---|---|
| Zitzewitz. Forensic Economics. *Journal of Economic Literature* | 10.1257/jel.50.3.731 | 偵測隱藏行為的研究依識別來源分為五類：直接觀測、兩種量測的比對、與誘因的相關、偏離誠實行為的模型、市場效率。本題屬第四類下的統計模型：不預設隱藏行為的型態，找在誠實行為下應屬罕見的偏離。另指出兩種量測之間的水準差可有無辜的解釋，較強的證據是差距隨誘因變動；第二種量測若可被申報者操弄則失效，氣象觀測不受航空公司控制 |
| Bolton, Hand. Statistical Fraud Detection: A Review. *Statistical Science* | 10.1214/ss/1042727940 | 統計學的 fraud detection 工具與應用領域的綜述。不預設偏誤方向、以偏離同儕為準的立場，在統計文獻中的出處 |
| Zinman, Zitzewitz. Wintertime for Deceptive Advertising? *American Economic Journal: Applied Economics* | 10.1257/app.20130346 | 滑雪場自報的降雪量高於政府氣象資料；差距在週末較大，而實際降雪不隨星期變動。與本題同以觀測氣象為基準稽核自報值；證據來自差距隨誘因變動，而非差距的水準 |
| Parsons, Sulaeman, Yates, Hamermesh. Strike Three: Discrimination, Incentives, and Evaluation. *American Economic Review* | 10.1257/aer.101.4.1410 | 裁判的好壞球判定隨投手的族裔而異，且主要出現在裁判受監督較少之處。判定者的裁量與可客觀量測的事實並存：天氣歸因是判定，氣象觀測是事實 |
| Pierce, Snyder. Ethical Spillovers in Firms: Evidence from Vehicle Emissions Testing. *Management Science* | 10.1287/mnsc.1080.0927 | 排放檢驗的通過率有組織層級的效應：同一檢驗員在不同機構工作時，通過率向所在機構靠攏。裁量性判定中以機構為固定效應的先例；其識別假設是未觀測的受檢對象特徵不隨機構系統性變動，與公司主效應的假設同形 |
| Crawford, Petela. Applications of Machine Learning to the Identification of Anomalous ER Claims | 2206.08093 | 依診斷群內的嚴重程度代碼分布，找異常偏高的單筆理賠；群組避免罕見診斷的分數不穩定。本題也先界定可比條件，但要比較多欄分鐘配置，且兩個方向的偏離都納入清單。該文合併診斷代碼是因為單筆分數的分母太小，且以結果變數的平均為分群依據；本題的公司係數由全部格子合併估出，格子不依申報結果合併，理由見 `decisions.md`。文中查核排序的 random forest 使用已審核案件，本題沒有相同標籤 |
| Shekhar, Leder-Luis, Akoglu. Unsupervised Machine Learning for Explainable Health Care Fraud Detection | 2211.02927 | 以服務類型或病人特徵尋找相似醫院，比較 DRG 代碼分布及支出，並與其他異常偵測器整合排序；公開的反詐欺案件僅提供部分正例。本題借用同儕比較與查核優先排序，不把偏離解讀為已證實的不實申報 |
| Jacob, Levitt. Rotten Apples: An Investigation of the Prevalence and Predictors of Teacher Cheating. *Quarterly Journal of Economics* | 10.1162/00335530360698441 | 合併兩個指標（成績的異常波動、班內答案的可疑型態）偵測教師作弊，並由誘因的變動佐證。以多個指標的一致偏離排出查核對象 |
| Geruso, Layton. Upcoding: Evidence from Medicare on Squishy Risk Adjustment. *Journal of Political Economy* | 10.1086/704756 | 同樣的投保人在診斷影響給付的制度下被記入較高的風險分數；編碼強度隨保險人與醫療提供者的整合程度升高。控制底層狀態後比較申報者的分類強度，與本題同一結構 |
| Sanghavi, Pan, Caudry. Assessment of nursing home reporting of major injury falls for quality measurement on Nursing Home Compare. *Health Services Research* | 10.1111/1475-6773.13247 | 護理之家自報的重大傷害跌倒，逐人連結至住院理賠紀錄後比對；以含機構隨機效應的線性 multilevel model 找出與漏報相關的個人與機構特徵，並指出此類模型可用於鎖定查核對象。有逐筆的獨立紀錄，故能估漏報率的絕對水準；本題的氣象觀測是共變量，不是成因的真值，只能比較公司間的相對偏離 |
| Dawid, Skene. Maximum Likelihood Estimation of Observer Error-Rates Using the EM Algorithm. *Applied Statistics* | 10.2307/2346806 | 真實類別未知時各評定者錯誤率的最大概似估計。識別來自同一對象有多筆評定，並假設給定真實類別後各筆評定獨立。本題每班航班只由一家公司申報一次，概似只含各公司申報類別的邊際分布，真實成因的分布與錯誤率無法分離。故不估錯誤分類率，改假設同一層內各公司面對的真實成因分布相同，比較公司間的相對偏離 |
| Aelvoet et al. Miscoding: A threat to the hospital care system. How to detect it? *Revue d'Épidémiologie et de Santé Publique* | 10.1016/j.respe.2009.02.206 | 以醫院與時間趨勢的交互項找相對全國的編碼偏離，再用定向病歷查核追查。與本題「相對偏離→監理優先清單」接近；但其隨機與定向查核在不同醫院進行，且核查項目不同，作者明言不能據此比較兩種查核的效率。趨勢法也會漏掉早已達穩定偏高水準的醫院。本題既無獨立查核，清單更不能當不實申報判決 |

這批研究能把偏離稱為不實申報或估出錯誤率，靠的是誘因的變動、事後的外部
複查、逐筆的獨立紀錄，或同一對象的重複評定。本題皆無，結論止於偏離同儕。

## 領域

| 文獻 | arXiv／DOI | 支撐什麼 |
|---|---|---|
| Forbes, Lederman, Tombe. Quality Disclosure Programs and Internal Organizational Practices: Evidence from Airline Flight Delays. *AEJ: Microeconomics* | 10.1257/mic.20130164 | 準點率排名以到達延誤 15 分鐘為界，航空公司對門檻附近航班的反應因公司而異。申報行為受制度誘因影響、且公司間異質的先例 |
| Aggarwal, Schenone. Incentives and Competition in the Airline Industry. *Review of Corporate Finance Studies* | 10.1093/rcfs/cfz002 | 高階主管獎金與準點率掛鉤的公司準點率確有改善，但部分公司以拉長排定飛行時間達成。績效指標被策略性操作的直接證據 |
| Bilotkach, Lakew. Causes and Effects of Air Traffic Delays: Evidence from Aggregated Data. *Journal of Transport Economics and Policy* | 10.3828/jtep.2022.56.2.103 | 以延誤成因資料研究機場集中度與延誤的關係，把成因分為可歸於航空公司與否。成因欄位被下游研究當成可信資料使用的例子 |
| Borsky, Unterberger. Bad weather and flight delays: The impact of sudden and slow onset weather events. *Economics of Transportation* | 10.1016/j.ecotra.2019.02.002 | 天氣對延誤的影響依天氣事件的型態而異（僅依標題，摘要未取得） |

## 申報的誘因：補償義務與延誤保險

成因欄位不只是統計資料：延誤是否「可歸責於航空公司」決定旅客能得到什麼。這是
各公司有動機把延誤記入天氣的制度背景，也說明清單對誰有用。

| 文獻 | arXiv／DOI | 支撐什麼 |
|---|---|---|
| 14 CFR § 234.4. Reporting of on-time performance | ecfr.gov/current/title-14/section-234.4 | 成因分類的法規定義。Extreme weather 的認定「依適用的法規標準及／或航空公司的判斷」，判斷權在申報者 |
| Bureau of Transportation Statistics. Understanding the Reporting of Causes of Flight Delays and Cancellations | bts.gov（網頁） | weather 欄只收極端天氣；非極端天氣造成的延誤歸 NAS |
| Bureau of Transportation Statistics. Technical Directive Number 14: On-Time Reporting | bts.gov（網頁） | 航空公司可依事先提交的 Procedural Statement，一致採用「全部成因或主要成因」及「自第一分鐘或六分鐘起」的申報口徑。公司的零值與多成因比例可能反映合法口徑，跨公司比較及模擬的誠實基準須納入此差異；若拿不到各公司聲明，這是不可消除的識別限制 |
| U.S. DOT. Airline Customer Service Commitments for Controllable Flight Disruptions. *Federal Register* 90 | 文件編號 2025-22415 | 大型航空公司承諾在延誤或取消「可歸責於公司」時提供改訂、餐食與住宿；交通部以成因是否在公司控制內認定此承諾是否適用。成因分類與旅客補償直接相連的官方文件 |
| U.S. DOT. Cause of Airline Delay and Cancellation Categories Under Section 511(b) of the FAA Reauthorization Act of 2024. *Federal Register* 91 | 文件編號 2026-18040 | 修訂成因申報：十類事件自 Air Carrier 代碼移出，另立新類別。本研究的樣本期間適用修訂前的分類 |
| Gnutzmann, Śpiewanowski. Can consumer rights improve on-time performance? Evidence from European Air Passenger Rights. *Transport Policy* | 10.1016/j.tranpol.2023.03.017 | 歐盟旅客權益法規（延誤須賠償，特殊情況除外）對準點表現的影響（僅依標題，摘要未取得） |
| Fukui, Nagata. Flight cancellation as a reaction to the tarmac delay rule: An unintended consequence of enhanced passenger protection. *Economics of Transportation* | 10.1016/j.ecotra.2014.02.004 | 停機坪延誤罰則實施後，航空公司以取消航班因應。旅客保護規則改變航空公司的營運與申報對象（僅依標題，摘要未取得） |
| Fukui, Dresner, Miyoshi. Penalties as prices? Evaluating airline strategic responses to tarmac delay penalties in the US. *Transportation Research Part A* | 10.1016/j.tra.2026.105063 | 以 difference-in-differences 比較受罰與未受罰公司：罰則的嚇阻效果有限，罰款相對於淨利小時更弱。監理行動對航空公司行為的實際效果，與監理優先清單的用途相關 |

延誤保險的學術文獻很少。Crossref 查得的只有保費計算的會議論文與理賠平台的
設計，未列入。parametric 型的延誤保險以延誤時數為給付條件，不問成因，故不
構成申報的誘因；與成因相連的是上表的補償承諾。

## 模擬研究

| 文獻 | arXiv／DOI | 支撐什麼 |
|---|---|---|
| Morris, White, Crowther. Using simulation studies to evaluate statistical methods. *Statistics in Medicine* | 1712.03198 | ADEMP 框架，模擬研究的設計與報告方式 |
| Schreck, Slynko, Saadati, Benner. Statistical Plasmode Simulations: Potentials, Challenges and Recommendations | 2305.06028 | 保留真實共變量、只模擬結果的 plasmode 設計 |
| Shaw et al. A cautionary note for plasmode simulation studies in the setting of causal inference | 2504.11740 | 比較兩種 plasmode：處理與共變量一起重抽（Sample Treatment）會使 propensity score 類估計量顯得偏誤、涵蓋率不足，只重抽共變量再生成處理（Generate Treatment）則不會；建議生成模型取有彈性的形式。本設計保留真實的執飛公司，屬前者 |
| Karim. A Plasmode Benchmark Can Refute but Cannot Certify | 2609.26186 | 模擬只能否證方法，不能證實，結論的界限 |
| Schuemie et al. Empirical confidence interval calibration for population-level effect estimation studies in observational healthcare data. *PNAS* | 10.1073/pnas.1708282114 | 從真實資料中的 negative controls 合成效果已知的 positive controls，用以檢查區間涵蓋率；本題可借「植入已知偏移」的驗證思路，但不能直接移植其因果效果校準公式 |
| Steinbuss, Böhm. Benchmarking Unsupervised Outlier Detection with Realistic Synthetic Data | 2004.06947 | 先以真實資料中已標記為正常的樣本擬合生成模型，再由模型生成正常樣本及局部、依賴關係與全域異常；各偵測器表現隨異常型態改變。本題借其區分異常作用範圍的思路，檢查特定機場、月份或天氣條件下的改報；但沒有可靠的「正常申報」標籤，不能照搬其生成流程，且模擬結果只涵蓋所設型態 |
| Emmott et al. A Meta-Analysis of the Anomaly Detection Problem | 1503.01158 | 異常比例、難度、聚集程度及特徵相關性均改變偵測表現；模擬設計須變動受影響公司數、偏移強度、稀疏程度與天氣情境，並同時報告排序及查核名單的表現 |
| Goswami et al. Unsupervised Model Selection for Time-series Anomaly Detection | 2210.01078 | 無真值的時間序列中，植入合成異常可作選模的替代訊號之一；支持用模擬比較候選方法，但該文亦表明替代訊號有誤差，不能把植入異常的高分等同真實資料準確率 |

## 單位層級清單的模擬驗證

| 文獻 | arXiv／DOI | 支撐什麼 |
|---|---|---|
| Brakenhoff, Roes, Moons, Groenwold. Outlier classification performance of risk adjustment methods when profiling multiple providers. *BMC Medical Research Methodology* | 10.1186/s12874-018-0510-1 | 重抽真實病人共變量後，重新生成機構歸屬與二元結果，指定異常機構，變動異常者比例、偏離強度與機構規模及異常身分的關聯，再比對 sensitivity、specificity、PPV、NPV。固定效應較易找出異常者，也更常誤列；隨機效應相反。本題應以公司為單位評估清單。其結果模型刻意採隨機截距、所有調整模型皆正確設定且無未觀測混雜，因此不能據此斷言隨機效應在本題較準；須改變資料生成機制 |
| Ding, Hubbard, Rutter, Simon. Assessing the accuracy of profiling methods for identifying top providers: performance of mental health care providers. *Health Services and Outcomes Research Methodology* | 10.1007/s10742-012-0099-5 | 用模擬真值評估機構的門檻分類與排序；當機構間真實差異小時，各方法的分類準確率均差，故應提供排序不確定性，並在模擬中納入接近無公司差異的情境 |
