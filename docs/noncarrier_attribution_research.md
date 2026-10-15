# 非 Air Carrier 歸因的二類稽核：方法與證據

## 研究對象與分類邊界

每班有申報成因分鐘數的航班，令 `noncarrier = weather + NAS + security + late_aircraft`，與 `carrier` 相加為申報總分鐘數。此處的 `noncarrier` 只表示**未記入 Air Carrier 欄**，不可命名為「外部原因」「不可控原因」或已證實的「甩鍋」。BTS 將 Air Carrier 定義為航司控制內的事由、NAS 包括非極端天氣及機場／航管因素；`late-aircraft` 只指出同機尾的前序航班晚到，**不記錄其根本原因**。[BTS 成因定義](https://www.bts.gov/topics/airlines-and-airports/understanding-reporting-causes-flight-delays-and-cancellations)

此切法與航空文獻的「可由航司控制」切法不同。Bilotkach 與 Lakew 分析 BTS 資料時，把 `carrier + late-aircraft` 暫視為較可能受航司控制，將後者稱為半內生，並明言前序航班也可能因天氣或 NAS 晚到。該文是**二分成因份額已有航空應用**的先例，卻不能支持把 `late-aircraft` 一律當作外因。本題若關注申報是否避開 `carrier`，可採前段的欄位切法；若關注實質責任，須另報 `weather + NAS + security` 對 `carrier + late-aircraft` 的敏感度版本，且兩者都不能從欄名推出真正責任。[Bilotkach 與 Lakew 原文，§2.1、§3](https://pauloslakew.github.io/research/BilotkachLakew_Feb2019.pdf)、[BTS 成因定義](https://www.bts.gov/topics/airlines-and-airports/understanding-reporting-causes-flight-delays-and-cancellations)

BTS 的申報指令容許公司選擇「全部成因或主要成因」及從第一分鐘或達門檻才追蹤，但須一致執行並提交 Procedural Statement；到達延誤超過出發延誤的差額歸 NAS。因此，即使兩家公司對底層事件判斷相同，合法的申報口徑、空中增加的延誤、前序航班配置仍可能改變二類份額。這些因素是公司比較的設計條件，不能被二分消除。[BTS Technical Directive No. 14，Reporting Causes 與 Procedural Statement](https://www.bts.gov/topics/airlines-and-airports/number-14-time-reporting)

## 已有模型能否直接回答

| 做法 | 在二類問題中的地位 | 證據及適用界線 |
|---|---|---|
| 現有四類 Poisson／多項模型的預測值加總 | **同一已估模型的事後彙總**。在共同航班或單元條件下先求各類預測分鐘，再加總非 `carrier` 類並標準化，可問模型隱含的二類份額；不需增加模型。 | 多項與 Poisson 對數線性等價見 [Lee、Green、Ryan 原文](https://arxiv.org/abs/1707.08538)。若各類的線性預測值為 η，二類 log odds 是 `logsumexp(η_noncarrier) − η_carrier`，一般**不是各類公司係數的線性和**；這是模型式子的代數結論。現有模型依 `docs/design.md` 剔除 `security` 後重新歸一化，因此其預測和只對「其餘四欄」成立，與納入五欄的二類目標分母不同。 |
| 重新擬合二類 Poisson／binomial logit | **更換結果粒度與參數限制的主模型**，不是新統計方法。直接估相同條件下公司把分鐘記在 `noncarrier` 而非 `carrier` 的差異；省去四類內部配置，但無法說明流向哪一欄。 | 若每個航班／聚合單元有自身的共同 Poisson 截距，固定該單元總分鐘後的二類條件概似與 binomial logit 相同，是 [Baker 的 multinomial–Poisson 轉換](https://doi.org/10.2307/2348134)在二類的特例；[Lee 等原文](https://arxiv.org/abs/1707.08538)討論同一轉換。分鐘數不能當成彼此獨立的 Bernoulli 判斷，故 binomial 變異數及一般標準誤不可照單全收，應以航班／適當群組的相關及穩健推論處理。 |
| 逐班 fractional logit | **另一個條件平均數模型**：用逐班 `noncarrier / total`，可保留恰為零或一的航班，保留逐班氣象；可以選每班等權或依總分鐘加權，兩者回答不同問題。 | [Papke、Wooldridge 原始工作論文](https://www.nber.org/papers/t0147)明確允許 0 與 1，並以準概似估條件平均數；不要求 beta 分布。若加權恰用申報總分鐘且採相同 logit 線性預測式，分數方程與二類 binomial 配置模型相同，**不是獨立的新識別來源**。逐班等權及不同氣象控制才可能提供實質不同的檢查。 |
| 端點／兩部模型 | **新增可辨的行為面向**：先問是否全記 `carrier`、是否全記非 `carrier`，再問混合記錄時的份額；適用於端點質量很大且不同公司有不同端點模式。 | [Ospina、Ferrari 原文](https://arxiv.org/abs/1103.2372)支援端點質量與內部連續份額分開建模；它**不證明**端點是系統預設、真實單一成因或刻意歸因。現有 Y2 的 one-inflated 模型移至此處是同類結構，重點是二類結果可能同時有 0 與 1。 |
| 分層列聯表、CMH、GEE、經驗貝氏、名次區間 | 分別檢查分層關聯、相關性、稀疏單位的收縮及清單不確定性；皆已在 `docs/methods.md` 的候選池。二分後可更簡單地套用，**不會自行補上成因真值**。 | [Normand、Shahian 原文](https://arxiv.org/abs/0710.4622)說明機構 profiling 的風險調整、樣本量及排序問題；這些方法處理估計與清單品質，不替代稽核驗證。 |

兩類模型的最大價值是**直接對齊「避開 Air Carrier 欄」的問題**。現有四類模型仍應保留，用來辨識二類高值來自天氣、NAS 還是 `late-aircraft`：不同的來源有不同的申報規則與可核查資料。只看二類可把特定天氣欄異常與普遍遠離 `carrier` 的型態混在一起；現有四類模型已顯示這兩種型態均可能存在（見 `docs/findings.md`），此為本研究資料的觀察而非外部文獻結論。

## 甩鍋推論可借用的稽核設計

Silverman 與 Skinner 在臨床上接近、但給付不同的診斷類別之間比較較高給付類別的占比；其可移植的思路是**先界定可裁量的分類邊界，再問申報者在可比較情境下是否系統性偏向一側**。該文沒有航空成因真值，不能證明本題任何特定公司的動機。[Silverman、Skinner 原始論文](https://www.nber.org/papers/w8133)

Geruso 與 Layton 以同一保險人的風險分數在不同給付誘因下的變化檢視診斷編碼強度，顯示**誘因變動或可比較對照**如何增加「策略性申報」論證的說服力。本題若沒有相應的制度變動、同一航班的獨立判讀，只有同儕控制後的份額差，不能承接其因果結論。[Geruso、Layton 原始論文](https://www.nber.org/papers/w21222)

Sanghavi、Pan、Caudry 把護理之家自報跌倒事件逐筆連到醫療理賠，才能估報告遺漏；其多層模型用於尋找報告差異。本題的觀測氣象是候選成因的外部資訊，**並非逐班成因的獨立真值**，故能做風險調整與查核排序，無法照搬漏報率估計。[Sanghavi 等原始論文摘要與方法](https://pubmed.ncbi.nlm.nih.gov/31884706/)

DOT 監察長已發現 BTS 對成因申報欠缺一致性核查程序、技術指令可被不同解讀，且航空公司申報與 FAA 資料有高比例不一致。這支援建立**查核優先清單**；FAA 與 BTS 的不一致本身尚未裁定哪一方正確，不能當逐班 gold standard。[DOT OIG 官方稽核報告](https://www.oig.dot.gov/library-item/46490?page=1)

## 識別限制與適用次序

1. **主要結果宜稱「非 Air Carrier 申報傾向」**。同條件下公司差異只能辨識相對申報型態；航路、前序航班、班表、客艙與地面作業、未觀測天氣，以及合法申報口徑，皆可能造成差異。[BTS 成因定義](https://www.bts.gov/topics/airlines-and-airports/understanding-reporting-causes-flight-delays-and-cancellations)、[BTS Technical Directive No. 14](https://www.bts.gov/topics/airlines-and-airports/number-14-time-reporting)
2. **先報二類與四類並列，再換切法**。`noncarrier` 高且四類模型的天氣、NAS、前序航班均偏高，和只有天氣偏高的情境，指向不同查核方向。另報排除 `late-aircraft` 或把它與 `carrier` 同組的版本，衡量結論有多少由未標明根因的傳播延誤驅動。[BTS 成因定義](https://www.bts.gov/topics/airlines-and-airports/understanding-reporting-causes-flight-delays-and-cancellations)、[Bilotkach 與 Lakew 原文](https://pauloslakew.github.io/research/BilotkachLakew_Feb2019.pdf)
3. **天氣題的好天氣對照不能直接搬來**。晴天時 NAS 仍可由壅塞或航管造成，`late-aircraft` 仍可由前序班機造成；因此「晴天時非 `carrier` 應為零」不是有效的排除限制。好天氣可用來查看天氣欄，不能直接驗證四欄加總後的非 `carrier` 欄。[BTS 成因定義](https://www.bts.gov/topics/airlines-and-airports/understanding-reporting-causes-flight-delays-and-cancellations)
4. **無獨立標籤時不估不實申報率**。Dawid–Skene 的無真值錯誤率模型仰賴同一對象的重複評定；本題每班只有該航司的一筆分類，真實成因分布與錯誤分類機率無法從單筆申報分開。若要判定是否甩鍋，需取得逐班營運紀錄、前序航班根因或獨立查核樣本。[Dawid、Skene 原始論文](https://doi.org/10.2307/2346806)、[BTS 成因定義](https://www.bts.gov/topics/airlines-and-airports/understanding-reporting-causes-flight-delays-and-cancellations)

**方法判斷：**二類 Poisson／logit 最直接回答申報邊界問題；逐班等權 fractional logit 檢查加權與聚合口徑；端點模型只在 0／1 大量堆積且公司模式不同時增加行為資訊。四類模型與 `late-aircraft` 分組敏感度是解讀二類結果的必要配套。稽核清單應表述為「值得查核的相對異常」，不是不實申報的判決。
