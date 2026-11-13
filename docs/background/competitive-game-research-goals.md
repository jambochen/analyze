# 一對一對戰資料研究的目標與交付

本筆記比較論文想回答的問題與證據終點，用於判斷現有分析的定位。這是針對性文獻查詢，不是系統性回顧。論文年份是書目資訊；不記錄本專案的估計值。與本研究的對照是閱讀後的判斷，不是原作者對本研究的評價。

核心對照採 Gee（近況）、Chowdhary（連勝連敗）、Yuksel（配對）、Chen–Joachims（預測）及 EOMM（平台目標）。這五篇的下載版本、源碼可得性與逐篇方法評讀見[核心方法與可採用部分](competitive-game-methods.md)。

## 同來源或同平台的西洋棋資料

### Quantifying human performance in chess

Chowdhary、Iacopini、Battiston，Scientific Reports（2023）。使用 Lichess 官方資料庫的 2013–2016 年資料，分析聚焦 Blitz；與本專案同來源且期間重疊，不是相同清理後樣本。

- **目標**：高手與新手的競技生涯有何差別？連勝連敗、開局多樣性與專精如何隨能力及生涯階段變化？
- **交付**：大規模行為型態與生涯描述；不以交付停手規則或上線配對系統為必要終點。
- **資料要求**：連勝部分使用賽果序列；開局部分另需棋步或開局分類。
- **與本研究的關係**：連勝的比較基準本身可成為研究問題。其打亂時間順序的虛無模型保留個人總勝和負，與本研究按局前條件計算預期的問題不同；觀察到序列結構不直接識別心理機制。

來源：[正式論文與 Methods、Data availability](https://pmc.ncbi.nlm.nih.gov/articles/PMC9902564/)。

### Investigating Experiential Effects in Online Chess using a Hierarchical Bayesian Analysis

Gee、Seese、Curley、Ward，arXiv（2025，以下核對其預印本）。Lichess 上抽取不同評分群體及特級大師帳號的對局歷史，非本專案官方月檔的 arena 全體樣本。

- **目標**：近期勝負與下一局成功機率是否有共同關係？是否依玩家、能力、速度及歷史長度而變？
- **交付**：全體與個人層次的贏輸效應估計，以及模型處理時間結構的檢查；未見全體一致的效應也是研究答案。
- **資料要求**：賽果、玩家、分差、顏色與時間，不依賴棋局引擎分析。
- **與本研究的關係**：最接近近況與連續對局問題。研究效應是否存在與提供行為建議是不同終點；不能僅因控制分差便將估計解讀為心理效果。

來源：[論文全文，Introduction、Data、Model Validation](https://arxiv.org/html/2503.21713v1)。

### Learning in online chess increases with more time spent thinking and diversity of experience

Schut、Russek、Kuperwajs、Mattar、Ma、Griffiths，CogSci（2025）。使用 Lichess 玩家的長期對局資料；此處核對會議論文的書目與摘要，未完成全文方法評讀。

- **目標**：思考時間與經驗多樣性是否能預測長期技能進步？檢查人工智慧學習原理是否也適用於人類。
- **交付**：練習型態與能力成長的關聯證據，研究終點是技能發展。
- **資料要求**：落子思考時間、開局位置及長期能力軌跡，超出單純勝負紀錄。
- **與本研究的關係**：這是可考慮的不同研究方向，不能把現有短期近況模型直接改稱學習效果，也不能由摘要推出某種訓練必然有效。

來源：[會議論文書目與摘要](https://escholarship.org/uc/item/5c76v07h)。

### Measuring the Non-Transitivity in Chess

Sanjaya、Wang、Yang，Algorithms（2022；預印本 2021）。使用 Lichess 與 FICS 真人對局。

- **目標**：單一由強到弱的排序是否足以描述競爭？真人棋局是否也存在類似剪刀石頭布的循環相剋結構？
- **交付**：以 Nash clustering 與循環計數量化策略空間，並討論對群體式 AI 訓練的意涵。
- **資料要求**：對戰關係、賽果與評分，核心不要求逐步引擎分析。
- **與本研究的關係**：提供評分模型以外的問題定位；其彙整策略空間結論不能直接視為個別玩家相剋的穩定因果證據。

來源：[作者預印本](https://arxiv.org/abs/2110.11737)、[正式出版頁](https://doi.org/10.3390/a15050152)。

## 相近的雙人競賽問題

### Cognitive performance in competitive environments: Evidence from a natural experiment

González-Díaz、Palacios-Huerta，Journal of Public Economics（2016）。資料為職業西洋棋多局對抗賽，不是 Lichess。

- **目標**：高壓競爭中的先後處境是否影響認知表現？利用首局執白由隨機程序決定、後續交替執白的制度研究整場勝負。
- **交付**：制度順序對競爭結果的自然實驗證據，以及心理機制的解讀。
- **資料要求**：多局對抗賽的制度、首局顏色分派與結果。
- **與本研究的關係**：示範同類賽果如何回答機制或制度問題；隨機來源必須對應到實際分派的處理，不能將所有賽後關聯都當成隨機效果。

來源：[作者提供的正式論文](https://eio.usc.es/pub/julio/papers/21._Cognitive_performance_in_competitive_environments-_Evidence_from_a_natural_experiment_web.pdf)。

### Predicting Matchups and Preferences in Context

Chen、Joachims，KDD（2016）。以 StarCraft II、網球等資料評估情境式雙人結果模型。

- **目標**：納入玩家特徵及比賽情境，能否比單一實力模型更好預測雙人對戰結果，並容納非遞移關係？
- **交付**：通用機率模型與樣本外預測比較；配對只是潛在用途，並非已證實的部署成果。
- **資料要求**：對戰結果與局前玩家／比賽特徵，不必將局中操作當輸入。
- **與本研究的關係**：預測品質本身可以構成論文終點，但新增模型仍需對應明確的基準缺口。

來源：[作者提供的論文全文](https://www.cs.cornell.edu/~tj/publications/chen_joachims_16b.pdf)。

### Whole-History Rating: A Bayesian Rating System for Players of Time-Varying Strength

Rémi Coulom，Computers and Games（2008）。實證使用 KGS 圍棋，不是 OGS。

- **目標**：由歷史賽果估計隨時間改變的實力，兼顧預測準確度與計算速度。
- **交付**：WHR 評分演算法、技能軌跡與不確定性，以及不同評分方法的預測與效率比較。
- **資料要求**：主要需要玩家、日期與賽果；讓子、貼目及新手成長仍需適當處理，不需棋步。
- **與本研究的關係**：直接支持歷史賽果驅動的預測與動態評分定位。完整歷史的回溯重建與當下可用資訊的預測須區分。

來源：[作者研究頁](https://www.remi-coulom.fr/WHR/)、[作者論文與圍棋實驗](https://www.remi-coulom.fr/WHR/WHR.pdf)。

### TrueSkill Through Time: Revisiting the History of Chess

Dangauthier、Herbrich、Minka、Graepel，NIPS（2007）。使用 ChessBase 歷史西洋棋賽果，非 Lichess。

- **目標**：利用完整生涯重建玩家在各時點的技能及可信程度，並刻畫和局傾向。
- **交付**：歷史技能後驗分布、實力軌跡及玩家別和局傾向模型。
- **資料要求**：玩家、時間、勝敗和，核心不需棋步。
- **與本研究的關係**：實力變化的解讀本身是交付；用較晚賽果回頭修正較早技能屬回溯推論，不能直接當局前預測。

來源：[NeurIPS 正式書目](https://papers.neurips.cc/paper_files/paper/2007/hash/9f53d83ec0691550f7d2507d57f4f5a2-Abstract.html)、[全文](https://papers.nips.cc/paper/3331-trueskill-through-time-revisiting-the-history-of-chess.pdf)。

### Skill-Based Matchmaking for Competitive Two-Player Games

Cem Yuksel，Proceedings of the ACM on Computer Graphics and Interactive Techniques（2024）。一般雙人競技的合成模擬，不是具名遊戲的真實對局實證。

- **目標**：快速追蹤玩家實力改變，透過配對接近目標勝率，降低長連勝與長連敗。
- **交付**：評分更新法與配對演算法，利用已知潛在實力的模擬比較追蹤與配對表現。
- **資料要求**：依賽果與維護的玩家狀態更新，不需棋步或操作特徵。
- **與本研究的關係**：貼近評分轉成配對決策，但模擬表現不等於真實體驗或留存收益。此文的出版處不應記為 ACM Transactions on Graphics。

來源：[作者研究頁](https://www.cemyuksel.com/research/matchmaking/)、[全文與 Experimental Setup](https://www.cemyuksel.com/research/matchmaking/i3d2024-matchmaking.pdf)。

### EOMM: An Engagement Optimized Matchmaking Framework

Chen、Xue、Kolen、Aghdaie、Zaman、Sun、El-Nasr，WWW（2017）。使用 EA 未具名遊戲的 1v1 賽果；原文未確認遊戲名稱，不能寫成 FIFA。

- **目標**：降低配對後停止遊玩的風險，讓投入度成為配對目標。
- **交付**：技能模型、停止遊玩風險模型與最小權重完美匹配構成的決策框架。用真實資料估模型，再模擬比較政策。
- **資料要求**：賽果、歷史間隔、進球差、近期遊玩量與後續再玩紀錄，不需逐步操作。
- **與本研究的關係**：示範從局前機率接到平台目標，但其政策比較不是線上隨機實驗；特定時間窗內未再玩也不等於永久流失。

來源：[作者團隊全文與 Case Study](https://fuxiailab.github.io/EOMM/WWW2017_EOMM.pdf)、[預印本](https://arxiv.org/abs/1702.06820)。

### Video Game Telemetry as a Critical Tool in the Study of Complex Skill Learning

Thompson、Blair、Chen、Henrey，PLOS ONE（2013）。使用 StarCraft II 雙人人類對戰 replay、玩家背景資料及職業比賽資料。

- **目標**：不同熟練程度所依賴的認知與操作能力是否相同？
- **交付**：從視角切換、動作延遲等行為辨識技能級別，並比較技能階段的重要行為。
- **資料要求**：需要 replay 操作與視角移動，單純賽果不夠。
- **與本研究的關係**：屬技能與學習研究；不能把辨識技能的結果直接當成練習方法的因果效果。

來源：[PLOS ONE 全文，Data Collection 與 Analysis](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0075129)。

### Attention-based Initiative in Real Time Strategy

Oxenstierna、Murali、Björkman，DiGRA（2024）。使用 AoE II 雙人對戰 replay，以 MGZ 解析操作。

- **目標**：把主動權轉成注意力與空間活動指標，檢查其與對局優勢的關係。
- **交付**：可量化的主動權、隨比賽進行更新的勝負預測與特徵比較。
- **資料要求**：局中操作、位置、對象與時間，對局摘要不夠。
- **與本研究的關係**：遊戲相同，但屬局中預測；不能作為僅用局前資訊模型的同條件準確度基準。

來源：[DiGRA 正式論文](https://dl.digra.org/index.php/dl/article/download/2224/2221/2239)。

## 對現有分析定位的判斷

文獻的終點可分為現象與機制、未來賽果預測、回溯技能重建、配對決策及局中行為解讀。它們使用相近遊戲，卻不需要回答同一個問題。

本專案的近況與校準可對照 Gee、Coulom 及 Chen–Joachims；連勝的條件比較可對照 Chowdhary；配對取捨可對照 Yuksel 與 EOMM。這些對照提供研究問題的定位，不表示已達到相同的效益，也不表示本專案必須增加相同模型。

若把問題定位為「現有評分遺漏什麼資訊，這些資訊在哪些群體仍有預測增量」，可用現有時間外比較回答。若定位為「應如何配對」，須指定平衡、尾端或個人保護目標，現有結果只支持固定批次的取捨。若定位為「如何練習、休息或取得局中主動權」，則需要不同的資料或識別設計。

本次未核對到直接使用本專案 OGS 資料版本或 AoE2 粉絲網站快照的對應論文；這不表示此類文獻不存在。KGS 不等於 OGS，AoE2 replay 不等於本專案的賽果快照，同平台也不等於同一分析樣本。
