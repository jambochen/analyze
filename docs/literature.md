# 文獻

標「已查證」者已在出版者頁面或 arXiv 核對標題與出處；其餘僅見於搜尋結果或作者頁面，引用前須核對。

## 連勝連敗與 tilt

- **Streaks and Coping: Decoding Player Performance in League of Legends Using Big Data from Top Players' Matches**（FDG 2024）。以頂尖玩家的對戰序列分析連勝連敗後的表現與換角、休息的關係。與本題主問題最接近。https://dl.acm.org/doi/fullHtml/10.1145/3665463.3678787
- **Understanding Tilt in Esports: A Study on Young League of Legends Players**（CHI 2021）。高中電競選手的問卷與訪談，界定 tilt 的觸發因素，屬質性研究。https://dl.acm.org/doi/pdf/10.1145/3411764.3445143
- **Playing with Streakiness in Online Games: How Players Perceive and React to Winning and Losing Streaks in League of Legends**（CHI 2018）。https://dl.acm.org/doi/pdf/10.1145/3173574.3174152
- **Investigating Experiential Effects in Online Chess using a Hierarchical Bayesian Analysis**（Gee、Seese、Curley、Ward，arXiv 2503.21713，已查證，已讀全文）。以 Lichess 資料檢驗贏輸效應：近幾局的勝率是否影響下一局。連續對局以開局間隔切分，Bullet 與 Blitz 門檻不同；模型為階層貝氏 logistic 迴歸，控制分差並加玩家層隨機效應。結論為沒有全體一致的效應，少數玩家個別有。資料是先選定玩家、再爬取其全部對局；論文稱資料與爬取程式碼放在作者的 GitHub，未查得網址。與主問題最接近；本題的差異在於以 arena 的隨機對手識別上一局結果的效果、全體玩家而非抽樣、和局不丟棄、處理續下的選擇偏誤。https://arxiv.org/abs/2503.21713
- **Quantifying human performance in chess**（Chowdhary、Iacopini、Battiston，arXiv 2207.07780，已查證，已讀全文）。Lichess 2013 至 2016 年全部對局。以打亂時間順序為虛無模型，發現連勝連敗長於隨機，新手的連勝更長；另發現高手專精少數開局，但連續兩局間比新手更常換開局。虛無模型只保留勝和負總數，忽略等級分的變化：進步中的玩家等級分追不上實力，本來就會連勝。本題以對局前等級分的期望得分為基準，修正的正是這一點。資料取自官方資料庫 database.lichess.org 的月檔，與本研究同一來源。https://arxiv.org/abs/2207.07780
- **Fast Algorithm for Calculating Probability of Chess Winning Streaks**（arXiv 2503.11669，已查證）。給定每局勝和負機率，計算各種連勝的機率。https://arxiv.org/abs/2503.11669
- **Surprised by the Hot Hand Fallacy? A Truth in the Law of Small Numbers**（Miller、Sanjurjo，Econometrica 86(6)，2018，已查證）。證明在有限長度的序列中，「連續 k 次之後下一次的成功率」這種常見的量測本身有偏誤：序列越短、連勝長度越長，偏誤越大；修正後推翻了 hot hand 經典研究的結論。附屬問題若以「連輸 k 局之後的表現」為量測，須處理同一種選擇偏誤。https://econometricsociety.org/publications/econometrica/2018/11/01/surprised-hot-hand-fallacy-truth-law-small-numbers
- **Cognitive performance in competitive environments: Evidence from a natural experiment**（González-Díaz、Palacios-Huerta，Journal of Public Economics 139，2016，已查證）。職業棋賽以抽籤決定誰先執白，先執白者的勝率明顯較高，實力相近時差距更大；作者解讀為比賽中的心理狀態影響認知表現。是「前一局的處境影響下一局表現」在西洋棋上以隨機指派識別的先例。https://researchonline.lse.ac.uk/id/eprint/67144
- **Kramnik vs Nakamura: A Chess Scandal**（arXiv 2409.06739，已查證）。以線上快棋連勝爭議為例，說明解讀連勝時的檢察官謬誤。https://arxiv.org/abs/2409.06739

## 匹配與流失

- **EOMM: An Engagement Optimized Matchmaking Framework**（EA，arXiv 1702.06820，已查證，已讀全文）。以投入度而非勝率最佳化匹配：把等待配對的玩家建成完全圖，以 minimum weight perfect matching 求總流失最小的配對。關鍵假設為給定本局結果、對手狀態不再影響流失；策略比較以 EA 資料設定的模擬進行。方法與本研究的用法見 `docs/methods.md`。https://arxiv.org/abs/1702.06820
- **EnMatch**：以玩家投入度為目標的匹配（網易伏羲，AAAI 2024）。https://ojs.aaai.org/index.php/AAAI/article/view/28760
- **A Data-Driven Decision Support Framework for Player Churn Analysis in Online Games**（網易伏羲，KDD 2023）。資料為網易內部，不公開。https://dl.acm.org/doi/10.1145/3580305.3599759
- 網易伏羲的 OptMatch（KDD 2020）與全域匹配（KDD 2021）：見作者頁面 https://wu-runze.github.io/collection-archive/
- **Match experiences affect interest: Impacts of matchmaking and performance on churn in a competitive game**（Kang、Suh、Kim，Heliyon 2024，已查證，已讀全文）。Netmarble 手遊的伺服器紀錄，論文明寫資料不公開。玩家日為單位、14 天無對局為流失，以雙向固定效應 logistic 迴歸估計分差、分差變異、勝率與連勝連敗對流失的效果。配對由系統依分數決定，作者承認為觀察資料。方法與本研究的用法見 `docs/methods.md`。https://pmc.ncbi.nlm.nih.gov/articles/PMC10839887/
- **The Role of Skill in Matchmaking**（Activision 白皮書，以新聞報導核對存在與內容，全文未讀）。Call of Duty: Modern Warfare 3 的內部實驗，逐步放寬技術配對並追蹤離開與留存，結論為放寬後中途離開與不再遊玩增加；另含實力相當下連敗發生率的機率基準。內部資料，未經獨立驗證。https://www.activision.com/cdn/research/CallofDuty_Matchmaking_Series_2.pdf
- **Skill-Based Matchmaking for Competitive Two-Player Games**（Yuksel，ACM TOG 2024）。https://dl.acm.org/doi/10.1145/3651303
- **Opponent Indifference in Rating Systems: A Theoretical Case for Sonas**（arXiv 2209.03950，已查證）。玩家能看到對手分數並挑選或中止時，等級分系統會誘使玩家策略性選對手。支撐「大廳自選對手混淆分差效果」的論點。https://arxiv.org/abs/2209.03950
- **Mean-Field Analysis and Optimal Control of a Dynamic Rating and Matchmaking System**（arXiv 2512.22038，已查證）。以平均場理論分析等級分準確度與配對強度的關係。https://arxiv.org/abs/2512.22038
- **Paths, Trees, and Flowers**（Edmonds，Canadian Journal of Mathematics 17，1965，已查證）。一般圖上最大匹配的 blossom 演算法，minimum weight perfect matching 的實作依據。https://doi.org/10.4153/CJM-1965-045-4
- **Rule Designs for Optimal Online Game Matchmaking**（arXiv 1911.11852，已查證）。以排隊理論分析 2v2 配對規則與等待時間。https://arxiv.org/abs/1911.11852
- **CUPID**（arXiv 2406.19720）與 **CHAMP**（arXiv 2609.04870），已查證。MOBA 的重新配對系統，兼顧公平與位置偏好。https://arxiv.org/abs/2406.19720 、https://arxiv.org/abs/2609.04870
- **Cinder: A fast and fair matchmaking system**（arXiv 2602.17015，已查證）。組隊玩家實力分散時的公平配對。https://arxiv.org/abs/2602.17015
- **Competitive Balance in Team Sports Games**（arXiv 2006.13763，已查證）。以比分差預測對局平衡度。https://arxiv.org/abs/2006.13763

搜尋範圍內沒有找到以觀察資料、在隨機或近似隨機配對下估計匹配品質對留存因果效果的研究：EOMM 用模擬，CUPID 等是系統設計。此為主問題一的新穎性所在。範圍限於 arXiv 關鍵字搜尋，ACM 與 IEEE 的電競研究未系統搜尋。

## 其他對局資料上的研究

資料來源與取得方式見 `docs/other-data.md`。

- **What Drives Team Success? Large-Scale Evidence on the Role of the Team Player Effect**（Elbert 等，EC '25，arXiv 2506.04475，已查證）。以 Age of Empires II 的大型面板資料，利用玩家被近似隨機分配到臨時團隊，比較實際結果與依等級分的預測，分離個人對團隊的貢獻。論文頁面未寫明資料來源。https://arxiv.org/abs/2506.04475
- **The Leaderboard Illusion**（arXiv 2504.20879，已查證）。Chatbot Arena 的閉源模型被抽中對戰的頻率較高，加上選擇性公開分數，使排名偏誤。配對抽樣的設計會影響等級分。https://arxiv.org/abs/2504.20879
- **Improving Your Model Ranking on Chatbot Arena by Vote Rigging**（arXiv 2501.17858，已查證）。利用 Elo 機制，以操縱投票影響目標模型的排名。https://arxiv.org/abs/2501.17858
- **The Evaluation of Rating Systems in Team-based Battle Royale Games**（arXiv 2105.14069，已查證）。比較評估等級分系統的指標，部分指標受新玩家影響大。https://arxiv.org/abs/2105.14069

## 留存與流失的方法

- **Playtime Measurement with Survival Analysis**（arXiv 1701.02359，已查證）。以存活分析衡量遊戲時長。https://arxiv.org/abs/1701.02359
- **Measuring Player Retention and Monetization using the Mean Cumulative Function**（arXiv 1709.06737，已查證）。以 mean cumulative function 處理玩家尚未流失造成的設限。https://arxiv.org/abs/1709.06737
- **Churn Prediction in Mobile Social Games: Towards a Complete Assessment Using Survival Ensembles**（arXiv 1710.02264）與 **Games and Big Data: A Scalable Multi-Dimensional Churn Prediction Model**（arXiv 1710.02262），已查證。以 survival ensembles 預測流失，明確處理設限。https://arxiv.org/abs/1710.02264 、https://arxiv.org/abs/1710.02262
- **Game Data Mining Competition on Churn Prediction and Survival Analysis using Commercial Game Log Data**（arXiv 1802.02301，已查證）。NCSOFT 的流失預測競賽，資料跨越月費制改為免費制的前後。https://arxiv.org/abs/1802.02301
- **It's About Time: Using Discrete-Time Survival Analysis to Study Duration and the Timing of Events**（Singer、Willett，Journal of Educational and Behavioral Statistics 18(2)，1993，已查證）。離散時間 hazard 模型：每個時段一個截距的 logistic 迴歸，處理右設限。https://journals.sagepub.com/doi/10.3102/10769986018002155
- **Who Benefits from AI? Self-Selection, Skill Gap, and the Hidden Costs of AI Feedback**（arXiv 2409.18660，已查證）。線上棋平台上，越有動機的玩家越常使用 AI 回饋，造成效果的假象。本資料做不了同一題，但自我選擇造成假象的論證結構與續下的選擇偏誤相同。https://arxiv.org/abs/2409.18660

## 參考點與損失規避

- **A Model of Reference-Dependent Preferences**（Kőszegi、Rabin，Quarterly Journal of Economics 121(4)，2006，已查證）。以近期的理性預期作為參考點，得失相對於預期而定。對應「玩家對實際得分與期望得分的落差反應，而非對輸贏本身反應」的解釋；期望得分即參考點。https://rabin.scholars.harvard.edu/publications/model-reference-dependent-preferences
- **Is Tiger Woods Loss Averse? Persistent Bias in the Face of Experience, Competition, and High Stakes**（Pope、Schweitzer，American Economic Review 101(1)，2011，已查證）。PGA 巡迴賽 250 萬次推桿，以標準桿為參考點，頂尖選手同樣表現出損失規避。是「有經驗、高競爭下參考點效應仍存在」的實證先例。https://faculty.wharton.upenn.edu/wp-content/uploads/2012/04/Is-Tiger-Woods-Loss-Averse.pdf

## 選擇偏誤與量測誤差的方法

- **Review of inverse probability weighting for dealing with missing data**（Seaman、White，Statistical Methods in Medical Research 22(3)，2013，已查證）。IPW 修正完整案例分析的偏誤，涵蓋缺失機制模型的選擇、權重截斷、穩定化權重與 augmented IPW。續下的選擇等同於後段局序的「缺失」，以續下模型的倒數加權。https://journals.sagepub.com/doi/10.1177/0962280210395740
- **Correcting for regression dilution bias: comparison of methods for a single predictor variable**（Frost、Thompson，Journal of the Royal Statistical Society Series A 163(2)，2000，已查證）。解釋變數含量測誤差時迴歸係數往 0 收縮，比較六種修正係數的估計法。只適用於 classical 量測誤差；Glicko 的等級分為事後平均，誤差屬 Berkson 型，不使斜率衰減，所以本研究不以它解釋校準斜率小於 1，只作為對照的出處。https://ideas.repec.org/a/bla/jorssa/v163y2000i2p173-189.html

## 等級分系統

- **Skill Issues: An Analysis of CS:GO Skill Rating Systems**（arXiv 2410.02831，已查證）。以同一資料比較 Elo、Glicko-2 與 TrueSkill，並模擬等級分如何影響之後的配對。https://arxiv.org/abs/2410.02831
- **Paired comparison models with strength-dependent ties and order effects**（arXiv 2505.24783，已查證）。配對比較模型同時處理和局與先手優勢，兩者皆隨實力增強。以錦標賽西洋棋為例。https://arxiv.org/abs/2505.24783
- **Rating competitors in games with strength-dependent tie probabilities**（arXiv 2506.11354，已查證）。依上述模型建立等級分系統。https://arxiv.org/abs/2506.11354
- **Understanding and Pushing the Limits of the Elo Rating Algorithm**（arXiv 1910.06081，已查證）。寫出 Elo 對和局的隱含假設，提出可調和局頻率的 κ-Elo。https://arxiv.org/abs/1910.06081
- **Empirical parameterization of the Elo Rating System**（arXiv 2512.18013，已查證）。以預測準確度調校 Elo 參數。https://arxiv.org/abs/2512.18013
- **Engine-Equal, Human-Unequal**（arXiv 2607.25655，已查證）。以 Lichess 資料，用「實際結果減等級分預測」量測偏差，並以不重疊的帳號分組驗證可重複性。主題不同，校準與驗證設計可參考。https://arxiv.org/abs/2607.25655
- **Measuring the Non-Transitivity in Chess**（arXiv 2110.11737，已查證）。以真人對局資料量化勝負的循環結構。https://arxiv.org/abs/2110.11737
- **QuickSkill: Novice Skill Estimation in Online Multiplayer Games**（arXiv 2208.07704，已查證）。新手等級分不準的冷啟動問題，對應本題的新手期。https://arxiv.org/abs/2208.07704
- **Analysis of player's in-game performance vs rating: Case study of Heroes of Newerth**（arXiv 1305.5189，已查證）。比較遊戲內表現與 MMR，指出玩家會利用系統弱點。https://arxiv.org/abs/1305.5189
- 團隊賽的實力加總與以行為特徵評分：arXiv 2106.11397、2207.00528、2112.04379，已查證。用於「類推到電競的界線」
- Glicko-2 的原始定義（Glickman）：待補。Lichess 的實作見「Lichess 的等級分實作」

## Lichess 的等級分實作

以下皆為 lila 原始碼與 commit 的一手出處，已查證。

- **參數**：2014 年 7 月的版本（commit `f09a2e9893`）初始值為等級分 1500、RD 350、volatility 0.06，τ = 0.75（`modules/round/src/main/PerfsUpdater.scala`、`modules/rating/src/main/java/glicko2/RatingCalculator.java`）。每一局各自更新一次，一局即一個 rating period。等級分低於 800 時以 800 計算（`math.max(800, perf.glicko.rating)`）
- **分層**：同一局同時更新全站、執白或執黑、棋種、速度類別（Bullet、Blitz、Slow）與 pool 的等級分；pool 對局不更新全站等級分。線上存入對局的 RatingDiff 是全站等級分的變動
- **改用 Glicko-2**：commit `bd6896a731`「convert all site to glicko ratings」、`6732496607`「migrate game collection with glicko ratings」、`d4d8b0e695`「fix rating calculator tau factor」，2013 年 12 月 17 日至 21 日
- **等級分下限**：commit `90efa51203`「glicko2 rating lower bound」，2014 年 4 月
- **等級分重構與遷移**：commit `505fb753f5`「perfs refactoring WIP」、`4a071d186b`「faster ratings migration」、`90208cbdd5`「rename Slow to Classical」、`6ce10d41a0`「make the standard perf a function of its sub perfs」，2014 年 7 月 26 日至 29 日
- **全部歷史對局的重算**：`modules/api/src/main/RatingFest.scala`（2014 年 7 月 27 日至 30 日，最後一版 commit `c89cb1eb33`）。先清空所有玩家的等級分，再依對局建立時間的先後逐局以當時的 `PerfsUpdater` 重算，並以 `resetGameRatings = true` 改寫每局存的對局前等級分與 RatingDiff。跳過未完成、從自訂局面開始與自己對自己的對局；有一方被標為作弊的對局取消計分、移除 RatingDiff。每次並行處理 4 局（`bulkSize = 4`），同一玩家相鄰的對局落在同一批時處理順序可能顛倒
- **重算所用的規則**：`modules/round/src/main/PerfsUpdater.scala`（commit `bfad573339`「tweak rating recalculator」）。Bullet、Blitz、Classical 只由標準棋對局更新，變體對局不影響；無限時（Unlimited）對局併入 Classical 計分。Glicko-2 的實作為 `org.goochjs.glicko2` 程式庫
- **通信棋獨立計分**：commit `804567893e`「support correspondence games in ratings and histories」，2014 年 12 月
- **灌分帳號不更新**：commit `e75a9d8608`「don't update boosters ratings」，2015 年 2 月
- **provisional 標示**：2014 年底的原始碼（`a798929cd3`，含頁面模板）沒有以 RD 標示新手的規則。`Glicko.provisional` 由 commit `363d62d603`「remember if game player perf was provisional」加入，門檻 RD ≥ 100，2015 年 6 月 1 日；隔日 commit `6fc785b387`「raise provisional glicko2 deviation threshold」提高為 RD ≥ 110（`modules/rating/src/main/Glicko.scala`）
- **研究期間之後**：commit `4673e347e8`「very slightly re-inject rating points (+1% for winner)」，2015 年 7 月

月檔的等級分與 RatingDiff 並非線上當時存入的全站值：整段期間（含 2013 年）本局等級分加 RatingDiff 大多等於同一玩家同一時間類別下一局的等級分（`scripts/explore/rating_consistency.py`），可見月檔的數值已改寫為各速度類別的值，即 `RatingFest` 重算的結果：2014 年 7 月底以前的數值來自這次重算，之後為線上即時計算。不相符的部分，候選原因有：2014 年 12 月以前無限時對局與 Classical 共用等級分，但月檔時間制為 `-`、檢查中歸入通信棋；重算時的並行處理使順序顛倒；作弊帳號的對局取消計分。重放的驗證見 `docs/glicko-replay.md`「不相符的來源」。

## Lichess 的配對機制

以下皆為 Lichess 原始碼（lila）與 API 規格的一手出處，已查證。

- **2014 年 arena 錦標賽的配對**：`modules/tournament/src/main/arena/PairingSystem.scala`（commit `f09a2e9893`）。空閒玩家超過 12 人時隨機配對，12 人以下時避免重複對手並優先配對等待最久者，不使用等級分。此前的 `modules/tournament/src/main/Pairing.scala` 邏輯相同。https://github.com/lichess-org/lila/blob/f09a2e9893/modules/tournament/src/main/arena/PairingSystem.scala
- **arena 改依等級分與排名配對**：commit `91155129bc`「pair arena players based on rating」、`7efa063e52`「use ranking in arena tournament pairing」、`355477ae1e`「let player rating slightly impact arena pairings」，皆在 2015 年 4 月 20 日至 23 日；5 月有 `d32490926f`、`9ec9e25558`、`d5fb74155d`、`4256ddaa49`、`4dec1bb29a` 等後續調整。為前瞻驗證樣本止於 2015 年 4 月 19 日的依據。https://github.com/lichess-org/lila/commit/91155129bc
- **2014 年的 pool 試行**：2014 年 6 月起的一系列 commit，含「pool games don't affect global rating」；2014-10-01 的「hide homepage pool buttons」停用入口。可在 lila 以 commit 搜尋 `pool` 查得
- **錦標賽的執子顏色**：commit `7032c7c355`「randomize piece colors in tournament pairings」在 2015 年 4 月，表示 2014 年錦標賽的顏色分配可能不是隨機的，具體規則待查。https://github.com/lichess-org/lila/commit/7032c7c355
- **大廳的分數範圍篩選**：commit `909f5a8e89`「fix lobby filter rating range」（2015-01-08），表示此前已有此功能。https://github.com/lichess-org/lila/commit/909f5a8e89
- **現行 quick pairing（pool）**：2016 年 11 月底至 12 月初的 `pool` 分支 commit
- **API 的配對來源欄**：`GameSource` 列出 `source` 的取值；以 ID 匯出對局的端點一次最多 300 個 ID，不需帳號。API 規格的 rate limiting 一節要求同一時間只發一個請求，收到 HTTP 429 時至少等一分鐘。官方資料庫的批量檔案（各棋種 rated 對局、轉播、puzzles、局面評估、開局）皆不含配對來源。https://github.com/lichess-org/api/blob/master/doc/specs/lichess-api.yaml 。https://github.com/lichess-org/api/blob/master/doc/specs/schemas/GameSource.yaml 、https://github.com/lichess-org/api/blob/master/doc/specs/tags/games/api-games-export-_ids.yaml

## 檢定與離散分析

除另有標示者外，以下出處未在本專案逐一查證。

- **Glicko 對一局結果的預測式**（Glickman，The Glicko system，已查證，已讀全文，含其範例的重算）。「Expected outcome of a game」一節以 $g(\sqrt{RD_i^2 + RD_j^2})$ 把分差往 0 收縮；更新公式中的期望得分則只含對手的 RD。Glicko-2 的說明（The Glicko-2 system）給出 $\mu = (r-1500)/173.7178$、$\phi = RD/173.7178$ 的換算。三個基準的 Glicko 形式所依據。http://www.glicko.net/glicko/glicko.pdf 、http://www.glicko.net/glicko/glicko2.pdf
- **相依資料下的概似比與 score 檢定**：Rotnitzky、Jewell（Biometrika 77(3)，1990）給出工作概似下概似比統計量的加權卡方分布；Boos（The American Statistician 46(4)，1992）的 generalized score test
- **非巢狀模型的比較**：Cox（Proceedings of the Fourth Berkeley Symposium，1961）的 tests of separate families；Vuong（Econometrica 57(2)，1989）的概似比檢定
- **類別資料分析**：Agresti，Categorical Data Analysis。Cochran–Mantel–Haenszel 檢定、Mantel–Haenszel 估計量、Breslow–Day 檢定、adjacent-categories logit、partial proportional odds、loglinear 模型、Cochran–Armitage 趨勢檢定
- **和局的配對比較模型**：Davidson（JASA 65，1970）
- **有序結果的配適度檢定**：Lipsitz、Fitzmaurice、Molenberghs（Applied Statistics 45，1996）
- **risk-adjusted CUSUM**：Steiner、Cook、Farewell、Treasure（Biostatistics 1(4)，2000）
- **觀察研究的敏感度分析**：Rosenbaum，Observational Studies
- **部分識別的界限**：Manski（American Economic Review 80(2)，1990）；Lee（Review of Economic Studies 76(3)，2009）
- **隨機化實驗中的共變數調整**：Freedman（Advances in Applied Mathematics 40(2)，2008）；Lin（Annals of Applied Statistics 7(1)，2013）。加入共變數只影響精度，不影響隨機化推論的有效性
- **truncation by death 與 principal stratification**：Zhang、Rubin（Journal of Educational and Behavioral Statistics 28(4)，2003）。survivor average causal effect 的界限
- **pattern-mixture 模型**：Little（JASA 88，1993）。依退出時點分組處理縱貫資料的退出
- **邊界上的概似比檢定**：Self、Liang（JASA 82，1987）。變異數成分為 0 時，虛無分布為卡方的混合
- **最適停止**：Chow、Robbins、Siegmund，Great Expectations: The Theory of Optimal Stopping（1971）。單調問題下 one-stage look-ahead 規則為最適
- **CUSUM 的最適性**：Lorden（Annals of Mathematical Statistics 42(6)，1971）；Moustakides（Annals of Statistics 14(4)，1986）
- **賭博系統與 optional stopping**：Doob，Stochastic Processes（1953）；Dubins、Savage，How to Gamble If You Must（1965）
- **dynamic treatment regime**：Murphy（Journal of the Royal Statistical Society Series B 65(2)，2003）。依逐期狀態決定處理的方案及其價值的估計
- **多條規則比較的 data snooping**：Hansen（Journal of Business & Economic Statistics 23(4)，2005）的 SPA 檢定；Romano、Wolf（Econometrica 73(4)，2005）的逐步檢定

