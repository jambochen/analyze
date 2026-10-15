# 文獻

標「已查證」者已在出版者頁面或 arXiv 核對標題與出處；其餘僅見於搜尋結果或作者頁面，引用前須核對。

## 連勝連敗與 tilt

- **Streaks and Coping: Decoding Player Performance in League of Legends Using Big Data from Top Players' Matches**（FDG 2024）。以頂尖玩家的對戰序列分析連勝連敗後的表現與換角、休息的關係。與本題主問題最接近。https://dl.acm.org/doi/fullHtml/10.1145/3665463.3678787
- **Understanding Tilt in Esports: A Study on Young League of Legends Players**（CHI 2021）。高中電競選手的問卷與訪談，界定 tilt 的觸發因素，屬質性研究。https://dl.acm.org/doi/pdf/10.1145/3411764.3445143
- **Playing with Streakiness in Online Games: How Players Perceive and React to Winning and Losing Streaks in League of Legends**（CHI 2018）。https://dl.acm.org/doi/pdf/10.1145/3173574.3174152
- **Investigating Experiential Effects in Online Chess using a Hierarchical Bayesian Analysis**（Gee、Seese、Curley、Ward，arXiv 2503.21713，已查證，已讀全文）。以 Lichess 資料檢驗贏輸效應：近幾局的勝率是否影響下一局。連續對局以開局間隔切分，Bullet 與 Blitz 門檻不同；模型為階層貝氏 logistic 迴歸，控制分差並加玩家層隨機效應。結論為沒有全體一致的效應，少數玩家個別有。資料是先選定玩家、再爬取其全部對局；論文稱資料與爬取程式碼放在作者的 GitHub，未查得網址。與附屬問題最接近；本題的差異在於全體玩家而非抽樣、和局不丟棄、處理續下的選擇偏誤、並看留存。https://arxiv.org/abs/2503.21713
- **Quantifying human performance in chess**（Chowdhary、Iacopini、Battiston，arXiv 2207.07780，已查證，已讀全文）。Lichess 2013 至 2016 年全部對局。以打亂時間順序為虛無模型，發現連勝連敗長於隨機，新手的連勝更長；另發現高手專精少數開局，但連續兩局間比新手更常換開局。虛無模型只保留勝和負總數，忽略等級分的變化：進步中的玩家等級分追不上實力，本來就會連勝。本題以對局前等級分的期望得分為基準，修正的正是這一點。資料取自官方資料庫 database.lichess.org 的月檔，與本研究同一來源。https://arxiv.org/abs/2207.07780
- **Fast Algorithm for Calculating Probability of Chess Winning Streaks**（arXiv 2503.11669，已查證）。給定每局勝和負機率，計算各種連勝的機率。https://arxiv.org/abs/2503.11669
- **Kramnik vs Nakamura: A Chess Scandal**（arXiv 2409.06739，已查證）。以線上快棋連勝爭議為例，說明解讀連勝時的檢察官謬誤。https://arxiv.org/abs/2409.06739

## 匹配與流失

- **EOMM: An Engagement Optimized Matchmaking Framework**（EA）。以投入度而非勝率最佳化匹配。https://arxiv.org/abs/1702.06820
- **EnMatch**：以玩家投入度為目標的匹配（網易伏羲，AAAI 2024）。https://ojs.aaai.org/index.php/AAAI/article/view/28760
- **A Data-Driven Decision Support Framework for Player Churn Analysis in Online Games**（網易伏羲，KDD 2023）。資料為網易內部，不公開。https://dl.acm.org/doi/10.1145/3580305.3599759
- 網易伏羲的 OptMatch（KDD 2020）與全域匹配（KDD 2021）：見作者頁面 https://wu-runze.github.io/collection-archive/
- **Match experiences affect interest: Impacts of matchmaking and performance on churn in a competitive game**（Heliyon 2024）。https://pmc.ncbi.nlm.nih.gov/articles/PMC10839887/
- **The Role of Skill in Matchmaking**（Activision 白皮書）。含實力相當下連敗發生率的機率基準，以及放寬技術匹配的實驗。https://www.activision.com/cdn/research/CallofDuty_Matchmaking_Series_2.pdf
- **Skill-Based Matchmaking for Competitive Two-Player Games**（Yuksel，ACM TOG 2024）。https://dl.acm.org/doi/10.1145/3651303
- **Opponent Indifference in Rating Systems: A Theoretical Case for Sonas**（arXiv 2209.03950，已查證）。玩家能看到對手分數並挑選或中止時，等級分系統會誘使玩家策略性選對手。支撐「大廳自選對手混淆分差效果」的論點。https://arxiv.org/abs/2209.03950
- **Mean-Field Analysis and Optimal Control of a Dynamic Rating and Matchmaking System**（arXiv 2512.22038，已查證）。以平均場理論分析等級分準確度與配對強度的關係。https://arxiv.org/abs/2512.22038
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
- **Who Benefits from AI? Self-Selection, Skill Gap, and the Hidden Costs of AI Feedback**（arXiv 2409.18660，已查證）。線上棋平台上，越有動機的玩家越常使用 AI 回饋，造成效果的假象。本資料做不了同一題，但自我選擇造成假象的論證結構與續下的選擇偏誤相同。https://arxiv.org/abs/2409.18660

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
- Glicko-2 的原始定義（Glickman）與 Lichess 的等級分實作：待補

## Lichess 的配對機制

以下皆為 Lichess 原始碼（lila）與 API 規格的一手出處，已查證。

- **2014 年 arena 錦標賽的配對**：`modules/tournament/src/main/arena/PairingSystem.scala`（commit `f09a2e9893`）。空閒玩家超過 12 人時隨機配對，12 人以下時避免重複對手並優先配對等待最久者，不使用等級分。此前的 `modules/tournament/src/main/Pairing.scala` 邏輯相同。https://github.com/lichess-org/lila/blob/f09a2e9893/modules/tournament/src/main/arena/PairingSystem.scala
- **arena 改依等級分與排名配對**：commit `91155129bc`「pair arena players based on rating」、`7efa063e52`「use ranking in arena tournament pairing」、`355477ae1e`「let player rating slightly impact arena pairings」，皆在 2015 年 4 月 20 日至 23 日；5 月有 `d32490926f`、`9ec9e25558`、`d5fb74155d`、`4256ddaa49`、`4dec1bb29a` 等後續調整。為識別二的改制依據。https://github.com/lichess-org/lila/commit/91155129bc
- **2014 年的 pool 試行**：2014 年 6 月起的一系列 commit，含「pool games don't affect global rating」；2014-10-01 的「hide homepage pool buttons」停用入口。可在 lila 以 commit 搜尋 `pool` 查得
- **錦標賽的執子顏色**：commit `7032c7c355`「randomize piece colors in tournament pairings」在 2015 年 4 月，表示 2014 年錦標賽的顏色分配可能不是隨機的，具體規則待查。https://github.com/lichess-org/lila/commit/7032c7c355
- **大廳的分數範圍篩選**：commit `909f5a8e89`「fix lobby filter rating range」（2015-01-08），表示此前已有此功能。https://github.com/lichess-org/lila/commit/909f5a8e89
- **現行 quick pairing（pool）**：2016 年 11 月底至 12 月初的 `pool` 分支 commit
- **API 的配對來源欄**：`GameSource` 列出 `source` 的取值；以 ID 匯出對局的端點一次最多 300 個 ID，不需帳號。API 規格的 rate limiting 一節要求同一時間只發一個請求，收到 HTTP 429 時至少等一分鐘。官方資料庫的批量檔案（各棋種 rated 對局、轉播、puzzles、局面評估、開局）皆不含配對來源。https://github.com/lichess-org/api/blob/master/doc/specs/lichess-api.yaml 。https://github.com/lichess-org/api/blob/master/doc/specs/schemas/GameSource.yaml 、https://github.com/lichess-org/api/blob/master/doc/specs/tags/games/api-games-export-_ids.yaml

## 本資料可做的其他方向

不屬於本題的研究問題，記錄供取捨。皆已在 arXiv 查證。

- **開局行為**：開局熱門程度符合 Zipf 定律（0704.2711）；新變化的出現率隨前序頻率衰減（1309.0336）；以線上平台資料建立開局相似度網路並預測玩家將改下的開局（2206.14312）。本資料有 `ECO` 與 `Opening`
- **對手網路**：線上與實體棋賽的對手網路結構（1708.06694）；以對手網路的中心性預測排名，資料混用 Chess.com 與 Dota 2（2501.19220）。本資料由 `White`、`Black` 即可建網路
- **賽制公平性**：瑞士制在奇數輪時多一局白棋的玩家得分較高（2410.19333）。2014 年已有瑞士制，API 的 `source` 可區分 arena 與 swiss
