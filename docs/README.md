# 文件索引

要回答什麼問題，讀哪一份。一次任務通常只需讀一份。總覽與最終結論見根目錄 `README.md`。

## 研究設計與方法

| 文件 | 回答 |
|---|---|
| `report-focus.md` | 報告採用什麼核心目標？文獻、模型、驗證與用途邊界如何組成論證？ |
| `application-scope.md` | 共用分析基礎如何服務多種平台與玩家問題，哪些延伸尚待驗證？ |
| `entry-context.md` | 首局與開局間隔如何驗證額外資訊？如何比較適用範圍與覆蓋代價？ |
| `design.md` | 研究問題是什麼？識別從哪裡來？交付物？前瞻驗證怎麼做？類推到電競的界線？ |
| `methods.md` | 每個方法的模型、假設與診斷？對應哪些修課內容？基礎方法何時升級？哪些方法不採用、為什麼？ |
| `player-perspective.md` | 哪些分析能幫玩家判讀戰績、評估對局與理解停手證據？ |
| `sequence-diagnostics.md` | 固定局前機率如何對照連敗及累積偏差？人工參考與生成檢定差在哪裡？ |
| `player-alerts.md` | 如何比較連敗與對手調整後的近期表現提醒？ |
| `matching-policy.md` | 固定已配對玩家後，如何比較替代配法與代價？ |
| `temporal-validation.md` | 完整時間外驗證與跨遊戲評估的切分、候選及判準？ |
| `plan.md` | 各研究問題的主要做法與對照？結論之間如何依賴？範圍外不做哪些分析？待辦？ |

## 結果：`results/`

| 文件 | 回答 |
|---|---|
| `entry-context.md` | 首局／間隔資訊是否超過重新校準？排除高 RD 或支持不足的情境是否更可靠？ |
| `sequence-diagnostics.md` | 連敗與累積偏差的解讀如何受玩家權重與長視窗支持影響？ |
| `player-alerts.md` | 哪些近期表現提醒有下一局資訊？提醒對象與高頻玩家的影響是什麼？ |
| `matching-policy.md` | 平均、極端對局、批內最差及個人保護目標各有何取捨？誰的配對變差，哪些代價仍不可識別？ |
| `temporal-validation.md` | 主基準與選項都限訓練期時，近況和新手延伸在跨遊戲測試是否仍改善？ |
| `calibration.md` | 等級分給出的期望得分是否校準？偏差在哪裡？ |
| `expected-score-baselines.md` | 三個期望勝率的基準哪個好？相依資料下修正的概似比檢定水準如何？ |
| `novice-period.md` | 新手期如何界定？校準偏差在門檻處是否消失？ |
| `novice-clusters.md` | 新手的偏差收斂速度是否因人而異？ |
| `player-heterogeneity.md` | 平均沒有效果是否掩蓋了少數有效果的玩家？ |
| `random-pairing.md` | 2014 年 arena 的配對是否確與等級分無關？permutation test 該用哪些配對批次？ |
| `crossover.md` | 剛被隨機分到強對手之後，下一局的表現是否低於期望？總效果與直接效果有何不同？ |
| `continuation.md` | 被分到強對手是否改變續下？考慮續下的選擇後，結論能否推到所有玩家？ |
| `rating-filter.md` | 剛輸或剛贏之後，系統的期望得分是否偏？能否分開等級分調得不夠與 tilt？ |
| `state-adjusted-probs.md` | 依上一局修正後的勝、和、負機率比系統的期望準多少？保留哪些近況項？ |
| `history-depth.md` | 比上一局更早的對局能否讓勝率更準？有沒有持續數局的失常狀態？ |
| `folk-stop-rules.md` | 「輸了今天休息」這類民間停止規則有沒有用？停下之後隔天的第一局表現如何？ |
| `prospective-validation.md` | 2014 年的結論在沒看過的 2015 年資料上是否重現？ |
| `other-games.md` | 換一個遊戲與等級分系統（OGS 圍棋、AoE2），期望勝率的偏差、近況的修正與停止規則的結論是否仍成立？ |
| `streaks.md` | 連輸幾局之後是否差於期望？「系統製造連敗」需要操控才能解釋嗎？ |

### 附錄：`results/appendix/`

| 文件 | 回答 |
|---|---|
| `match-quality-leave.md` | 隨機配對下，匹配品質是否影響續下與離開？機制是什麼？ |
| `eomm-assumption.md` | 「給定結果，對手不影響離開」是否成立？等實力配對是否最好？ |
| `fixed-effects-churn.md` | 電競文獻常用的固定效應流失模型在 Lichess 上能否複製？自選對手與隨機配對下的估計差多少？ |

## 資料與前處理：`data/`

| 文件 | 回答 |
|---|---|
| `sessions.md` | 續下、離開、回訪如何定義？門檻怎麼決定？`player_games.parquet` 有哪些欄位？ |
| `glicko-replay.md` | 重放 Glicko-2 能否重現表頭的等級分？重放的 rating deviation 能否採用？ |
| `speed-class.md` | 月檔的時間類別與等級分所屬的速度類別有何不同？修正後結論是否改變？ |
| `data-fetching.md` | 資料怎麼下載？`SOURCE.md` 要記錄什麼？ |
| `other-data.md` | Lichess 以外還有哪些可取得的對局資料？ |

資料來源、欄位與授權見 `data/raw/lichess/SOURCE.md`。

## 背景：`background/`

| 文件 | 回答 |
|---|---|
| `literature.md` | 有哪些文獻？借用的方法在原論文中解決什麼？Lichess 的規則出自哪個 commit？ |
| `competitive-game-research-goals.md` | 使用同來源或相似雙人對戰資料的論文，各以什麼問題與交付為目標？ |
| `competitive-game-methods.md` | 最接近本研究的五篇論文用了哪些方法？原文在哪裡？哪些已做、可借用或不適合直接採用？ |
| `community-perception.md` | 電競玩家怎麼看匹配系統？開發商怎麼回應？ |
| `topic-selection.md` | 為什麼選這個題目？比較過哪些候選？資料能回答與回答不了什麼？ |

## 內部工作規範：`internal/`

| 文件 | 回答 |
|---|---|
| `computing.md` | 遇過哪些記憶體、速度與數值問題？各用什麼解法？ |
| `player-aggregation.md` | 先彙總到玩家再檢定，能否在哪些分析上取代 sandwich？ |
| `report-language.md` | 報告與簡報如何介紹新術語、資料特徵及方法文獻？ |
| `slides.md` | 簡報文字的體例，與投影片和口稿的分工 |

`internal/blog/` 是效能與工程問題的覆盤紀錄，依時間順序寫過程、走錯的路與實測耗時，是文件規範中「只寫現況」「不寫估計量」的例外；現行做法以 `computing.md` 為準。

| 文件 | 回答 |
|---|---|
| `blog/continuation-selection-profiling.md` | 怎麼用 py-spy、perf 找出熱點？哪些推論被實測推翻？有哪些坑？ |

術語見根目錄 `GLOSSARY.md`。文件不寫估計值，所有數值在 `outputs/tables/`、`outputs/figures/` 與腳本輸出。
