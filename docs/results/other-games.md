# 其他遊戲的外部驗證

腳本：`scripts/ogs_replay.py`（OGS 等級分的重放）、`scripts/other_games_ogs.py`、`scripts/other_games_aoe2.py`（分析）。重放與 OGS 官方參考實作的比對：`scripts/explore/ogs_replay_matches_reference.py`。共用 module：`scripts/external.py`（主基準的重估、校準、近況的逐步加入、停止規則的判定，兩個遊戲共用）、`scripts/state_adjusted.py`（對數分數與 dyadic 標準誤）、`scripts/stop_rules.py`（停止規則的回測）、`scripts/runs.py`（連串的長度）。表在 `outputs/tables/other_games_ogs_*.csv`、`outputs/tables/other_games_aoe2_*.csv`，Brier 分數的分解在 `outputs/tables/forecast_decomposition.csv`（`scripts/forecast_decomposition.py`）。規格見 `docs/plan.md`「其他遊戲的外部驗證」，資料見 `data/raw/ogs/SOURCE.md`、`data/raw/aoe2/SOURCE.md`。本文件只寫定性結論，數值一律見對應的輸出檔。


基準、近況選項與最終測試分開的共同流程見 `docs/results/temporal-validation.md`。該分析排除 OGS 自我對局、對 AoE2 加入上一局已完成的檢查，並明列 OGS 缺少開局時間的限制；本文件的既有輸出保留。

## 問題

Lichess 的結論是在一個遊戲、一種配對機制上得到的。換一個遊戲與等級分系統，三件事是否仍成立：系統的期望勝率不校準、近況能在樣本外修正它、民間停止規則沒有用。這裡只作外部效度的檢驗，不作因果識別：兩個遊戲的對手都不是隨機指派的。

兩份資料的配對機制不同。OGS 混合自動配對與自選挑戰；AoE2 的 1v1 ranked 由系統依等級分排隊配對，是三份資料中最接近電競的一份。

## 資料與做法

- **OGS 圍棋**：官方 `online-go/goratings` 的對局資料庫。沒有對局前的等級分，依 OGS 公開的分析程式重放 Glicko-2，前 30 萬局與官方參考實作的等級分、RD 與期望勝率逐局相同。重放是把 OGS 現行的演算法回溯套到歷史對局上，不是當年線上的版本。同一秒結束的對局另以對局代碼排序；極端的讓子使期望勝率捨入成 0 或 1，參考實作在爆冷時溢位中斷，這類對局記為不計分
- **OGS 的樣本**：選與 Lichess arena 最接近的條件：19 路、不讓子、非通訊棋、雙方都不是機器人。2021、2022 年估計，2023 年評估。速度類別依 OGS 的分法分為 blitz 與 live（`other_games_ogs_sample.csv`）。新手為任一方 RD 不低於 160，即 OGS 程式中 provisional 的門檻
- **AoE2**：粉絲網站 aoe2-guide 的快照，網站自述取自遊戲官方的社群 API；維護者不明、未寫授權條款。只有 2026 年 6 月 1 日至 7 月 3 日，6 月 24 日之前估計、之後評估（`other_games_aoe2_sample.csv`）。系統的期望勝率以雙方的對局前等級分代入 Elo 公式，AoE2 實際的更新規則未公開。沒有 RD，新手為局前累計局數未滿 30
- **AoE2 的序列有缺漏**：快照漏收部分場次。「上一局」須上一場的 `new_rating` 等於本場的 `old_rating`，否則視為本日無上一局；兩者在本場開始前都已知。連勝連敗與停止規則只用一天內每一場都接得上的天
- **上一局與一天**：兩個遊戲都沒有錦標賽，「上一局」取同一位玩家同一天的上一局；一天以相隔超過 12 小時切開，與 Lichess 的停止規則同一套切法。OGS 只有結束時間，間隔以結束時間計
- **主基準**：Lichess 主基準去掉和局項的 binary logistic（平均等級分及其平方、分差及其與平均等級分的交互作用、RD 收縮項；AoE2 沒有 RD 項），在各自的估計段重估
- 校準、依狀態調整的勝率與停止規則的做法同 Lichess；停止規則以評估段開始的天回測，邊界同為 10 等級分

## 結論

### 系統的期望勝率：OGS 過度自信，AoE2 沒有

- OGS 的期望勝率在兩個速度類別都過度自信：校準斜率明顯低於 1，即高估強方的優勢，與 Lichess 的 Elo 形式相同（`other_games_ogs_calibration.csv`）。重估的主基準在評估段接近校準，樣本外對數分數較好，改善集中在新手
- AoE2 的 Elo 期望沒有過度自信，校準斜率略高於 1（`other_games_aoe2_calibration.csv`）；重估的主基準與它的對數分數幾乎沒有差別，新手組甚至略差（`other_games_aoe2_prediction.csv`）
- 差別可能來自配對機制：AoE2 依等級分排隊配對，分差的離散程度遠小於 OGS（兩份樣本表的 `rating_diff_sd`），高估強方的偏差在分差小時幾乎看不出來。這是推測，資料沒有分差大的 AoE2 對局可供檢驗
- Brier 分數的分解（`forecast_decomposition.csv`）：OGS 從系統到主基準的改善主要來自可靠度，與 Lichess 相同；AoE2 的系統已近乎校準，可靠度沒有可改善的空間

### 近況在樣本外修正期望勝率，三個遊戲都成立

- 雙方的上一局結果與上一局等級分變動依序加入，OGS 的兩個速度類別與 AoE2 都保留兩項，樣本外對數分數的 dyadic 區間不含 0（`other_games_ogs_steps.csv`、`other_games_aoe2_steps.csv`）
- 等級分變動的係數為正：剛漲分的一方好於期望，即等級分調得不夠，與 Lichess 相同（`other_games_ogs_coef.csv`、`other_games_aoe2_coef.csv`）。給定等級分變動之後，上一局結果本身的係數為負，與等級分變動高度相關，不單獨解讀
- 改善集中在新手。AoE2 的老手組，修正後的勝率反而比主基準差；一個可能的原因是近況項的係數由新手主導，套在老手身上過度修正，本分析沒有分組估計來檢驗。報告中修正後的勝率依新手與否分開陳述
- 分解上，近況主要增加解析度，即帶來等級分之外的分辨資訊，三個遊戲相同

### 連勝連敗

- OGS 連輸四局之後好於期望，方向與 Lichess 相反；AoE2 與 Lichess 相同，連輸之後差於期望、連勝之後好於期望，改用依狀態調整的勝率之後區間含 0，見 `docs/results/streaks.md`

### 停止規則：沒有支持「輸了就停」的證據

- OGS blitz 的兩條規則，被刪去的局都明顯好於保留的局，區間整段在正的邊界之外：照規則停下反而吃虧（`other_games_ogs_stop_rules.csv`）。方向與 Lichess Bullet 的前瞻驗證相同
- OGS live 的「一路下到第一次輸」，刪去的局略差於保留的局，Holm 校正後顯著，但區間整段落在 ±10 等級分之內：可偵測，但小到對玩家無感。「連輸兩局才停」的方向相反，同樣落在邊界內
- AoE2 的兩條規則，區間都完整落在 ±10 等級分之內，判定為效果小到可忽略（`other_games_aoe2_stop_rules.csv`）。這是三份資料中唯一兩條規則都達到「不必停」判定的
- 隔天重置：OGS blitz 的隔天第一局差於被刪去的局，與 Lichess 相同；OGS live 沒有這個型態；AoE2 的隔天第一局反而好於被刪去的局（`other_games_ogs_stop_rules_reset.csv`、`other_games_aoe2_stop_rules_reset.csv`）

## 解讀的界線

- OGS 的配對混合了自動配對與自選挑戰，資料分不開；系統的期望勝率不校準可能部分來自自選對手的行為，不全是等級分系統本身
- OGS 重放的是現行演算法，不是當年線上的版本；2021 至 2023 年的線上系統與重放可能不同
- AoE2 只有約一個月、來源的維護者不明且未寫授權條款；序列有缺漏，只用接得上的序列，被排除的天若與表現有關，停止規則與連串的結論會偏
- 一天以相隔時間切分；OGS 以結束時間計，live 的一局可長達數十分鐘，同一天內偶有同時進行的對局
- 這是外部效度的檢驗：方向一致支持 Lichess 的結論不只是 Lichess 的特例，但不能在這兩個遊戲上估計因果效果

## 對報告的意涵

- **期望勝率**：「等級分調得不夠」在三個遊戲都成立；「高估強方的優勢」在分差範圍大的 Lichess 與 OGS 成立，在依等級分排隊配對的 AoE2 看不出來
- **修正後的勝率**：加入雙方近況改善樣本外預測，在三個遊戲都成立，改善集中在新手；AoE2 的老手組反而變差，類推到電競時須分開陳述
- **停止規則的建議**：三個遊戲都沒有支持「輸了就停」的結果；最接近電競的 AoE2 兩條規則都小到可忽略，OGS 的 blitz 與 Lichess Bullet 的前瞻驗證方向是照規則停下反而吃虧
