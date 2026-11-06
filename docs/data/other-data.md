# 其他對局資料

Lichess 以外、可取得原始資料且與等級分匹配相關的對局資料。用途是補足 Lichess 與電競之間的距離：Lichess 2014 年錦標賽提供隨機配對的因果識別，但不是電競式的排隊配對。各研究的出處見 `docs/background/literature.md`。

## 候選

| 遊戲 | 原始資料 | 規模與欄位 | 匹配機制 | 能否串出玩家序列 | 用過的研究 | 限制 |
|---|---|---|---|---|---|---|
| Age of Empires II DE（即時戰略） | 粉絲網站 AOE2 Complete Guide 的快照（`https://aoe2-guide.vercel.app/data`，`matches.parquet`、`players.parquet`），自述取自遊戲官方的社群 API；已下載至 `data/raw/aoe2/`。aoestats.io 的每週 dump 為另一來源，下載時整站無法連線 | 快照只涵蓋一個版本；所下載者為 2026-06-01 至 07-03，1v1 ranked 36 萬餘場（`data/raw/aoe2/SOURCE.md`）。含 `profile_id`、對局前等級分 `old_rating`、開局與結束時間、勝負 | 系統排隊自動配對，有 1v1 與團隊賽，Elo 類等級分 | 可以，但快照漏收部分場次，須以等級分是否接續篩選 | arXiv 2506.04475（EC '25）利用團隊的近似隨機分配估計隊友效應 | 維護者不明、未寫授權條款，使用者同意作外部效度的參考；新版本上線即重置；沒有 RD；更新規則未公開 |
| OGS 圍棋（online-go.com） | 官方 repo `online-go/goratings` 的 `data/ogs-data.db`（SQLite，git LFS），已下載至 `data/raw/ogs/` | 約 2.4 GB，3,081 萬局，2005 至 2023 年，2014 年起才有規模。欄位只有黑白方代碼、勝者、讓子數、貼目、每手時間、是否超時與結束時間（`data/raw/ogs/SOURCE.md`）；repo 授權為 MIT | Glicko-2，有讓子棋；配對方式有自動配對與公開挑戰 | 可以，玩家代碼與結束時間足以串出序列 | OGS 以此資料調整其等級分系統 | 沒有對局前等級分，須重放；只有結束時間，沒有開局時間；沒有配對來源，分不出自動配對與自選挑戰，也沒有 ranked 旗標；含機器人 |
| Chatbot Arena（大型語言模型對戰） | Hugging Face `lmarena-ai` 的資料集，如 `arena-human-preference-140k` | 每列有 `model_a`、`model_b`、`winner`（含 `tie`、`both_bad`）、`timestamp`。CC BY 4.0 | 系統抽樣哪兩個模型對戰 | 對象是模型而非玩家，沒有留存問題 | arXiv 2504.20879 指出閉源模型被抽中的頻率較高而使排名偏誤；arXiv 2501.17858 利用 Elo 機制操縱投票 | 與電競的玩家匹配不同 |
| Dota 2 | OpenDota 公開 API，不需金鑰 | `publicMatches` 有場次平均段位 `avg_rank_tier`、`lobby_type`、勝負 | 系統配對，隱藏 MMR | 不行：公開對局沒有玩家 ID，只有公開個人檔案的玩家可逐一查 | arXiv 上未找到匹配相關研究 | 只能做場次層級的分析 |
| CS:GO 職業賽 | arXiv 2410.02831 的 skillbench GitHub，資料爬自 HLTV | 2017 至 2022 年約 1 萬場職業賽 | 賽事抽籤，不是排隊配對 | 職業選手可串 | arXiv 2410.02831 比較 Elo、Glicko-2、TrueSkill | 不是匹配系統的資料 |

## 排除

- **星海 2**：SC2 Pulse 沒有公開的批量匯出；Battle.net API 須申請金鑰，且資料只保留一段時間
- **Chess.com**：公開 API 只能逐一查詢玩家的每月對局，沒有批量檔案
- **LoL**：Riot API 須金鑰，有速率限制與使用條款，未查到可直接下載的原始資料

## 評估

AoE2 最接近電競：系統依等級分排隊配對，有對局前等級分、玩家 ID 與時間戳。分差由系統決定、不是隨機的，只能作外部效度的檢驗，不能估計因果效果。取得的快照時間跨度只有約一個月，且序列有缺漏。

OGS 的規模大、授權清楚。讓子棋本身是以規則補償實力差距的匹配設計，為其他資料所無。自動配對與自選挑戰分不開，所以不能當作排隊配對的外部驗證，只能檢驗期望勝率的校準與近況，且須先重放等級分。

與 Lichess 的分工：Lichess 以隨機配對提供因果識別；OGS 與 AoE2 檢驗同樣的關係在其他配對機制下是否成立。

## 範圍

只做期望勝率的校準、加入近況後的預測、連勝連敗與停止規則的回測，不做因果識別（`docs/plan.md`「其他遊戲的外部驗證」，結論見 `docs/results/other-games.md`）。兩者都沒有和局；AoE2 沒有 rating deviation，新手以累計局數界定。
