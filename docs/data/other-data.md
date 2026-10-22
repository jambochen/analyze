# 其他對局資料

Lichess 以外、可取得原始資料且與等級分匹配相關的對局資料。用途是補足 Lichess 與電競之間的距離：Lichess 2014 年錦標賽提供隨機配對的因果識別，但不是電競式的排隊配對。各研究的出處見 `docs/background/literature.md`。

## 候選

| 遊戲 | 原始資料 | 規模與欄位 | 匹配機制 | 能否串出玩家序列 | 用過的研究 | 限制 |
|---|---|---|---|---|---|---|
| Age of Empires II DE（即時戰略） | aoestats.io 每週發布的 parquet 檔（`matches.parquet`、`players.parquet`），`https://aoestats.io/api/db_dumps` 列出全部檔案 | 有資料的檔案涵蓋 2022-08 至 2026-02，合計約 3,069 萬場。含 `profile_id`、對局前等級分 `old_rating`、對局後 `new_rating`、`started_timestamp`、`duration`、`leaderboard`、`map`、勝負 | 系統排隊自動配對，有 1v1 與團隊賽，Elo 類等級分 | 可以 | arXiv 2506.04475（EC '25）利用團隊的近似隨機分配估計隊友效應 | 第三方整理的資料，授權未寫明，須先確認使用條款；2026-02 之後的檔案為空；排隊配對規則（如分差上限如何隨等待時間放寬）未公開 |
| OGS 圍棋（online-go.com） | 官方 repo `online-go/goratings` 的 `data/ogs-data.db`（SQLite，git LFS） | 約 2.4 GB；README 稱有 1,200 萬局 ranked 對局。repo 授權為 MIT | Glicko-2，有讓子棋；配對方式有自動配對與公開挑戰 | 應可以，欄位待下載確認 | OGS 以此資料調整其等級分系統 | 欄位與時間範圍未確認；自動配對與自選挑戰能否區分未確認 |
| Chatbot Arena（大型語言模型對戰） | Hugging Face `lmarena-ai` 的資料集，如 `arena-human-preference-140k` | 每列有 `model_a`、`model_b`、`winner`（含 `tie`、`both_bad`）、`timestamp`。CC BY 4.0 | 系統抽樣哪兩個模型對戰 | 對象是模型而非玩家，沒有留存問題 | arXiv 2504.20879 指出閉源模型被抽中的頻率較高而使排名偏誤；arXiv 2501.17858 利用 Elo 機制操縱投票 | 與電競的玩家匹配不同 |
| Dota 2 | OpenDota 公開 API，不需金鑰 | `publicMatches` 有場次平均段位 `avg_rank_tier`、`lobby_type`、勝負 | 系統配對，隱藏 MMR | 不行：公開對局沒有玩家 ID，只有公開個人檔案的玩家可逐一查 | arXiv 上未找到匹配相關研究 | 只能做場次層級的分析 |
| CS:GO 職業賽 | arXiv 2410.02831 的 skillbench GitHub，資料爬自 HLTV | 2017 至 2022 年約 1 萬場職業賽 | 賽事抽籤，不是排隊配對 | 職業選手可串 | arXiv 2410.02831 比較 Elo、Glicko-2、TrueSkill | 不是匹配系統的資料 |

## 排除

- **星海 2**：SC2 Pulse 沒有公開的批量匯出；Battle.net API 須申請金鑰，且資料只保留一段時間
- **Chess.com**：公開 API 只能逐一查詢玩家的每月對局，沒有批量檔案
- **LoL**：Riot API 須金鑰，有速率限制與使用條款，未查到可直接下載的原始資料

## 評估

AoE2 最適合作為第二份資料：它是電競式的系統排隊配對，有對局前等級分、玩家 ID 與時間戳，連續對局、留存與匹配品質都能做，團隊賽也能接到 TrueSkill 類的問題。

與 Lichess 的分工：Lichess 以隨機配對提供因果識別；AoE2 檢驗同樣的關係在排隊配對中是否成立。AoE2 的分差由系統依等級分決定、不是隨機的，只能作外部效度的檢驗，不能直接估計因果效果。

採用 AoE2 前須先確認 aoestats 的使用條款，並查清 AoE2 排隊配對的規則。

OGS 次之。讓子棋本身是以規則補償實力差距的匹配設計，為其他資料所無；須先下載確認欄位。
