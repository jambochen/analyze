# 文件索引

要回答什麼問題，讀哪一份。一次任務通常只需讀一份。

| 文件 | 回答 |
|---|---|
| `design.md` | 研究問題是什麼？交付物？為什麼選這個題目？哪些方向不做？ |
| `plan.md` | 各方法要解決什麼問題？借用的方法在原論文與本研究中各解決什麼？方法之間的依賴與優先級？目前的進度與待辦？ |
| `calibration.md` | 等級分給出的期望得分是否校準？後續的匹配品質與基準該用什麼？ |
| `random-pairing.md` | 2014 年 arena 的配對是否確與等級分無關？randomization inference 該用哪些配對批次？ |
| `sessions.md` | 續下、離開、回訪如何定義？門檻怎麼決定？`player_games.parquet` 有哪些欄位？ |
| `speed-class.md` | 月檔的時間類別與等級分所屬的速度類別有何不同？修正後結論是否改變？ |
| `glicko-replay.md` | 重放 Glicko-2 能否重現表頭的等級分？重放的 rating deviation 能否採用？速度類別的規則何時生效？ |
| `match-quality-leave.md` | 隨機配對下，匹配品質是否影響續下與離開？效果的機制是什麼？ |
| `eomm-assumption.md` | EOMM 的關鍵假設（給定結果，對手不影響離開）是否成立？等實力配對是否最好？ |
| `fixed-effects-churn.md` | Heliyon 2024 的固定效應模型在 Lichess 上能否複製？自選對手與隨機配對下的估計差多少？ |
| `topic-options.md` | 題目還有哪些候選？資料能回答與回答不了什麼？生統、實驗設計、工統與可靠度有哪些方法可借用？ |
| `story-methods.md` | 平台端配對與玩家端停止規則兩條故事線，各需要哪些方法？文獻與完成狀態？兩條差在哪？ |
| `directions.md` | 本資料可做的匹配相關方向，依完整性、資料、統計模型、與電競的貼近程度如何排名？ |
| `other-data.md` | Lichess 以外還有哪些可取得原始資料的對局資料？哪一份適合作為第二份資料？ |
| `literature.md` | 這個問題有哪些文獻？哪些出處已查證？ |
| `glossary.md` | 這個概念用哪個術語？哪些詞不用？ |
| `report-language.md` | 報告與簡報如何介紹新術語、資料特徵及方法文獻？ |
| `slides.md` | 簡報上的文字怎麼寫？標題、條列、文言成分、標點的體例。 |

資料來源、欄位與授權見 `data/raw/lichess/SOURCE.md`。

寫作規範見 `CLAUDE.md`「文件規範」：不寫數值、只寫現況、不寫階段編號。
所有數值在 `outputs/tables/`、`outputs/figures/` 與腳本輸出。
