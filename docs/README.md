# 文件索引

要回答什麼問題，讀哪一份。一次任務通常只需讀一份。

## 研究設計與計劃

| 文件 | 回答 |
|---|---|
| `design.md` | 研究問題是什麼？識別從哪裡來？交付物？前瞻驗證怎麼做？類推到電競的界線？ |
| `methods.md` | 借用的方法在原論文與本研究中各解決什麼？每個方法的模型、假設與診斷？基礎方法何時升級？ |
| `plan.md` | 每個研究問題用哪些方法、各屬哪一級定位？結論之間如何依賴？已得出的結論改變了哪些走向？目前的進度與待辦？ |
| `topic-selection.md` | 為什麼選這個題目？定題的原則？比較過哪些候選、各自與 demo 的可行性？資料能回答與回答不了什麼？ |

## 已完成的分析

| 文件 | 回答 |
|---|---|
| `calibration.md` | 等級分給出的期望得分是否校準？後續的匹配品質與基準該用什麼？ |
| `random-pairing.md` | 2014 年 arena 的配對是否確與等級分無關？randomization inference 該用哪些配對批次？ |
| `sessions.md` | 續下、離開、回訪如何定義？門檻怎麼決定？`player_games.parquet` 有哪些欄位？ |
| `speed-class.md` | 月檔的時間類別與等級分所屬的速度類別有何不同？修正後結論是否改變？ |
| `glicko-replay.md` | 重放 Glicko-2 能否重現表頭的等級分？重放的 rating deviation 能否採用？速度類別的規則何時生效？ |
| `expected-score-baselines.md` | 三個期望勝率的基準哪個好？修正的概似比檢定在相依資料下的檢定水準如何？ |
| `novice-period.md` | 新手期如何界定？Lichess 的 provisional 門檻從何而來？與累計局數的代理差多少？校準偏差在門檻處是否消失？ |
| `crossover.md` | 剛被隨機分到強對手之後，下一局的表現是否低於期望？總效果與直接效果有何不同？ |
| `match-quality-leave.md` | 隨機配對下，匹配品質是否影響續下與離開？效果的機制是什麼？（附錄） |
| `eomm-assumption.md` | EOMM 的關鍵假設（給定結果，對手不影響離開）是否成立？等實力配對是否最好？（附錄） |
| `fixed-effects-churn.md` | Heliyon 2024 的固定效應模型在 Lichess 上能否複製？自選對手與隨機配對下的估計差多少？ |

## 資料與文獻

| 文件 | 回答 |
|---|---|
| `literature.md` | 這個問題有哪些文獻？Lichess 的等級分與配對規則出自哪個 commit？哪些出處已查證？ |
| `community-perception.md` | 電競玩家怎麼看匹配系統（連勝連敗、EOMM、MMR）？開發商怎麼回應？出處在哪？ |
| `other-data.md` | Lichess 以外還有哪些可取得原始資料的對局資料？哪一份適合檢驗外部效度？ |

資料來源、欄位與授權見 `data/raw/lichess/SOURCE.md`。

## 運算

| 文件 | 回答 |
|---|---|
| `computing.md` | 遇過哪些記憶體、速度與數值問題？成因是什麼？各用什麼解法？執行重分析的原則？ |

## 寫作

| 文件 | 回答 |
|---|---|
| `report-language.md` | 報告與簡報如何介紹新術語、資料特徵及方法文獻？ |
| `slides.md` | 簡報上的文字怎麼寫？標題、條列、文言成分、標點的體例。 |

術語的定義與統計術語的用法見根目錄 `GLOSSARY.md`。寫作規範見 `CLAUDE.md`「文件規範」：不寫數值、只寫現況、不寫階段編號。所有數值在 `outputs/tables/`、`outputs/figures/` 與腳本輸出。
