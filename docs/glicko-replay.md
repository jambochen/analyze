# Glicko-2 重放：重現表頭的對局前等級分

腳本：`scripts/glicko_replay.py`（全資料重放與比對）、`scripts/glicko_rust_matches_python.py`（Rust 與 Python 一致性）。共用的 Python 參考實作與輸入整理在 `scripts/glicko.py`，Rust 版在 `rust/glicko_replay/`（Python 模組名 `glicko_rs`）。速度類別規則上線時間的探索在 `scripts/explore/speed_boundaries.py`。表在 `outputs/tables/glicko_replay_*.csv`，圖在 `outputs/figures/glicko_replay_*.png`，逐局結果在 `data/processed/glicko_replay.parquet`。本文件只寫定性結論，數值一律見對應的輸出檔。

## 問題

表頭只有對局前等級分，沒有 rating deviation（RD）與 volatility。從對局史逐局重放 Lichess 的 Glicko-2，若能重現表頭的等級分，就同時得到每局雙方的對局前 RD 與 volatility，供 Glicko-2 形式的期望得分、新手期（provisional）的界定，以及分解校準斜率小於 1 的來源（`docs/calibration.md`）。

## 移植的規則

規則一律取自 lila 原始碼，以 `gh api repos/lichess-org/lila/contents/<path>?ref=<sha>` 讀取。

| 規則 | 內容 | 出處 |
|---|---|---|
| Glicko-2 計算 | `org.goochjs.glicko2`：初始值 1500、RD 350、volatility 0.06，τ = 0.75；volatility 以 Illinois 法求解，收斂門檻 0.000001、上限 3000 次，判斷式用嚴格的 `fC * fB < 0`；不收斂時拋出例外，雙方保留輸入值 | `modules/rating/src/main/java/glicko2/RatingCalculator.java`、`Rating.java`、`RatingPeriodResults.java`、`Result.java` @ `bfad573339` |
| rating period | 每局一個 rating period，雙方都有結果，閒置期間不增加 RD | `PerfsUpdater.scala` 的 `updateRatings` @ `bfad573339` |
| 等級分下限 | 計算的輸入取 `max(800, rating)`，存回的是計算結果本身，可低於 800 | `modules/rating/src/main/Perf.scala` 的 `toRating` @ `bfad573339` |
| 存入對局的整數 | `Glicko.intRating` 為 `rating.toInt`，向零截斷，不是四捨五入 | `modules/rating/src/main/Glicko.scala` @ `bfad573339` |
| 哪些對局更新 | rated、已完成、至少兩步（`accountable`）、雙方都不是作弊帳號；2015 年 2 月起灌分帳號也不更新 | `PerfsUpdater.scala` @ `bfad573339`、`e75a9d8608`；`Game.scala` @ `bfad573339` |
| 速度類別 | Bullet、Blitz、Classical 只由標準棋更新；`estimateTotalTime` 為底秒加每步加秒乘 30，小於 180 秒為 Bullet、小於 480 秒為 Blitz，其餘含無鐘對局併入 Classical | `PerfsUpdater.scala` @ `bfad573339`；scalachess `Speed.scala`、`Clock.scala` @ `1ceba9320d` |
| 全部歷史重算 | RatingFest 依建立時間逐局重算全部 rated 對局，改寫對局前等級分與 RatingDiff；跳過未完成、自訂局面、自己對自己；作弊帳號的對局改為非 rated（不在 rated 資料中） | `modules/api/src/main/RatingFest.scala` @ `c89cb1eb33`；`GameRepo.unrate` @ `c89cb1eb33` |
| 線上期間的表頭 | 對局建立時寫入當時的 `perf.intRating`，結束時才更新等級分並寫入 RatingDiff | `modules/game/src/main/Player.scala` 的 `withUser` @ `c89cb1eb33` |
| 通信棋獨立計分 | 有 `daysPerTurn` 的對局計入通信棋 | `PerfsUpdater.scala` @ `804567893e` |
| 無限時併入通信棋 | 無鐘對局一律為通信棋，Classical 的上限延到 21599 秒 | `PerfsUpdater.scala` @ `9938ca688a`、`e6321758a9`；scalachess `Speed.scala` @ `0cefcae7c8` |
| 每步加秒改乘 40 | `estimateTotalTime` 改為底秒加每步加秒乘 40 | scalachess `1fc3f0d44a`；lila 子模組更新 `395564dcf8` |

2014 年 7 月底至 2015 年 6 月 `PerfsUpdater.scala` 的其他變更（三棋種、antichess、atomic、horde、pool 移除、開局錦標賽）都只涉及變體棋或 pool，不影響標準棋的速度類別。2014 年 6 月至 10 月的 pool 對局，表頭存的是 pool 等級分而非速度類別的等級分（`PerfPicker.main` 對 pool 對局回傳 pool 的 perf），但速度類別仍照常更新；資料中無法辨認哪些是 pool 對局。

月檔的 `Event` 欄的時間類別是以每步加秒乘 40 的公式事後標的，與 2015 年 4 月以前實際計分所用的類別不同（例如 300+5 在當時計入 Blitz）。重放的速度類別一律依上表由 `base_seconds`、`increment_seconds` 與建立時間重新判定，不用資料中的 `time_class`。

### 規則變更的實際上線時間

commit 時間只是上線時間的下限。以只看表頭的等級分鏈判斷（`scripts/explore/speed_boundaries.py`）：一局的對局前等級分應接在同一帳號「計分所用的類別」上一局的等級分加 RatingDiff 之後，對兩種規則下類別不同的對局，看它接上哪一條鏈。

- 每步加秒改乘 40：commit 之後約十餘小時，表頭才從接上舊類別轉為接上新類別，採用推定的上線時間（`glicko.T_EST40`）。以 commit 時間為準的變體相符較低（`glicko_replay_variants.csv`、`glicko_replay_daily.csv`）
- 通信棋獨立計分：無鐘對局的表頭在 commit 當日起改接通信棋的新鏈（首局 1500）。此後到無限時併入通信棋之前，資料無法區分有 `daysPerTurn` 的通信棋與無鐘的無限時對局（時間制都記為 `-`），故一律視為通信棋；「無限時對局維持 Classical 到第二次 commit」的變體相符較低。上線時間取 commit 時間或當日零時，差異可忽略

## 實作與一致性驗證

Python 參考實作（`scripts/glicko.py`）逐行照 Java 原始碼移植，Rust 版（`rust/glicko_replay/src/lib.rs`）逐行對應 Python 版。兩者都不改變浮點運算的順序；Java 的 `Math.pow(x, 2)` 以 `x * x` 實作。Rust 與 Python 都經 glibc 的 `exp`、`log`，以 native 指令集編譯後也不產生 FMA，所以可以要求逐位元一致。

驗證子集取 2013 年第一季的全部對局（404,289 局）。對手的狀態互相依賴，子集必須是某時點以前的全部對局，不能抽樣。比對純重放與以表頭校正的重放兩種模式下每局雙方的 rating、RD、volatility 與旗標，以及三個輔助函式（多組參數的平行比對、只看表頭的對局史檢查、單局更新）。**全部逐位元一致**，容差為零（`glicko_replay_consistency.csv`）。

另以「雙方都是在資料內首局、表頭皆為 1500」的對局檢驗移植本身：這類對局的更新不受任何未知狀態影響，從初始值算出的 RatingDiff 應與表頭相同。重算期間這類對局全部相符；線上期間的不符，一部分是分出勝負卻記為 0 的對局（見「限制」），其餘符合同時進行的對局所致：表頭是建立對局時寫入的，結束時另一局已先更新了等級分（`glicko_replay_newcomers.csv`）。

速度：Rust 版在驗證子集上明顯快於 Python 版（`glicko_replay_timing.csv`，含 numpy 轉換的開銷）；全資料重放一次只需數秒，各組規則與參數的敏感度以 rayon 平行跑（`glicko_replay_full_timing.csv`）。

## 從 1500 起的純重放不可行：左截斷

資料始於 2013-01，但 Lichess 此前已有 rated 對局，RatingFest 重算時也包含這些對局。資料內帳號的首局表頭多數不是 1500：這些帳號的起始狀態未知。純重放從 1500 起算，不只這些帳號錯，誤差還經由對局傳給對手，於是全資料的相符比例很低，且隨時間只緩慢改善（`glicko_replay_variants.csv` 的 `pure_replay_from_1500`；差距分布見 `glicko_replay_pure_error.csv`、`glicko_replay_pure_error.png`）。

## 以表頭校正的重放

表頭雖是截斷後的整數，仍是每局對局前等級分的直接觀測：真值落在 $[h, h+1)$。重放對每個（帳號，速度類別）維護真值可能落在的區間：

1. 對局前，以表頭的區間與目前的區間取交集；交集為空時以表頭的區間重設
2. 以區間中點作為 rating，照 lila 的規則更新；雙方的 RD 與 volatility 完全由重放產生，不經任何校正
3. 更新量同時加到區間的兩端，下一局再與新表頭取交集。連續幾局後區間收窄，小數部分也被定出

資料內首局表頭不是 1500 的帳號，以表頭的區間起算，RD 取固定值（`glicko.RD_TRUNC`）、volatility 取初始值。這個 RD 依變體比較取相符最高者（`glicko_replay_variants.csv` 的 `rd_trunc_*`）。

判準：「校正前」由上一局推出的區間中點，截斷後是否等於本局表頭（`point`）；較寬的判準為區間與表頭相容（`consistent`）。首局沒有上一局可推，不計入。只有表頭可作比對時才比對：重算期間未更新的對局，表頭是重算前留下的舊值，RatingFest 不改寫它，所以不比對也不用來校正；線上期間的表頭是建立對局時寫入的，不論之後是否更新都可比對。

## 與表頭相符的程度

相符比例依月份與時間類別見 `glicko_replay_month.csv`、`glicko_replay_month.png`，依時期見 `glicko_replay_era.csv`，依帳號在資料內的累計局數見 `glicko_replay_experience.csv`、`glicko_replay_experience.png`，規則變更前後逐日見 `glicko_replay_daily.csv`、`glicko_replay_daily.png`。

- **Bullet、Blitz、Classical 在整段期間大多相符**。2013 年初因左截斷較低，之後隨帳號累積局數迅速上升，2013 年下半年起趨於穩定
- **雙方表頭對局史都乾淨時幾乎全部相符**。乾淨指帳號在資料內首局為 1500，且表頭的等級分鏈此前從未中斷（`share_point_both_clean`）
- **重算期間的 Classical 最高，線上期間降低**。重算依建立時間逐局處理，無鐘對局與 Classical 同序；線上期間更新發生在對局結束時，表頭卻是建立時寫入的，無鐘的長局與同時進行的對局使兩者的順序不一致
- **2014 年 6 月至 10 月 Bullet 與 Blitz 明顯下降**，與 pool 試行的期間吻合：該月份的不相符集中在 60+0 與 300+0 兩種時間制，其他月份這兩種時間制並不特別。pool 對局的表頭是 pool 等級分，與速度類別的鏈不連續
- **通信棋重現不了**。通信棋一局可跨數日，更新發生在對局結束時，而資料只有開始時間，無法得知更新的順序；後續分析本不使用通信棋
- **2014 年 7 月底前後**：重算結束後 Classical 的相符比例下降，其他類別無明顯斷點，可見重算與線上計算的規則對速度類別一致
- **2014 年 12 月前後**：通信棋獨立計分後 Bullet、Blitz 不受影響；Classical 除了少了無鐘對局，沒有其他變化

### 不相符的來源

`glicko_replay_attribution.csv` 依只看表頭的對局史，把每一方分成互斥的五類，各列相符比例與占不相符的份額：

- **本局表頭與自己上一局的表頭接不上**（`header_break_at_this_game`）：表頭本身不連續，任何重放都無法相符。原因包括 pool 對局、線上期間同時進行的對局、建立與結束的先後顛倒，以及重算時同一批並行處理的對局順序對調。重算期間此類很少，線上期間占不相符的份額較大
- **左截斷**（`left_truncated`）：首局表頭不是 1500 的帳號。初始 RD 與 volatility 未知，前段對局的更新量算不準，要累積相當局數後才與其他帳號相近（`glicko_replay_experience.csv` 的 `left_truncated`）
- **此前曾接不上**（`earlier_header_break`）：斷點之後區間被重設，小數部分要再數局才能定出
- **自己乾淨、對手不乾淨**：對手的 RD 或等級分有誤，經更新傳給自己，相符比例只略低於雙方都乾淨
- **雙方都乾淨**：仍不相符的部分主要是截斷邊界上的小數誤差

「RatingDiff 缺失即未更新」的假設成立：上一局沒有 RatingDiff 時，本局表頭等於上一局表頭的比例，不低於一般情形（`glicko_replay_unrated.csv`）；把缺 RatingDiff 的對局也當作更新的變體，相符明顯較低（`update_regardless_of_rating_diff`）。

## 規則解讀的比較

`glicko_replay_variants.csv` 依時期比較各變體的相符比例，逐月見 `glicko_replay_variants_month.csv`。凡原始碼有明確值者，原始碼的值都是相符最高或並列最高者，可作為移植正確的旁證：

- τ = 0.75 優於 0.5、1.0、1.2
- 有 800 下限優於沒有
- 表頭為截斷優於四捨五入
- 每步加秒改乘 40 採推定上線時間，優於 commit 時間，也優於不改
- 無鐘對局在通信棋獨立計分後即計入通信棋，優於維持 Classical 到第二次 commit
- 缺 RatingDiff 不更新，優於一律更新；分出勝負卻雙方 RatingDiff 皆為 0 的對局仍視為更新，優於視為未更新
- 左截斷帳號的初始 RD 只影響這類帳號的前段對局，各取值的差異在線上期間很小；採用的值在重算期間最高

## 重放的 RD 是否可信

相符比例檢驗的是 rating，RD 的檢驗另做：以校正後的對局前等級分配上重放的 RD 與 volatility 做一次更新，預測表頭的 RatingDiff；對照組把 RD 換成全體重放 RD 的中位數、volatility 換成初始值（`glicko_replay_step.csv`）。重放的 RD 下，預測的 RatingDiff 在 Bullet、Blitz、Classical 幾乎都與表頭相同；換成常數 RD 後明顯變差。RatingDiff 的大小由雙方的 RD 決定，所以這表示重放的 RD 接近 Lichess 當時實際使用的值。通信棋兩者都差，原因同上。

## 對後續分析的意涵

- **重放的 RD 與 volatility 可以採用**於 Bullet、Blitz、Classical，2013 年下半年起的對局。`glicko_replay.parquet` 的 `white_rd`、`black_rd` 等即對局前的值；`white_rating` 為校正後的對局前等級分，截斷後等於表頭
- **排除或另外處理左截斷帳號的前段對局**。`white_left_truncated` 標出資料內首局表頭不是 1500 的帳號，`white_prior_games` 為此前在同一類別的局數；這類帳號在累積到相符比例與其他帳號相近的局數之前（`glicko_replay_experience.csv`），RD 不可信。2014 年起左截斷帳號的占比已小
- **表頭鏈曾中斷的帳號**（`white_header_clean` 為假）RD 仍由重放產生，相符比例只略低，可以使用，但 2014 年 6 月至 10 月的 Bullet、Blitz 受 pool 影響，宜作敏感度分析
- **通信棋不可用**
- **新手期**可直接以重放的 RD 界定，不必再以累計局數代理；Lichess 在研究期間末以 RD 門檻標示 provisional（`docs/literature.md`「Lichess 的等級分實作」），門檻的取法另議
- **校準斜率的分解**：現在可以計算 Glicko-2 形式的期望得分（依對手 RD 收縮），並比較它與 Elo 形式在校準上的差異。重放的 RD 在穩定帳號身上相當集中（`glicko_replay_experience.csv` 的 `median_rd`、`p90_rd`），所以預期 RD 收縮只能解釋斜率小於 1 的一小部分，這與 `docs/calibration.md` 的推測一致，有待正式檢驗

## 限制

- **線上期間的更新順序**：更新在對局結束時發生，重放依開始時間排序。對大多數接連對局的帳號兩者一致，同時進行的對局與長局則會顛倒；表頭校正使錯誤只影響該局附近
- **重算的並行處理**：RatingFest 每次並行處理 4 局，同一帳號相鄰的對局若落在同一批，兩者可能讀到同一個起始狀態。批次的邊界由全部 rated 對局（含資料中沒有的變體棋）決定，無法重建；表頭鏈的斷點涵蓋這類情形
- **同一秒開始的對局**順序未知，以 `game_id` 排序
- **左截斷帳號的起始 RD 與 volatility 未知**，以固定值起算
- **分出勝負卻雙方 RatingDiff 皆為 0 的對局**：分差懸殊時真實的更新可能不到 1 分，但兩個 1500 新帳號之間不可能為 0，可見其中有未更新卻記為 0 的對局。視為未更新的變體相符較低，故採用視為更新
- **pool 對局無法辨認**，其表頭不是速度類別的等級分
- **帳號後來被標為作弊**：重算時作弊帳號的對局已不在 rated 資料中；線上期間被標記前的對局仍有 RatingDiff，照常更新

## 建置

`rust/glicko_replay/` 以 maturin 建置（pyo3、rust-numpy、rayon）。根目錄 `pyproject.toml` 把它列為依賴，並以 `[tool.uv.sources]` 指向本地路徑，`uv sync`、`uv run` 會自動建置；crate 的 `pyproject.toml` 以 `[tool.uv] cache-keys` 讓 Rust 原始碼變動時重建。`.cargo/config.toml` 設 `target-cpu=native`，release profile 開 `lto = "fat"`、`codegen-units = 1`、`opt-level = 3`：產出的二進位檔只適用本機的 CPU，換機器須重新建置。

Rust 版的熱點是逐局依序的 Glicko-2 更新與 volatility 的迭代求解，每局依賴前一局的狀態，無法以 numpy 向量化，也沒有現成套件實作 Lichess 的規則；這符合 `CLAUDE.md` 寫 Rust 擴充的兩個條件。單一序列內不並行；rayon 只用於同時跑多組規則與參數，以及各局互相獨立的單局更新。重放沒有矩陣或向量代數的運算，因此不使用 faer。
