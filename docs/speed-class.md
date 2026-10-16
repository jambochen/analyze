# 速度類別：表頭等級分屬於哪一個類別

規則的實作在 `scripts/glicko.py` 的 `speed()`，`scripts/lichess_games.py` 以它在 `data/processed/lichess_games.parquet` 加上 `speed` 欄。修正前後的比較在 `scripts/speed_fix_effect.py`，表在 `outputs/tables/speed_fix_effect_*.csv`。本文件只寫定性結論，數值一律見對應的輸出檔。

## 問題

Lichess 依速度類別（Bullet、Blitz、Classical、通信棋）分別維護等級分，表頭的 `WhiteElo`、`BlackElo` 是對局當時該局所屬速度類別的等級分。月檔 `Event` 欄的時間類別（`lichess_games.parquet` 的 `time_class`）卻不是當時的類別：它是匯出月檔時，以匯出當下的規則「底秒 + 每步加秒 × 40」事後標的。這條規則在 2015 年 4 月下旬才上線，此前計分用的是「底秒 + 每步加秒 × 30」；無鐘對局在 2014 年 12 月以前與 Classical 共用等級分，月檔卻一律標為通信棋。

以 `time_class` 分組，會把屬於不同速度類別的等級分放進同一組，也把同一速度類別的對局拆到兩組。例如 `300+5` 在 2015 年 4 月下旬以前計入 Blitz，月檔標為 Classical：在 Classical 的校準樣本裡，這些局的雙方等級分是 Blitz 的等級分。

## 判定規則與出處

沿用 Glicko-2 重放已驗證的規則（`docs/glicko-replay.md`「移植的規則」與「規則變更的實際上線時間」），不另訂一套。依對局的建立時間（`start`）、`base_seconds` 與 `increment_seconds`：

| 期間 | 有鐘的對局 | 無鐘的對局 |
|---|---|---|
| 2014-12-01 09:46 UTC 以前 | est = 底秒 + 30 × 每步加秒；小於 180 秒為 Bullet，小於 480 秒為 Blitz，其餘為 Classical | Classical |
| 2014-12-01 09:46 至 2015-04-25 07:00 UTC | 同上 | 通信棋 |
| 2015-04-25 07:00 UTC 以後 | est = 底秒 + 40 × 每步加秒，門檻同上 | 通信棋 |

- 門檻與 30 倍：scalachess `Speed.scala`、`Clock.scala` @ `1ceba9320d`；lila `PerfsUpdater.scala` @ `bfad573339`
- 改為 40 倍：scalachess `1fc3f0d44a`；上線時間由表頭等級分鏈推定（`scripts/explore/speed_boundaries.py`），晚於 commit 約十餘小時（`glicko.T_EST40`）
- 無鐘對局改計入通信棋：lila `804567893e`（`glicko.T_CORR`）

驗證是 Glicko-2 重放本身：依這套類別逐局重放，Bullet、Blitz、Classical 絕大多數對局的對局前等級分與表頭相符；把 40 倍的上線時間改為 commit 時間或整段都用 30 倍、無鐘對局維持 Classical 到第二次 commit，相符都較低（`glicko_replay_variants.csv`）。若以月檔的 `time_class` 為類別，等級分鏈會在受影響的時間制上斷開。

`time_class` 保留在 parquet 中只供對照，後續分析一律以 `speed` 分組。

### 無鐘對局

2014 年 12 月以前的無鐘對局在 `speed` 中屬 Classical，因為它們確實更新 Classical 的等級分。但它們沒有時限，一局可跨數日，開局時間不代表玩家當下在線，也沒有名目對局長度。校準、連續對局與其後的分析因此一律只用有鐘的對局：排除條件由「`time_class` 不是通信棋」改為「有鐘」，兩者在本資料中恰好是同一組對局，所以各分析的對局集合不變，改變的只是對局歸入哪一個類別。

## 受影響的範圍

`speed_fix_effect_cross.csv` 為兩種分類的交叉表，`speed_fix_effect_month.csv` 為各月有鐘對局中兩者不一致的比例，`speed_fix_effect_time_control.csv` 依時間制列出不一致的對局。

- **只往較慢的類別錯**：月檔的類別系統性地把對局標到比實際慢一級的類別。月檔標為 Classical 而實際計入 Blitz 的對局（`300+5`、`180+8`、`360+3` 等每步加秒較大的時間制）最多；其次是標為 Blitz 而實際計入 Bullet 的對局（以 `60+3` 為主）；沒有方向相反的情形，因為 40 倍的估計總是不小於 30 倍
- **期間**：有鐘對局的不一致從資料起點一直持續到 2015 年 4 月下旬，之後兩者一致。各月的比例大致穩定，占有鐘對局的少數
- **無鐘對局**：2014 年 12 月以前的無鐘對局在月檔標為通信棋、`speed` 為 Classical；分析一律排除，不影響樣本
- **arena 對局不受影響**：本資料中 arena 對局的時間制在兩種規則下都落在同一個類別，不一致全部發生在大廳對局。arena 的分析之所以仍會變動，是因為它們使用依類別估計的校準係數與連續對局門檻

修正改變的是類別歸屬，因而改變：依類別估計的校準模型（`calibration.py`，經 `calibrated.py` 進入匹配品質、期望得分殘差與玩家日的校準變數）、依前一局類別擬合的連續對局門檻（`sessions.py`，進入 `continue_session` 及其一半、兩倍的版本）、以類別為控制變數或分層的分析。各分析的對局集合不變；Glicko-2 重放本來就依此規則判定類別，結果不變。`arena_random_pairing.py` 只用配對批次與表頭等級分，不用類別、校準係數或門檻，不受影響，未重跑。

## 修正前後主要結論是否改變

逐項的前後對照見 `speed_fix_effect_estimates.csv`（每列為一個估計量，含差距及差距相對於修正前標準誤的倍數），依表與量彙整的變號與顯著性改變見 `speed_fix_effect_summary.csv`。

**主要結論全部不變**。各分析文件（`docs/calibration.md`、`docs/sessions.md`、`docs/match-quality-leave.md`、`docs/eomm-assumption.md`、`docs/fixed-effects-churn.md`）的定性結論在修正後逐項成立。

- **校準**：三個類別的校準斜率仍明顯小於 1，各檢定的結論與 Holm 校正後的判定都不變。變動最大的是 Classical：移出的每步加秒較大的時間制，原本是以 Blitz 的等級分混入 Classical；移出後 Classical 的校準斜率略升，偏差仍在。變號者只有與 0 無法區分的交互作用項
- **連續對局的門檻**：各類別的主門檻變動一至兩分鐘，比值門檻不變。續下率、離開率的水準只有小幅變化，類別、arena 與一般對局、輸贏之間的大小順序都不變
- **匹配品質的總效果**：三個主反應變數的方向、randomization inference 與 Holm 調整後的結論不變，係數的變動遠小於標準誤。變動明顯的只有以主門檻一半定義的連續對局結束：門檻變動相對於一局的長度不小，效果變小但仍是各反應變數中最大，「門檻越短效果越大」的梯度不變；該反應變數中和局的係數由正轉為與 0 無法區分，與 `docs/sessions.md` 指出和局的位置對門檻敏感一致
- **EOMM 關鍵假設**：Wald 檢定在各層、各反應變數仍拒絕；$\Delta$ 的正負與顯著性不變，一局一批的 `leave_session` 仍與 0 無法區分
- **Heliyon 模型的比較**：以原始等級分差為變數的設定完全不變（不經類別）；以校準期望得分或 $q$ 為變數的設定只有可忽略的變動

結論不變的原因：受影響的對局只占少數，而且移動發生在相鄰的類別之間，相鄰類別的校準係數本來就接近；arena 對局本身的類別沒有改變。

修正仍是必要的：以月檔類別分組時，同一組的雙方等級分可能屬於不同的等級分池，「依類別估計」的模型在定義上就不成立。這次的比較說明這個錯誤在本資料中的影響很小，不說明它可以忽略。
