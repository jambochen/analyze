# 計算性能

執行重分析的步驟，以及記憶體、速度與數值問題的成因與解法。耗時與峰值記憶體隨機器負載與口徑變動，只存在輸出檔與腳本輸出。

## 執行環境

本機 30 GB 記憶體、20 核心、RTX 3070（8 GB）。根分割區的 SSD 空間常不足，寫大檔前先查剩餘空間。

## 執行重分析的步驟

1. **估耗時**：實測一個工作單位（一次重抽、一次配適）的耗時，乘以數量得總時間。完成條件：有一個總時間的估計
2. **冒煙測試**：以少量重複跑完整支腳本，只看能否跑完。冒煙測試的工作數少，CPU 曲線不代表正式執行
3. **正式執行**：一次只跑一個重分析，包在 `systemd-run --user --scope -p MemoryMax=… -p MemorySwapMax=0` 內，以 `POLARS_MAX_THREADS`、`OMP_NUM_THREADS`、`OPENBLAS_NUM_THREADS` 限制執行緒。OpenBLAS 預設開到核心數，矩陣乘向量受記憶體頻寬限制，多開的工作執行緒在每次呼叫後空轉等待，佔住 hyperthread 與 E-core，拖慢主執行緒與 jax、polars 的執行緒；`OPENBLAS_NUM_THREADS=4` 時整支腳本反而較快（`continuation_selection.py`）。這些變數須在 import numpy 前生效，寫在命令列上。超過上限時行程直接被砍（exit 137），不拖進 swap
4. **監看**：平行池的每個工作完成時印一行已完成數，由經過時間推估剩餘時間；與第 1 步的估計差很多時停下來找原因

長時間的 API 抓取以 `systemd-run --user --unit=…` 在工作階段外執行，加 `Restart=on-failure`，日誌放專案內而非 `/tmp`。

熱點先向量化（numpy、polars、jax 的 `vmap`）；向量化不夠時直接寫 Rust，不必先經 numba。Rust 的條件見「Rust 擴充」。

## 記憶體

### polars 的 group_by

瓶頸是唯一分組數，不是列數、檔案大小或字串型別：group_by 的 hash table 須容納全部分組鍵才能輸出，串流與 `sink_parquet` 都救不了。依資料的天然分區（月份、速度類別）拆開聚合再合併；分組鍵巢狀於分區鍵時，結果與全庫相同。遇到 OOM 先估分組數。

### 帳號作分組鍵

帳號一律用整數代碼：`scripts/lichess_games.py` 把帳號轉成 Enum 再取物理碼，得到 `UInt32` 的 `white_id`、`black_id`；`player_games.parquet` 的 `player`、`opponent` 用同一套代碼。帳號名稱只在與 API 資料串接時使用。

### pyfixest 的資料副本

pyfixest 預設在模型物件中保留資料副本與去均值後的設計矩陣，兩樣本疊起來的模型會超過記憶體上限（`scripts/fixed_effects_churn.py`）。只需係數與 CRV1 共變異時以 `lean=True, store_data=False, copy_data=False` 擬合，轉成 pandas 後即 `del` 原表；完整模型與受限模型依序建立，取出所需的量即釋放並 `gc.collect()`；去均值用 `demeaner_backend="rust"`。

### 最大概似的逐塊計算

jax 對數百萬列一次求梯度與 Hessian，會展開與列數同長的中間陣列。`scripts/likelihood.py` 的 `fit` 把資料切成固定大小的塊，逐塊 jit 計算後加總。jit 過的函式以 `ll` 為鍵快取，`t0` 與 `free` 當作輸入：每次 fit 都建新閉包時每次都要重新編譯，編譯時間與資料量無關，小樣本的 fit 幾乎全花在編譯上。每塊補齊到 2 的冪次長度、以 mask 排除填充列，不同樣本大小共用少數幾種形狀的編譯結果。值、梯度與 Hessian 在同一個 jit 內算出並記住最後一點，trust-exact 在同一點依序要三者時只算一次。以得分為反應、線性預測式 `X theta` 的 logistic（`likelihood.logit_ll`，`continuation_selection.py`、`rating_filter_error.py`）改由 Rust 計算（`rust/likelihood/`，Python 模組名 `likelihood_rs`）：XLA 在 CPU 上逐塊依序執行，對逐元素運算與加總只平行到兩三個核，切小塊也不會同時執行；Rust 版各列以 rayon 分塊平行，一次掃過資料同時累加值、梯度與 Hessian。`logit_ll` 的 jax 寫法即參考實作，其他 `ll` 照舊走 jax。同一個 crate 也計算 sandwich 的 meat（`dyad_meat` 與 `lrt_player_dyad` 中依玩家分群的 meat）：numpy 版以 `np.unique` 排序再逐欄 `bincount`，全為單執行緒，在 Rust 版的 fit 之後成為主要的串列段落。Rust 版以平行排序分組；不以玩家代碼開累加陣列，因為代碼是全庫代碼，遠大於樣本內的玩家數，每條執行緒一份會配置、清零與相加大量記憶體，實測反而比 numpy 慢。驗證見 `scripts/likelihood_rust_matches_python.py`，模擬的玩家代碼須稀疏才測得到這一點。兩條路徑的加總順序不同，trust-exact 在預設的 `gtol` 內可能停在略有不同的點，個別模型的參數差到 1e-10、下游 p 值差到第六位有效數字。每個受限模型都要重估的分析，在各速度類別抽固定大小的簡單隨機樣本（`docs/results/calibration.md`「三類結果的有序模型」）。

## 速度

### jax 一律在 CPU 上

RTX 3070 的 float64 吞吐量遠低於 float32，最大概似、Hessian 與尾機率都需要 float64。用 jax 的腳本在檔案最上方、import jax 之前寫 `os.environ["JAX_PLATFORMS"] = "cpu"`：`likelihood.py` 的 `setdefault` 在 jax 已被 import 時來不及生效。GPU 只留給大量重複且 float32 足夠的模擬（`bench_nuts_backend.csv`）。

### 批次大小

工作切得太碎時，時間花在 Python 與 jit 函式的往返上，CPU 曲線呈鋸齒狀。`scripts/adjusted_lrt_size.py` 把一批重複疊成一個陣列，以 `jax.vmap` 同時做 Newton 迭代，各重複各自判斷步長；各重複的 dyadic meat 以「玩家代碼加重複次數的偏移」一次 `bincount` 算完。能以 `vmap` 表達的熱點不需要 Rust。

### 平行池

permutation test 的重抽由 `scripts/pairing_null.py` 的 `run` 以 fork 的行程池平行，`arena_crossover.py` 與 `match_quality_leave.py` 共用：
- 所有虛無分布的工作合成一個清單一次送出，最後一波才不會只剩少數行程在跑
- 每個工作一次跑一批重抽，批的大小取到工作數為行程數的數倍，但不小到讓呼叫開銷蓋過運算
- 第 i 次重抽的種子固定為虛無分布的序號與 i，結果不隨工作分到哪個行程而變
- 每次重抽只呼叫一次統計量函式，再從結果取各統計量；在 comprehension 內逐一取用會讓每次重抽重算整組，結果正確、只是慢，靠第 1 步的估計才看得出來
- jax 的執行緒啟動後再 fork 可能死結：平行池需要的模型以 numpy 與 scipy 估計，jax 的配適放在平行池結束之後
- 限制 XLA 執行緒的 `sched_setaffinity` 會被 fork 出的工作行程繼承，所以寫在用 jax 的腳本內、平行池之後，不寫在被 import 的 module 裡

每次重抽的時間分在重抽配法、對新對手的期望得分查表與統計量三段，量級相近（`scripts/explore/pairing_lookup_cache.py`）。查表 `E[速度類別, 自身, 對手]` 雖大於 L3，但兩局以上的批列數不多，查表不是瓶頸；改以 `calibrated.probs` 的封閉形式逐列計算反而較慢，因為 numpy 每一步都產生與列數同長的暫存陣列。行程數接近核心數時三段都變慢，擴展受記憶體頻寬、hyperthread 與 E-core 限制。整個 `run` 的耗時短，這一段不需要 Rust。

平行池以外的 polars 建表、逐位玩家的 Python 迴圈是單執行緒，jax 的最大概似只部分平行；這些段落短時不必處理。

### Glicko-2 的逐局重放

每局的更新依賴前一局的狀態，無法向量化，也沒有套件實作 Lichess 的規則，所以寫成 Rust（`rust/glicko_replay/`，pyo3、rayon，native 編譯與 fat LTO），以 Python 參考實作逐位元驗證（`docs/data/glicko-replay.md`「實作與一致性驗證」「建置」）。rayon 只用於同時跑多組規則與參數（`glicko_replay_timing.csv`、`glicko_replay_full_timing.csv`）。

### 玩家 block bootstrap 的修剪平均

`scripts/continuation_selection.py` 的 Lee 與 worst-case 界限要對每次重抽取加權修剪平均：續下者依反應變數排序，權重為玩家權重乘該次重抽的次數，累計權重到修剪比例為止。numpy 只能展開「重抽 × 列」的權重矩陣再 `cumsum`、`clip`，單執行緒且受記憶體頻寬限制，佔整支腳本的大半時間。套件沒有加權修剪平均的 bootstrap，所以寫成 Rust（`rust/weighted_trim/`，Python 模組名 `weighted_trim_rs`，pyo3、rayon，建置方式同 `rust/glicko_replay/`）：各次重抽以 rayon 平行，每次只持有一列權重，累計超過修剪比例即提前結束。只要加權平均時（IPW）傳入零欄的比例。同一個 crate 的 `tilt_means` 是有參數敏感度分析的傾斜平均：每個 gamma 都要以 brentq 求根，每次求值掃一遍續下者算 exp，numpy 只能逐一 gamma 呼叫、單執行緒；Rust 版移植 scipy 的 brentq（容差相同），各 gamma 以 rayon 平行。

`scripts/weighted_trim_rust_matches_python.py` 以原本的 numpy 與 scipy 寫法為參考實作比對；numpy 的 `sum` 為 pairwise、Rust 為逐項累加，只差浮點誤差，比的是相對誤差而非逐位元。

只用線性統計量（加權和、比值）的 bootstrap 不需要這個擴充：先依玩家加總，再以重抽次數矩陣相乘（`continuation_selection.py`、`folk_stop_rules.py`）。要乘的向量多時疊成一個矩陣一次相乘：重抽次數矩陣大於 L3，逐次的矩陣乘向量每次都從記憶體重讀整個矩陣，IPC 遠低於 1，BLAS 已用滿 AVX2 也無濟於事；矩陣乘矩陣時 BLAS 分塊重用，同樣的浮點運算量快數倍。

### 條件概似的 logistic 迴歸的分母

給定流失次數時，分母是所有流失日組合的總和。以初等對稱多項式的動態規劃逐位玩家精確計算，流失日多於一半時改以未流失日為事件，計算量取決於 $\min(k, n-k)$（`scripts/clogit.py`、`docs/results/appendix/fixed-effects-churn.md`）。

## Rust 擴充：僅限實測過的熱點

預設全部用 Python。只有同時滿足以下兩個條件，才考慮寫 Rust crate 以 pyo3 綁定：

1. **該段是經常觸發瓶頸**。該運算會頻繁或多處使用。
2. **既有套件解不掉**。該運算在 numpy / scipy / statsmodels / polars 裡沒有
   對應實作，或有實作但是純 Python 迴圈、未走 C/C++/Rust 加速

兩者皆成立時，Rust 的價值在 rayon 並行與 SIMD，適合的形態是自訂的逐列數值
迴圈、bootstrap 重抽、對每個群組重複擬合這類 numpy 向量化表達不出來的運算。

反之不要寫：能用 numpy 向量化改寫的、只跑一次的前處理、佔總時間比例低的段落。
維護一條 Rust 建置鏈的成本高於它省下的時間。

crate 放 `rust/<name>/`，以 maturin 建置，在 `pyproject.toml` 宣告依賴。
Python 側須保留一份等價的參考實作，用於驗證 Rust 版結果一致。

## 數值

### 加權卡方的尾機率

Imhof 的被積函數在無窮區間上振盪，統計量越大振盪越快、尾機率越小，數值積分的誤差蓋過真值。`scripts/likelihood.py` 的 `adjusted_lrt`：
- df 為 1 時用精確解 $\lambda\chi^2_1$
- df 大於 1 時，以 $\sin(A - \omega u) = \sin A \cos \omega u - \cos A \sin \omega u$ 拆開，第一個週期以 `quad` 積，之後以 Fourier 型積分（QAWF）
- 上界 $P(\lambda_{\max}\chi^2_{df} > x)$ 小於 $10^{-6}$ 時報這個嚴格的上界

### 受限模型

- **沒有自由參數**（例如 Elo 形式完全校準）：`fit` 只計算固定參數下的對數概似，不呼叫最佳化器
- **起始值**：以「把參數固定在起始值」實作受限模型時，起始值中受限的位置必須等於虛無假設的值（`expected_score_baselines.py` 的 `T0_GLICKO`）

## 操作

- **停止行程**：`pkill -f` 的 pattern 若也出現在執行它的 shell 命令列中，該 shell 會一起被砍（exit 144）；先以 `pgrep -f` 列出，用精確的 pattern 再砍
- **CPU 曲線**：使用率長期低於核心數、呈鋸齒狀，多半是呼叫開銷或單執行緒的 Python 迴圈；先改成批次，再考慮換語言
- **剖析**：`ptrace_scope` 為 1，py-spy 不能 attach 已在跑的行程，但由它啟動的子行程可以：`py-spy record --native --idle -f speedscope -o … -- python scripts/<name>.py`，外層加 `/usr/bin/time -v` 看平均使用的核心數與 context switch。perf 在 `perf_event_paranoid` 為預設的 4 時不可用，需要時以 `sudo sysctl kernel.perf_event_paranoid=2` 暫時放寬，重開機即恢復
- **CPU 尖刺的成因**：尖刺是 BLAS、jax、polars 的短暫多執行緒，尖刺之間的低谷才是瓶頸，多半是單執行緒 numpy 在掃大型暫存陣列；等記憶體時 CPU 照樣算作忙碌，不會呈尖刺；反過來，CPU 使用率高也不代表在運算，OpenBLAS 的空轉等待同樣算作忙碌，perf 中以 `blas_thread_server` 出現
