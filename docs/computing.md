# 計算性能的問題與解法

分析過程中遇到的記憶體、速度與數值問題，各自的成因與採用的解法。本文件只寫成因與做法，耗時與峰值記憶體一律見對應的輸出檔或腳本輸出；它們隨機器負載與口徑變動。

## 執行環境與原則

- 本機 30 GB 記憶體、20 核心、RTX 3070（8 GB）；根分割區的 SSD 空間常不足，寫大檔前先查剩餘空間
- **一次只跑一個重分析**：一個分析能用滿硬體時依序執行。多個分析同時跑、又沒有記憶體上限，曾耗盡記憶體拖垮整個桌面工作階段
- **每次執行都設記憶體上限**：包在 `systemd-run --user --scope -p MemoryMax=… -p MemorySwapMax=0` 內，並以 `POLARS_MAX_THREADS`、`OMP_NUM_THREADS` 限制執行緒。超過上限時行程直接被砍（exit 137），不會拖進 swap
- **熱點的處理順序**：先向量化（numpy、polars、jax 的 `vmap`），不行再用 numba，最後才寫 Rust。Rust 的條件見 `CLAUDE.md`
- **長時間的 API 抓取**：以 `systemd-run --user --unit=…` 在工作階段外執行，加 `Restart=on-failure`，日誌放專案內而非 `/tmp`

## 記憶體

### polars 的 group_by：瓶頸是分組數

**症狀**：串流（`collect(engine="streaming")`）在一億多列的對局資料上仍然記憶體不足。

**成因**：group_by 的 hash table 必須容納全部唯一的分組鍵才能輸出，是 pipeline breaker。瓶頸是唯一分組數，不是列數、檔案大小或字串型別；`sink_parquet` 同樣救不了，因為問題在聚合的中間狀態，不在輸出。曾誤歸因為「高基數字串」與「一次查太多欄」，據此改寫的函式其實不需要改。

**解法**：依資料的天然分區（例如月份、速度類別）拆開聚合再合併。分組鍵巢狀於分區鍵之內時，拆分結果與全庫完全相同，不是近似。遇到 OOM 先估分組數再動手。

### 帳號字串作分組鍵

**成因**：以帳號名稱分組與串接，每個鍵都是變長字串，hash table 與 join 的中間結果都大。

**解法**：`scripts/lichess_games.py` 把帳號轉成 Enum 再取其物理碼，得到 `UInt32` 的 `white_id`、`black_id`；`player_games.parquet` 的 `player`、`opponent` 用同一套代碼。帳號名稱只在與 API 資料串接時使用。

### 固定效應模型：設計矩陣與資料副本

**症狀**：`scripts/fixed_effects_churn.py` 在記憶體上限下兩次被砍，先後死在兩樣本疊起來的完全交互作用模型與 Mundlak 模型。

**成因**：pyfixest 預設在模型物件中保留資料副本與去均值後的設計矩陣；疊起來的模型列數加倍，固定效應也加倍。Mundlak 檢定原本同時持有完整模型與受限模型。

**解法**：
- 只需要係數與 CRV1 共變異的模型，以 `lean=True, store_data=False, copy_data=False` 擬合，轉成 pandas 後立即 `del` 原 polars 表
- 完整模型與受限模型依序建立，取出對數概似、係數與共變異後即釋放，再 `gc.collect()`
- 去均值用 `demeaner_backend="rust"`

### 最大概似的逐塊計算

**成因**：jax 對數百萬列一次求梯度與 Hessian，會展開與列數同長、乘上參數維度的中間陣列。

**解法**：`scripts/likelihood.py` 的 `fit` 把資料切成固定大小的塊，對數概似、梯度與 Hessian 逐塊 jit 計算後加總，逐局分數也逐塊算再接起來。校準與三個基準另在各速度類別抽固定大小的簡單隨機樣本估計，因為每個受限模型都要重估（`docs/calibration.md`「三類結果的有序模型」）。

## 速度

### jax 在 GPU 上算 float64

**症狀**：最大概似與模擬比預期慢很多，CPU 使用率低。

**成因**：jax 偵測到 GPU 就預設使用。RTX 3070 的 float64 吞吐量遠低於 float32，而最大概似、Hessian 與尾機率需要 float64。`likelihood.py` 雖以 `os.environ.setdefault("JAX_PLATFORMS", "cpu")` 設定，但腳本若在 import `likelihood` 之前已 import jax，設定就來不及生效。

**解法**：凡用 jax 的正式腳本，在檔案最上方、import jax 之前寫 `os.environ["JAX_PLATFORMS"] = "cpu"`。GPU 只留給大量重複且 float32 足夠的模擬。貝氏抽樣的後端比較見 `bench_nuts_backend.csv`。

### 模擬切得太碎

**症狀**：`scripts/adjusted_lrt_size.py` 的檢定水準模擬，CPU 使用率曲線呈鋸齒狀，大部分時間只有少數核心在忙。

**成因**：每次重複各自呼叫一次 scipy 最佳化器，每次呼叫都要經過 Python 與 jit 函式的往返；單次的資料量小，時間幾乎全花在呼叫開銷上，向量運算本身佔比很低。

**解法**：
- 一批重複疊成一個陣列，以 `jax.vmap` 同時做 Newton 迭代；步長逐次減半，各重複各自判斷是否接受
- 整批的步長都被接受即提前結束減半，整批的步長都夠小即結束迭代
- 各重複的 dyadic meat 以「玩家代碼加上重複次數的偏移」編碼，一次 `bincount` 算完整批
- 批的大小加大到每批的計算量足以蓋過呼叫開銷

批內向量化後就不需要 Rust：熱點能以 `vmap` 表達，不符合 `CLAUDE.md` 寫 Rust 的第二個條件。

### Glicko-2 的逐局重放

**成因**：每局的更新依賴前一局的狀態，volatility 又要迭代求解，無法以 numpy 向量化，也沒有現成套件實作 Lichess 的規則。

**解法**：寫 Rust 版（`rust/glicko_replay/`，pyo3、rayon，native 編譯與 fat LTO），以 Python 參考實作逐位元驗證（`docs/glicko-replay.md`「實作與一致性驗證」「建置」）。單一序列內不並行；rayon 只用於同時跑多組規則與參數。速度見 `glicko_replay_timing.csv`、`glicko_replay_full_timing.csv`。

### conditional logit 的分母

**成因**：給定流失次數時，概似的分母是所有可能流失日組合的總和，直接列舉不可行。

**解法**：以初等對稱多項式的動態規劃逐位玩家精確計算，流失日多於一半時改以未流失日為事件，計算量取決於 $\min(k, n-k)$；以 jax 自動微分求梯度與 Hessian（`scripts/clogit.py`、`docs/fixed-effects-churn.md`）。

## 數值

### 加權卡方的尾機率

**症狀**：Imhof 反轉公式在統計量大時算出負值，或給出不可信的 p 值；一般的 `quad` 耗盡細分次數。

**成因**：被積函數在無窮區間上振盪，振幅衰減慢；統計量越大，振盪越快，尾機率又越小，數值積分的絕對誤差蓋過真值。

**解法**（`scripts/likelihood.py` 的 `adjusted_lrt`）：
- df 為 1 時虛無分布即 $\lambda\chi^2_1$，用精確解
- df 大於 1 時，以 $\sin(A - \omega u) = \sin A \cos \omega u - \cos A \sin \omega u$ 拆開，第一個週期照原式以 `quad` 積，之後交給 Fourier 型積分（QAWF）
- 上界 $P(\lambda_{\max}\chi^2_{df} > x)$ 小於 $10^{-6}$ 時改報這個嚴格的上界，不再數值積分

### 受限模型沒有自由參數

**症狀**：「Elo 形式完全校準」的受限模型把參數全部固定，最佳化器收到空的參數向量與空的 Hessian 而出錯。

**解法**：`fit` 在沒有自由參數時只計算固定參數下的對數概似，不呼叫最佳化器。

### 受限模型的起始值

**症狀**：Glicko 形式的受限模型收斂到錯的值。

**成因**：受限參數固定在共用起始值 `T0` 上，而 `T0` 的分差斜率不是 0，等於檢定了錯的虛無假設。

**解法**：受限參數另給一組全為 0 的起始值（`expected_score_baselines.py` 的 `T0_GLICKO`）。凡以「把參數固定在起始值」實作受限模型，起始值中受限的位置必須等於虛無假設的值。

## 操作

- **`pkill -f` 會殺到自己**：pattern 若同時出現在執行 `pkill` 的 shell 命令列中，該 shell 也會被砍（exit 144）。改以 `pgrep -f` 先列出、用更精確的 pattern 再砍
- **監看 CPU 曲線判斷瓶頸**：使用率長期低於核心數、呈鋸齒狀，多半是呼叫開銷或單執行緒的 Python 迴圈，不是運算量本身；先改成批次再考慮換語言
