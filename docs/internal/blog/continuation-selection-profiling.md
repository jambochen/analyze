# continuation_selection.py 從 4.5 分鐘到 39 秒：一次剖析的覆盤

機器：i7-12700K（8 P-core + 4 E-core，20 執行緒，AVX2，無 AVX-512），30 GB，L3 25 MB。
耗時為這台機器上單次執行的實測，只供覆盤時比較量級。

## 起點

腳本的 CPU 曲線常呈尖刺。直覺是「Python 在等待或調度，不在運算」，想找各腳本共同的熱點改寫成 Rust。

## 第一步走錯：憑讀程式碼猜熱點

讀 `pairing_null.py` 時推論：期望得分表 `E[速度類別, 自身, 對手]` 有 72 MB，大於 L3，對手的等級分在批內不相關，查表近乎隨機存取，應該受快取限制；改以封閉形式逐列計算應較快。

寫了 microbench（`scripts/explore/pairing_lookup_cache.py`）在 1、4、9、18 個行程下比較：

- 查表每列約 13 ns，不是瓶頸：兩局以上的批只有三十多萬列，硬體的 memory-level parallelism 吸收了 miss
- 封閉形式反而**更慢**：numpy 每一步都產生一個與列數同長的暫存陣列
- 真正的問題是平行擴展：18 個行程時三段都慢約 4 倍

教訓：快取的推論要量過才算數。被擠爆的是 numpy 的暫存陣列，不是那張表。

## 工具與權限

### `/usr/bin/time -v`

最便宜的第一眼：
- `Percent of CPU`：平均用了幾個核
- `Voluntary context switches`：多代表在等鎖、I/O 或其他執行緒
- `Involuntary context switches`：多代表執行緒數超過核心數、被搶

### py-spy

```
py-spy record --native --idle -r 50 -f speedscope -o x.json -- python scripts/<name>.py
```

- `kernel.yama.ptrace_scope = 1` 時**不能 attach** 已在跑的行程，但**由 py-spy 啟動的子行程可以**。原本以為 py-spy 在本機不能用，其實只是 attach 不行
- `--native` 才看得到 numpy、BLAS 內部的 C 函式；代價是腳本變慢很多（39 秒的腳本在 `--native -r 100` 下跑了約兩分鐘），比例仍可參考，絕對時間不可信
- speedscope 格式是 JSON，以 Python 彙總每一行的 inclusive 比例最快，不必開瀏覽器

### perf

- Ubuntu 預設 `kernel.perf_event_paranoid = 4`，一般使用者完全不能用。`sudo sysctl kernel.perf_event_paranoid=2` 即可量自己行程的 user space 事件，重開機恢復
- 混合架構的事件分成兩組：`cpu_core/cycles/` 與 `cpu_atom/cycles/`。只寫 `cycles` 時兩組混在一起；要乾淨的數字就 `taskset -c 0` 綁在 P-core 上，只量 `cpu_core/...`
- Python 3.12 起有 perf trampoline：`python -X perf`，報告中 Python 函式會以 `py::` 出現。實際上 numpy、OpenBLAS 沒有 frame pointer，callchain 接不回 Python 層，最後還是靠 py-spy 對應行號
- `perf report --tid` 在這次沒有輸出；改以 `perf script -F comm,tid | sort | uniq -c` 數每條執行緒的樣本數，再 `perf script --tid <主執行緒>` 自己彙總葉節點

有用的事件組合：

```
perf stat -e cpu_core/cycles/,cpu_core/instructions/,cpu_core/LLC-load-misses/,\
cpu_core/fp_arith_inst_retired.256b_packed_double/,cpu_core/fp_arith_inst_retired.scalar_double/ -- ...
```

IPC（instructions / cycles）與 LLC miss 判斷受記憶體還是受運算限制；兩個 `fp_arith` 事件判斷浮點運算是否已經 SIMD。量之前先跑一次「只建資料、不運算」的版本當基線扣掉。

## 剖析結果與對應的改動

### 加權修剪平均 → Rust

py-spy：`trim_low`、`trim_high` 與建權重矩陣佔約六成。numpy 寫法要展開「重抽 × 列」的權重矩陣再 `cumsum`、`clip`，self time 前幾名全是 `DOUBLE_add`、`subtract`、`multiply`、`clip`、strided copy：單執行緒在掃大型暫存陣列，正是曲線尖刺之間的低谷。

寫成 `rust/weighted_trim/`：rayon 逐次重抽平行，每次只持有一列權重，累計超過修剪比例就提前結束。只要加權平均（IPW）時傳零欄的比例，同一個函式重用。

驗證：`scripts/weighted_trim_rust_matches_python.py` 以舊的 numpy 寫法為參考，最大相對差 1e-11（numpy 的 `sum` 是 pairwise，Rust 逐項累加），單次呼叫快 60–200 倍。整支腳本 4:33 → 1:12，輸出最大相對差 2e-13。

### OpenBLAS 的空轉 → 限制執行緒

改完後 `time -v` 顯示 CPU 1248%，看似用滿。perf 才看出真相：

- 87% 的 CPU 樣本在 OpenBLAS，其中 `blas_thread_server` 佔 56%：十幾條 BLAS 工作執行緒幾乎整段執行都在**空轉等待**
- Rust kernel 只剩 2.5%

`OPENBLAS_NUM_THREADS=4`：1:12 → 0:56，CPU 降到 471%。設為 1 也只要 0:59。空轉的執行緒佔住 hyperthread 與 E-core，拖慢主執行緒與 jax、polars。這個變數必須在 import numpy 之前生效，寫在命令列。

教訓：CPU 使用率高不代表在運算；尖刺也不代表在等記憶體，等記憶體時 CPU 照樣算忙碌。

### 矩陣乘向量 → 矩陣乘矩陣

主執行緒剩下最大的是 `dgemv`：`bsum` 每組呼叫約 75 次 `C @ v`，`C` 是 999 × 玩家數的 float64，大於 L3。

perf stat（單一 P-core，扣除基線）：

| | IPC | LLC miss | 浮點運算 |
|---|---|---|---|
| 75 次 GEMV | 0.33 | 約 5 億 | 全為 256-bit packed |
| 1 次 GEMM | 3.3 | 約 100 萬 | 全為 256-bit packed |

兩者的 SIMD 運算數幾乎相同，差別完全在資料重用。把所有（格, 等分, 玩家）的 `bincount` 疊成一個矩陣，一次 `C @ V.T`：0:56 → 0:39。

### SIMD 與 native 編譯：已經到頂

懷疑過沒有 native 編譯。逐層檢查：

- Rust：`.cargo/config.toml` 有 `target-cpu=native`。`objdump -d` 有大量 `ymm` 與 `vgatherqpd`（沒開 native 只會有 SSE2 的 `addsd`）
- numpy：`np.show_runtime()` 顯示執行時選 `X86_V3`（AVX2 + FMA）
- OpenBLAS：同一輸出的 `architecture: Haswell`，即 AVX2 + FMA；Alder Lake 在 OpenBLAS 裡就對應這一級，自己編也一樣
- CPU 本身沒有 AVX-512

受記憶體限制的運算，SIMD 再寬也只是等得更久。

### 失敗的嘗試：`tilt_mean` 向量化

perf 看到 libm 的 scalar `exp` 佔主執行緒約 13%，來自 `brentq` 逐一 gamma 求根。改成 81 個 gamma 疊成矩陣的 safeguarded Newton，結果一致（差 6e-15），但只快 1 秒：`expit` 飽和時 Newton 退回中點、已收斂的 gamma 仍在每次迭代重算。收益不值得多一段手寫迴圈，已還原。

## 坑

- **`outputs/` 不在 git**：正式腳本會覆寫輸出，跑之前先 `cp -p` 備份，跑完才能比對。這次因此發現舊輸出是過期的：主基準的係數在它之後重新產生過
- **`cp` 不加 `-p` 會丟 mtime**，之後就無法用時間判斷輸出與上游的先後
- **背景執行時 `print` 有緩衝**，log 到結束才一次寫入；要看進度就 `flush=True` 或 `python -u`
- **`uv add -D` 被 rtk 改寫後失敗**，用 `--dev`
- **`uv add ./rust/<crate>` 會建成 workspace member**，與既有 crate 的 `[tool.uv.sources] path = ...` 寫法不一致，手動改回
- **rust-numpy 0.30 的 `.shape()` 在 `PyUntypedArrayMethods` trait 裡**，要另外 `use`
- **Pyright 找不到編譯的擴充模組**（`reportMissingImports`），實際可 import，可忽略
- **py-spy `--native` 大幅拖慢執行**；要時間就另外跑一次不剖析的
- **perf 的 CPU 樣本包含空轉**；判斷實際耗時要只看主執行緒

## 時間線

| 版本 | 耗時 | CPU |
|---|---|---|
| numpy 原版（py-spy 下） | 4:33 | 409% |
| Rust 修剪平均 | 1:12 | 1248% |
| + `OPENBLAS_NUM_THREADS=4` | 0:56 | 471% |
| + `bsum` 改 GEMM | 0:39 | 448% |

## 後續：likelihood.fit

`likelihood.fit` 是多支腳本共用的 module。先寫 `scripts/explore/likelihood_fit_bench.py` 拆開時間，以最大的一格（72 萬列、3 塊）為例：

- jit 編譯約 0.5 秒：`fit` 每次都建新的閉包，每次呼叫都重新編譯；最後一塊較短、形狀不同，再編一次。編譯時間與資料量無關，小樣本幾乎全是編譯
- `minimize` 6 次迭代，f、g、h 各呼叫 7 次；同一點的前向計算重複三遍
- 收斂後再算一次 H 與分數

改法全在 jax 內，不需要 Rust：jit 過的函式以 `ll` 為鍵快取（`t0`、`free` 改為輸入）；每塊補齊到 2 的冪次長度、mask 排除填充列；值、梯度、Hessian 在同一個 jit 內算出並記住最後一點。

驗證：與舊版逐一比對 logistic（三種樣本大小、全參數、受限、全固定）與有序模型，θ 的相對差 1e-12 內、V 3e-11 內。單次 fit 快 1.4–2.4 倍；`continuation_selection.py` 0:39 → 0:34，`rating_filter_error.py` 0:34 → 0:17。

又踩到一次過期的輸出：`rating_filter_error` 重跑後與舊輸出大幅不同，但連不經過 fit 的 `sample.csv` 也不同。換回舊版 `likelihood.py` 再跑一次，新舊版只差 1e-10，差異來自舊輸出早於主基準係數的更新。比對前先確認基準本身是新的：同一份程式碼新舊版各跑一次，比的才是改動本身。

## 後續：tilt_mean 改 Rust

fit 改完再量，`tilt_mean` 反而成了主執行緒最大的一段。先前矩陣化的 Newton 版沒有變快，是因為演算法每次迭代都對 81 × n 全部重算；問題本身是 81 個獨立的一維求根，每次求值掃一遍 n 列的 exp，正好是 rayon 的形態。移植 scipy 的 brentq（`scipy/optimize/Zeros/brentq.c`，容差相同）到 `rust/weighted_trim/`，各 gamma 平行：單次呼叫快約 12 倍，與 scipy 差 2e-13；`continuation_selection.py` 0:34 → 0:27。

驗證時又一個坑：模擬資料用常態的 y，參考實作的 brentq 自己就在 |gamma| = 40 時報「f(a) 與 f(b) 同號」。實際的 y 是得分殘差，落在 [-1, 1]；模擬資料的值域要對齊實際資料，否則測到的是參考實作的邊界條件。

## 後續：「空轉」其實是平行度不足

使用者觀察到 CPU 曲線仍有低谷。perf 的空轉符號（`blas_thread_server`、rayon 與 Eigen 的 steal、wait）合計不到幾個百分點，不是空轉。把 `perf script` 的時間戳每秒分桶、依葉節點的 DSO 標出主執行緒與其他執行緒在做什麼，畫成時間軸，才看出 LRT 那十秒只用三、四個核：主執行緒一半時間在等 jax，XLA 的執行緒加上 JIT 出的程式碼（`perf-PID.map`）只有兩三個核。

切小塊讓 jax 非同步派送多塊，期望 XLA 同時執行：反而核數更少、耗時不變，XLA 在 CPU 上是一塊接一塊執行。改把線性預測式的 logistic 交給 Rust（`rust/logit_fit/`，後改名 `rust/likelihood/`），手寫值、梯度、Hessian 與逐列分數，各列 rayon 平行：單次 fit 快 3–10 倍；`continuation_selection.py` 0:27 → 0:20，`rating_filter_error.py` 0:17 → 0:13。

坑：`perf script -F tid,time,sym,dso` 不印 sym 與 dso，要加 `ip`；只要葉節點時加 `--max-stack=1`。整支腳本比對時有一個模型的 p 值差到第六位：trust-exact 預設 `gtol=1e-5`，加總順序不同就停在容差內不同的點，不是 Rust 算錯。

## 後續：每個核都在跑、但都沒跑滿

使用者觀察到系統監視器上每個核都有負載、但沒有一個跑滿。以 20 ms 分桶看平行度：近四成時間只有主執行緒在跑（其他核在等），四成時間八核以上平行。每秒平均後就是「每個核都沒跑滿」。這是 Amdahl 的串列段落，不是空轉。

找串列段落：perf 的 callchain 接不回 Python，改用 `py-spy record --idle`，主執行緒的樣本依時間排序，每秒取最常出現的 Python 行，對照 perf 的時間軸。串列段落依序是 `dyad_meat`（`np.unique` 加 39 次 `bincount`）、`rng.multinomial` 產生重抽次數、每次呼叫 Rust 之前的 fancy indexing。

`dyad_meat` 移到 Rust 後，整支腳本**反而慢了一秒**，連跑三次都一樣。原因：每條執行緒配一份「最大玩家代碼 × k」的累加器，而玩家代碼是全庫代碼，遠大於樣本內的玩家數；二十份大陣列的配置、清零與相加吃掉了平行的收益。單元測試用連續的小代碼，測不到。改為與分群 meat 相同的平行排序分組（每局在兩位玩家下各計一次，等於把兩欄的鍵接起來分組），並在驗證腳本加入稀疏代碼：`continuation_selection.py` 0:20 → 0:17.5，`rating_filter_error.py` 0:12.6 → 0:11.4。

教訓：模擬資料要對齊實際資料的**結構**（代碼稀疏、值域），不只是大小；改完一定要量整支腳本，單元測試的加速不代表實際加速。

`rng.multinomial` 留著：平行產生需改用多條獨立亂數串流，bootstrap 抽到的樣本會變，統計上等價但輸出不同，由使用者決定。

