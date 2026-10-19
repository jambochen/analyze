# 資料取得與溯源

## 一律用 CLI，不用 MCP

取得資料一律透過命令列，寫成 `scripts/fetch/` 下的腳本，不要用 MCP 工具。

理由：CLI 的每一步都留在腳本裡，別人照著跑得到同一份資料；MCP 呼叫留不下可重現的紀錄。

Lichess 以 curl 下載並以官方 sha256 驗證。Kaggle 資料用：

```bash
uvx --from kaggle kaggle datasets download -d <slug> -f <file> -p <dest> --unzip
```

Kaggle 憑證已設定完成：`~/.kaggle/credentials.json`（OAuth，帳號 `jambo2`）。憑證**絕不寫進專案任何檔案**。

## 資料溯源

每份下載的資料，在 `data/raw/<name>/SOURCE.md` 記錄：

- 資料源名稱與完整 URL
- 下載日期
- 授權條款與再利用限制
- 筆數與欄位數
- 欄位說明（名稱、型別、單位、編碼意義）
- 資料版本或更新頻率

政府開放資料經常改版且不保留歷史版本，下載當下即記錄，不可事後補。
