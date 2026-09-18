# 資料源清單

優先級與規則見 CLAUDE.md。已淘汰的資料源保留紀錄，不刪除。

| 優先級 | 資料源 | 需登入 | 取得方式 | 可自動取得 | 狀態 |
|---|---|---|---|---|---|
| 1 | Kaggle | MCP 已設定 | MCP | 是 | 可用 |
| 2 | 政府資料開放平臺 data.gov.tw | 否 | 直接 CSV 下載 | 待驗證 | 待驗證 |
| 2 | 內政部戶政司人口統計 | 否 | 網頁下載 | 待驗證 | 待驗證 |
| 3 | TCCIP | 是（需註冊） | 網頁下載 | 否 | 待驗證 |
| 3 | SEGIS 內政大數據 | 是 | 網頁下載，有下載限制 | 否 | 待驗證 |
| 4 | World Bank / OWID / 體育 / 遊戲數據 | 視來源 | API 或 CSV | 多數可 | 待驗證 |
| 5 | SRDA 學術調查研究資料庫 | 是（需申請） | 申請後下載 | 否 | 優先級最低 |

## 資料源細節

### Kaggle
- MCP server 已於 `.mcp.json` 設定
- 可直接搜尋 dataset、competition，並下載檔案

### 政府資料開放平臺 data.gov.tw
- https://data.gov.tw/
- 待驗證：是否有穩定的資料集 API

### 內政部戶政司人口統計
- https://www.ris.gov.tw/app/portal/346

### TCCIP
- https://tccip.ncdr.nat.gov.tw
- 需註冊帳號。若無法自動取得，需使用者手動下載並放入 `data/raw/`

### SEGIS 內政大數據模擬資料
- https://segis.moi.gov.tw/STATCloud/BigData
- 有下載限制。同上，可能需使用者手動操作

### SRDA
- https://srda.sinica.edu.tw/
- 需申請帳號，審核流程慢。優先級最低，其他來源都用盡再考慮
