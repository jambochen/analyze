# 四個資料集的檢視結果

四份都下載跑過了，結論是都不建議採用，理由各不相同。
notebook 放 gist，每節對應的儲存格標在小標下方，數字都可以自己驗。

## EV 電池健康

https://www.kaggle.com/datasets/srisyra02/ev-battery-health-prediction-dataset-20k

> notebook：`EV 電池健康` 節

合成資料，而且合成規則可以反推。

```
capacity_loss_percent + battery_health_percent   平均=100.0  標準差=0.0000000000
corr(state_of_health, battery_health_percent)    =  0.9898
corr(thermal_health_score, thermal_runaway_risk) = -0.9868
```

標準差 0，代表這兩欄恆等於 100，是同一個數字的兩種寫法。

缺失率也不對：70 欄裡 67 欄落在 3.05%–4.94% 的窄帶，剩下 3 欄剛好 0%。
真實感測器不會這樣缺，這是先造完整資料再按固定比例挖洞的特徵。

結果是好幾個「預測目標」其實是別的欄位算出來的。在這上面建模跑出來的
是合成程式的參數，不是電池行為。頁面也沒說資料來源。

## 學生考試表現

https://www.kaggle.com/datasets/spscientist/students-performance-in-exams

> notebook：`組員提議的四個資料集` 節（結構表）

8 欄裡 5 個類別、3 個分數，沒有時間、空間或階層結構。

不知道在講誰——沒有學校、年份、國家，`race/ethnicity` 編成 group A 到 E
且無對照表。母體沒定義，推論就沒有對象。

下載 51 萬次，新穎度也沒有空間。

## 臺北市實價登錄

https://www.kaggle.com/datasets/chrischien17/taiwan-taipei-city-real-estate-transaction-records

> notebook：`臺北市實價登錄` 節

加工過的二手版本，比內政部原始檔差。

時間跨度名義上 1986–2020，實際上 2019 年佔 93.3%，加上 2020 年共 98.3%，
剩下 8 個年度合計 103 筆。

兩個欄位名稱與內容對不上：

```
building_age （叫屋齡）  範圍 1907–2020  ← 實際是西元年
complete_year（叫建成年）範圍 0–108      ← 實際是民國年
```

行政區英譯不一致（`Beitou`、`shihlin` 無 District，`Zhongshan Area` 用 Area），
`urban_land_use` 有 `Quotient`、`work` 這類機翻殘留，`main_use` 裡
`Other` 與 `Others` 並存。原始中文編碼已遺失。

### 我先前查過內政部原始檔

追到內政部不動產成交案件實際資訊資料供應
系統，實測確認**季度批次可以直接下載，不用申請也不用 API key**：

```
歷史季度批次
https://plvr.land.moi.gov.tw/DownloadSeason?season={季別}&type=zip&fileName=lvr_landcsv.zip
季別格式：民國年 + S + 季，如 111S1、113S2、114S1
實測三個季別皆回 HTTP 200，檔案 13–19 MB
```

（注意本期批次只有 10 天份，買賣僅 7,185 筆，**必須用季度批次**。）

兩者對比：

| | Kaggle 這份 | 內政部原始檔 |
|---|---|---|
| 筆數 | 5,910 | 單季 12.8 萬，全期推估 600–700 萬 |
| 欄位 | 24（翻譯過） | **33**（主檔）+ 三個關聯子表 |
| 時間 | 2019 年佔 93.3% | 民國 101 年 8 月至今，約 14 年 |
| 範圍 | 只有台北市 | 22 縣市 |
| 市場別 | 只有買賣 | **買賣 + 預售屋 + 租賃** |
| 編碼 | 中文原始編碼已遺失 | 完整 |

原始檔多出來的欄位包括**主建物面積、附屬建物面積、陽台面積、電梯、
移轉層次、總樓層數**——這些都是 hedonic 模型的標準特徵變數，
Kaggle 那份全都沒有。另外還有 `_build` / `_land` / `_park` 三個子表
可經編號關聯。

所以如果要做房價，結論很明確：**用內政部原始檔，不要用 Kaggle 這份。**
代價是要自己寫腳本串接約 54 個季度檔，每季 225 個檔案。

### 不過我還是不建議做這個方向

即使用原始檔，這個題目仍有兩個問題：

**新穎度。** 房地產是熱門主題，Kaggle 上已有大量房價分析。

**我們沒有資訊優勢。** 房價預測的使用者是買賣雙方、仲介、銀行——
他們手上的資訊遠多於公開實價登錄（實際議價過程、物件狀況、貸款條件）。
我們算不出他們不知道的事。

原始檔本身的限制也記一下：**申報偏誤**（親友交易、含增建會使價格偏離
行情，備註欄有部分註記但不完全可靠）、**地址非座標**（要做空間分析得
自己地理編碼）、早期年度（民國 101–105 年）欄位定義可能與近年不同。

## AI 對就業薪資

https://www.kaggle.com/datasets/debayank2024/ai-impact-on-jobs-and-salaries-2020-2026

> notebook：`AI 對就業薪資` 節、`復現第三方分析` 節

這份表面品質最好——71,913 列、17 欄、幾乎沒有 null、附了 data dictionary。
問題不在資料髒，而在**它不含回答自己標題所需的變數**。

**一、標題承諾的東西不在檔案裡。**

資料集叫 "AI Impact on Jobs and Salaries"，描述引用了 ILO 的生成式 AI
職業曝險指數。但 17 個欄位全部列出來，沒有任何一欄是曝險度：

```
work_year, experience_level, employment_type, job_title, salary,
salary_currency, salary_in_usd, employee_residence, remote_ratio,
company_location, company_size, experience_level_label,
employment_type_label, work_mode, salary_outlier_flag,
role_family, isco_group_hint
```

附的 data dictionary 其實沒有誤導，它寫 `isco_group_hint` 是
"for joining **against** AI-exposure data"——明指這是拿來**對外接合**的鍵，
ILO 那份是**待接合的外部來源**，不是已併入的欄位。誤導來自 Kaggle 的
標題與描述。

實際拿到的東西是 ai-jobs.net 的薪資表加兩個分類欄。要做 AI 曝險的題目，
得自己去 ILO 取指數再接，那這份資料就沒什麼加值了。

**二、就算接上曝險分數，也算不出「影響」。**

資料是職缺薪資的自我回報快照：沒有個人 ID（同一人跨年追蹤不了）、
沒有公司 ID、也沒有未受 AI 衝擊的對照組（全部都是 AI／資料類職缺）。

缺了這些，能做的上限是「曝險度不同的職業，薪資差多少」——那是橫斷面
關聯，不是「AI 造成了什麼」。而要回答前者，官方薪資調查的資料品質
遠比自願填報好。

**三、連橫斷面關聯都難算，因為跨年比較不成立。**

```
work_year   筆數     remote_ratio 均值
2020          75          62.0
2021         214          69.6
2022       1,114          58.5
2023       4,538          36.8
2024      27,690          22.5
2025      38,282          23.3
```

2024+2025 佔了 91.8%，標題寫的 2026 年一筆都沒有。

`remote_ratio` 那條曲線看起來完美呈現「疫情遠端高峰後回落」。但
2020–2021 合計才 289 筆，而且早期會去 ai-jobs.net 填報的人本來就偏向
遠端優先的新創。**樣本組成的變化和真實趨勢完全混在一起，分不開。**

### 這個問題躲不掉：票數最高的 notebook 也中了

https://www.kaggle.com/code/sasakitetsuya/ai-salaries-and-the-vanishing-middle

這份資料底下票數最高的分析，方法做得相當完整——事前假說、OLS 控制
年資／國家／公司規模／工作型態、`bucket × year` 交互項、兩組敏感度
分析，作者最後還否定了自己原假說的兩條腿。他也獨立證實了第一點，
notebook 裡直接寫 "the exposure columns are **not present**"。

**但他最強的那條結論站不住**，撞的正是第三點。

他的結論是「管理職 2023→2025 調整後薪資 +22%」。復現他的樣本後：

```
Middle          2023  n= 3,756
Management      2023  n=   140     ← +22% 是從這個基期算的
People-facing   2023  n=    11
```

而且那一組裝的人換了：

```
2023 (n=140)  ：Data Manager 58、Head of Data 21、Data Analytics Manager 19
2025 (n=4,400)：Manager 2,238、Director 472、Data Manager 326
```

2023 年是「資料部門的主管」，2025 年主體變成泛用的 `Manager`／
`Director`，組內 Executive 比例從 24.3% 掉到 13.3%。
不是同一群人薪水在漲，是這組裝的人換了。他有控制年資，
但控不住職稱本身的語意在漂移。

同樣的問題也影響他其他幾條結論——Middle 組從 3,756 筆長到 25,127 筆，
組成一樣在變。

**這就是我不建議用這份資料的主因**：有假說、有控制變數、有敏感度分析、
作者還願意否定自己的分析，仍然躲不掉樣本組成的問題。
這不是分析技巧能補救的，得在選題階段就篩掉。

---

## 我做了哪些

- 四份都下載實測，溯源與完整欄位說明我留了一份紀錄
- 復現了 sasakitetsuya 的 notebook，確認他主結論的樣本問題
- 房價方向另外追到內政部原始檔，下載路徑已試通

實價登錄那條路徑如果有人想接，可以直接用。

## 幾個踩過的坑，給大家參考

- `null_count` 是 0 不代表沒缺失。有些調查資料把「不知道／拒答」編成
  碼 9，是有效整數。我之前看過一欄 36% 是無效回答但 `null_count` 回報 0
- 不要假設叫 id 的欄位全域唯一。碰過只在郡內唯一的，單欄去重低估 5.5 倍
- 聚合前檢查字串拼法。`JOSEPH R BIDEN` 與 `JOSEPH R. BIDEN` 差一個句點，
  直接 group_by 低估 17%
- 欄位名稱不等於語意（上面實價登錄的屋齡存西元年就是一例）
- 欄位相加恆為常數、相關係數接近 ±1，是合成資料的指紋
- 跨年比較前先分年看各組樣本數與組內構成，不要只看總列數
- polars 的瓶頸是分組數不是列數。1.66 億列全掃 0.1 GB，
  但 4,620 萬組的 group_by 撐爆 16 GB
