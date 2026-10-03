#set document(title: "美國航班天氣延誤申報的歸因異常稽核：分析提案")
#set page(paper: "a4", margin: (x: 2.6cm, y: 2.4cm), numbering: "1")
#set text(font: ("Noto Sans CJK TC",), lang: "zh", region: "tw", size: 11pt)
#set par(justify: true, leading: 0.85em, first-line-indent: 0em)
#show raw: set text(font: ("DejaVu Sans Mono", "Noto Sans CJK TC"), size: 9.5pt)
#set heading(numbering: "1.1")
#show heading: it => block(above: 1.4em, below: 0.7em)[#it]
#show heading.where(level: 1): it => block(above: 1.8em, below: 0.9em)[#it]
#set figure(gap: 0.8em)
#show ref: it => {
  let el = it.element
  if el != none and el.func() == heading {
    let n = numbering(el.numbering, ..counter(heading).at(el.location()))
    link(el.location(), if el.level == 1 [第 #n 章] else [第 #n 節])
  } else { it }
}

#let rows(name) = csv("../outputs/tables/" + name + ".csv").slice(1)
#let pick(tbl, key, col: 1) = float(tbl.find(r => r.at(0) == key).at(col))
#let pct(x, d: 0) = str(calc.round(x * 100, digits: d)) + "%"
#show figure.caption: set text(size: 9.5pt)

#align(center)[
  #text(size: 17pt, weight: "bold")[美國航班天氣延誤申報的歸因異常稽核]

  #v(0.3em)
  #text(size: 12pt)[以實測氣象為基準建立監理優先清單]

  #v(0.6em)
  #text(size: 10.5pt)[統計研究所　學期分析報告　提案]
]

#v(1.2em)

#block(fill: rgb("#f7fafc"), inset: 1em, radius: 3pt, width: 100%)[
  #text(size: 10pt)[
    *摘要。* 美國交通部要求航空公司為達門檻的延誤航班申報成因，其中天氣屬
    免責事由，不計入公司自身的準點績效。這些欄位由航空公司自行申報，監理機關
    不例行查核，交通部監察長辦公室亦曾對其一致性提出疑慮，但未指出偏誤方向，
    也未提供替代的查核方法。本提案以機場實測氣象為外部基準，在同一機場、同一
    月份、同樣天氣條件下比較各航空公司的天氣歸因行為，量化其偏離同儕的方向與
    程度，產出供監理單位使用的查核優先清單。分析不預設偏誤方向，雙向偏離皆
    列入。交付物為排序清單而非預測模型，主張止於「值得進一步查證」。
  ]
]

= 問題 <ch:problem>

== 自報欄位的可信度缺口

美國運輸統計局（Bureau of Transportation Statistics，BTS）要求到達延誤達
門檻的航班申報延誤成因，分為五個欄位。申報方式是拆分鐘數，不是勾選成因：
每一欄填入歸因於該成因的延誤分鐘數，五欄加總等於該航班的到達延誤。其中 `carrier` 為航空公司自身責任，
`weather`（天氣）、`NAS`（航管與機場限制）、`late-aircraft`（前段航班晚到）
則屬對外歸因。

這些欄位由航空公司自行填報，監理機關不例行查核其正確性。美國交通部監察長
辦公室（Office of Inspector General，DOT OIG）曾對申報的一致性提出疑慮，但
沒有人指出偏誤的方向，也沒有人提出替代的查核方法。目前只知道欄位不能盡信，
不知道哪些公司有問題、問題有多大，本研究處理的就是這一點。

== 歸因具有制度後果

對外歸因的三個欄位皆屬免責事由，不計入航空公司自身的準點績效，也影響補償
責任的認定，申報行為因此有系統性偏離的動機。有動機不代表偏誤必然朝某一方向
發生，本研究不預設方向，理由見#ref(<sec:direction>)。

== 外部基準的接合 <sec:baseline>

判斷天氣歸因是否異常，須把航班紀錄與氣象觀測接起來。兩套資料都是公眾領域：
BTS 提供航班紀錄，美國國家海洋暨大氣總署（NOAA）的整合地表資料庫
（Integrated Surface Database，ISD）提供逐時氣象觀測。但 BTS 以機場代碼
標識地點，ISD 以氣象測站代碼標識，兩者之間沒有官方維護的對照表。接合是
工程量的問題，與資料保密無關。

== 為什麼是這份資料

本研究採用的資料集已把兩者接起來：發布者建置了機場對 ISD 測站的對照表，
並逐班以資訊截止點做 as-of join，把起飛前的實測氣象掛到每一班航班上。

自行接合完全可行，上游都是公開資料，這份資料集做的是彙編而非取得獨有材料。
不自建的理由是成本與稽核目的無關。其一，自建等同重做測站對照、逐班 as-of
join、METAR 降水的跨報告去重、前段航班的視窗閘控。其二，本資料集的比對方法
與測站距離逐列可查（見#ref(<sec:station>)），自建不會更透明。其三，時間範圍
與申報門檻這兩項邊界來自上游，換一條接合路徑也不會改變。

二次整理的代價是彙編過程可能污染被稽核的欄位，這一點以#ref(<sec:verify>)的上游
核對處理。

= 資料 <ch:data>

== 資料集的來源與邊界

#figure(
  table(
    columns: (auto, 1fr),
    stroke: none,
    align: (left, left),
    row-gutter: 0.45em,
    [名稱], [US Flights Joined to Observed Airport Weather],
    [發布者], [Arimancy LLC（Kaggle）],
    [識別碼], [`arimancy/observed-weather-us-flights`],
    [授權], [CC BY 4.0],
    [上游], [US DOT BTS 航班資料、NOAA NCEI ISD 逐時氣象、OurAirports],
    [分析單位], [一列 = 一班已執飛的美國國內線航班],
  ),
  caption: [資料集的基本身分。上游三者皆為政府的公眾領域資料，發布者所做的是彙編。],
)

NOAA 已停止 ISD 的全球逐時匯入，面板停在最後一個完整氣象月，之後不會再延伸。

資料集含 2019 至 2025 共七個年度檔，其中 2025 年檔只涵蓋一至七月。天氣
延誤有強烈的季節性（冬季風雪、夏季雷雨），缺少後五個月的年度不能與完整
年度並列進入固定效應，故主要分析只取 2019 至 2024 六個完整年度，2025 年檔
僅在與其他年度的同月比較時使用。

== 時間與地理覆蓋

#figure(
  image("../outputs/figures/data_monthly_volume.png", width: 88%),
  caption: [逐月航班量。上為年度檔全部航班，下為稽核樣本（達申報門檻且遭遇惡劣天氣，界定見#ref(<sec:sample>)）。],
)

主要分析的六個完整年度共 #rows("monthly_volume").len() 個月。全部航班的疫情斷裂由#ref(<sec:era>)處理。
稽核樣本由天氣決定，冬季尖峰每年重複，疫情初期幾近於零。季節性是樣本本身
的結構，固定效應須細到月份才能吸收。

#let geo = csv("../outputs/tables/airport_geo.csv", row-type: dictionary)
#let offshore = geo.filter(g => {
  let lon = float(g.longitude)
  let lat = float(g.latitude)
  not (lon >= -130 and lon <= -60 and lat >= 23 and lat <= 50)
})
#let geo-share = offshore.map(g => int(g.n_all)).sum() / geo.map(g => int(g.n_all)).sum()
#figure(
  image("../outputs/figures/data_airport_map.png", width: 88%),
  caption: [美國本土的起點機場。點大小為六年全部航班數，顏色為該機場航班進入稽核樣本的比例。阿拉斯加、夏威夷與屬地的 #offshore.len() 座機場未繪出，其航班佔全部的 #pct(geo-share, d: 1)。],
)

本土沒有整片區域缺席。比例較高者多為東北部與五大湖一帶的中小機場，與已知
的天氣地理一致；這些機場班次少，比例不穩，此圖只用於確認覆蓋。

== 申報行為的時期變動 <sec:trend>

#figure(
  image("../outputs/figures/data_yearly_trend.png", width: 92%),
  caption: [逐年的三個申報指標（上）與到達延誤的四分位（下），皆取稽核樣本全量。],
)

申報指標在疫情期間抬升、其後回落，單一年度的橫斷面看不到，這是納入六年的
理由。同期延誤的四分位反而左移，兩者方向不一致，該段的抬升可能來自航班組成
的重組，未必是申報行為改變，故#ref(<sec:era>)不對該段單獨作行為解讀。

== 一筆紀錄的資訊構成 <sec:record>

#figure(
  table(
    columns: (auto, 1fr),
    stroke: none,
    align: (left, left),
    row-gutter: 0.4em,
    [班次], [`operating_carrier`、`flight_number`、`tail_number`],
    [航段], [`origin_iata`、`dest_iata`、`distance_miles`],
    [時刻], [`sched_dep_utc`、`sched_arr_utc`、`info_cutoff_utc`],
    [壅塞], [`origin_sched_departures_this_hour`、同日同機場班次],
    [前段], [`prior_leg_arr_delay_min`、`prior_leg_turnaround_min`],
    [氣象], [兩端機場各一組 `wx_`，含當下觀測與起飛前數個時點],
    [結果], [`arr_delay_min`、`cancelled`、`diverted`],
    [自報], [`reported_` 五欄],
  ),
  caption: [單一航班所攜帶的資訊。唯一鍵為班次、航段與當地飛行日，樣本內無重複列。],
)

結果與自報以外的欄位在航班排定時即已知，可作控制變數。取消與轉降的航班帶
旗標留在檔案中，本分析只取已執飛者。

`prior_leg_` 另有用途。上一航段晚到會消耗周轉緩衝，本航班的延誤便有明確的
機械成因，應記入 `late-aircraft`。這是外生於本班申報的衝擊，可作為同儕比較
之外的第二個判準，但只適用於有前一航段的航班（當日首航段的處理見@tbl:missing）。

== 欄位的前綴系統與洩漏防護 <sec:oracle>

資料集以前綴標示欄位相對於資訊截止點（`info_cutoff_utc`，即 UTC 排定起飛
時刻）的時序位置。

#figure(
  table(
    columns: (auto, 1fr),
    stroke: (x, y) => if y == 0 { (bottom: 0.7pt) },
    align: (left, left),
    row-gutter: 0.4em,
    [前綴], [語意],
    [`wx_`], [資訊截止點之前或當下觀測到的氣象],
    [`oracle_`], [該航班自身事件時刻所在小時的氣象],
    [`reported_`], [航空公司自我回報的延誤成因],
    [（無前綴）], [班表、機場、公司、壅塞度、日曆、前段航班],
  ),
  caption: [欄位的四個類別。前綴系統使洩漏防護得以整組執行。],
)

`oracle_` 與截止點沒有保證的時序關係：提早起飛的航班，事件時刻可能早於
截止點；該小時的例行報告也可能晚於事件本身。用於預測延誤即構成洩漏。

發布者明言 `oracle_` 的用途為回溯與因果分析。本研究估計歸因傾向，屬回溯
用途，故保留起飛時刻與排定到達時刻兩族。實際到達時刻一族的觀測時點取決於
航班何時落地，延誤越久取到的時點越晚，與延誤本身機械相關，以前綴整族移除
（逐欄挑選難免遺漏）。

保留的理由是申報人事後填表，知道航班實際遭遇的天氣，只用 `wx_` 等於以更少
的資訊判斷申報是否恰當；且兩者的目的地天氣類別在稽核樣本中有相當比例
不一致。公司的主效應以 `wx_`、`oracle_`、兩組並用三種設定並列，不單獨以
`oracle_` 為主要設定。惡劣天氣的樣本仍以 `wx_` 判定，使三種設定落在同一批
航班上。

`oracle_` 的另一個用法是把實際天氣相對起飛前的變化當成判準：天氣確實惡化時，
公司是否更常歸因天氣。這是公司自己與自己比，不受航網結構干擾，與
`prior_leg_` 同屬第二個判準。

== 結果變數的分布

#figure(
  image("../outputs/figures/slides_delay_dist.png", width: 82%),
  caption: [到達延誤的分布與申報門檻（橘線）。取六年年度檔全部航班，橫軸截於 −60 至 240 分。門檻以右的航班才有申報成因的義務。],
)

班表留有緩衝，多數航班提早或準時抵達，門檻切在右尾，稽核樣本從一開始就只
是右尾的一小塊，`reported_` 的高缺失率也由此而來。到達延誤只用於界定樣本與
描述；推論層建模的是延誤分鐘在各成因欄位之間的分配（見#ref(<sec:simplex>)）。

== 被稽核的對象

`reported_` 五欄是被稽核的對象，不得作為解釋變數：拿自報的天氣延誤分鐘數
解釋延誤本身是循環論證。發布者稱這五欄「self-reported and unreliable」，並以
實測氣象取代它們做預測；本研究反過來把它們當作研究對象，新穎之處在此。

發布者的品質保證（截止點閘控、洩漏測試）為預測用途設計，未驗證申報欄位能否
用於行為稽核，此缺口由#ref(<sec:verify>)的上游核對補上。

== 結構性缺失

`reported_` 五欄的缺失完全由申報門檻決定，故缺失率相同。這種缺失不是隨機的，
後果有三：

+ 稽核只能在達門檻的航班上進行（見#ref(<sec:sample>)）。
+ 不能填零。未達門檻的航班沒有申報義務，填零等於宣稱它們回報零分鐘天氣
  延誤，會憑空製造大量的零，把所有係數往下拉。
+ 研究問題收斂為「該報的時候怎麼報」，不回答「所有航班的延誤成因為何」。

#figure(
  image("../outputs/figures/slides_missingness.png", width: 72%),
  caption: [各欄位的缺失率，取六年年度檔全部航班（申報門檻篩選之前）。橘色為 `reported_` 五欄，缺失率完全相同。],
)

其餘欄位的缺失成因不同，處理也不同。

#let miss = rows("missingness")

#figure(
  table(
    columns: (auto, 1fr, 1fr),
    stroke: (x, y) => if y == 0 { (bottom: 0.7pt) },
    align: (left, left, left),
    row-gutter: 0.4em,
    [類型], [欄位（缺失率）], [處理],
    [不符條件不填報], [`reported_` 五欄（皆 #pct(pick(miss, "reported_weather_delay_min"))）], [不插補，用以界定樣本],
    [不符條件不填報], [雪深 #pct(pick(miss, "wx_origin_snow_depth_cm"), d: 1)、陣風 #pct(pick(miss, "wx_origin_wind_gust_ms"))], [填 0，另立 `_absent` 指示變數],
    [物理上無定義], [風向], [轉為正弦與餘弦分量],
    [測站未觀測], [溫度、能見度、雲冪（皆 ≤ #pct(calc.max(..("wx_origin_temp_c", "wx_origin_visibility_m", "wx_origin_ceiling_m").map(k => pick(miss, k))), d: 1)）], [不處理，保留 null],
    [不適用], [`prior_leg_` 一組（當日首航段）], [填 0，另立 `prior_leg_available`],
  ),
  caption: [缺失的五種成因與對應處理。],
) <tbl:missing>

雪深與陣風無事可報即不填，填零才是真實語意；另立指示變數，是因為「測站不
觀測」與「當日無雪」在資料上形態相同。`prior_leg_` 同理：當日首航段填零表示
「不適用」，與「上一段準點到達」同樣是零，須靠 `prior_leg_available` 分開。

風向的三百五十九度與一度實際相差兩度，數值上卻相差三百五十八。正弦餘弦
分量消除此斷點，且靜風時兩分量皆為零，正好對應無風向。

== 樣本界定 <sec:sample>

#figure(
  image("../outputs/figures/slides_sample_funnel.png", width: 78%),
  caption: [樣本界定的兩道篩選，橫軸為對數刻度。],
)

#let funnel = rows("sample_funnel")
兩道篩選都不是資料清理的選擇。申報門檻（留下 #pct(float(funnel.at(1).at(4)))）
來自申報制度的定義域；惡劣天氣（再留下 #pct(float(funnel.at(2).at(4)))）來自
研究問題：天氣晴朗時討論天氣歸因並無意義。結論因此只及於這些航班，不代表
所有航班。

惡劣天氣取起點或目的地為 IFR 或 LIFR（依能見度與雲冪界定的飛行規則分類），
實際以單端惡劣為大宗。起點惡劣影響起飛、目的地惡劣影響落地，歸因空間不同，
兩端氣象因此分開入模，不合併為單一指標。

== 氣象基準的品質 <sec:station>

#figure(
  image("../outputs/figures/slides_station_match.png", width: 80%),
  caption: [測站比對的距離分布與方法構成。統計單位為機場，取資料集的機場對照表全量，非逐班加權。],
)

對照表是本資料集對#ref(<sec:baseline>)兩套編碼的接法。比對順序為 ICAO 呼號
精確比對、半徑內最近測站、人工；每列記錄 `station_match_method` 與
`station_distance_km`，距離過遠或非呼號比對者另有旗標。

絕大多數機場為呼號精確比對，距離中位數不到一公里，符合測站多設於機場內的
事實。「僅保留呼號比對的機場」的敏感度分析仍會執行，但依此分布判斷影響有限，
氣象基準不是主要風險。

== 稽核樣本的規模與公司構成

#let cells = rows("cell_carriers")
#let ck(k, d) = str(calc.round(pick(cells, k), digits: d))
#let thou(x) = {
  let c = x.clusters()
  c.enumerate().map(((i, d)) => if i > 0 and calc.rem(c.len() - i, 3) == 0 { "," + d } else { d }).join()
}

#grid(
  columns: (0.9fr, 1.1fr),
  gutter: 1.4em,
  [
    #table(
      columns: (auto, auto),
      stroke: none,
      align: (left, right),
      row-gutter: 0.4em,
      ..rows("sample_dims").zip(("班", "家", "座", "架", "格", "欄"))
        .map(((r, u)) => (r.at(0), thou(r.at(1)) + " " + u)).flatten(),
    )
  ],
  [
    格子多不等於可識別。公司效應來自同一格子內不同公司的比較，只有格子內
    有兩家以上公司才提供資訊。每格平均 #ck("每格平均公司數", 1) 家公司、
    中位數 #ck("每格公司數中位數", 0) 家；有兩家以上公司的格子佔
    #pct(pick(cells, "兩家以上格子佔比"))，卻涵蓋 #pct(pick(cells, "兩家以上格子的航班佔比")) 的航班。

    單一公司的識別力見下表的格子數。格子少的公司，係數與所在機場的效應較
    難分離；總樣本量足夠，受限的是這些公司的係數雜訊。
  ],
)

#let cb = csv("../outputs/tables/carrier_baseline.csv").slice(1)
#let wan(x) = str(calc.round(float(x) / 1e4, digits: 1))
#figure(
  text(size: 9pt, table(
    columns: (auto, 1fr, auto, auto, auto, auto, auto, auto, auto),
    stroke: (x, y) => if y == 0 { (bottom: 0.7pt) },
    align: (left, left, left, left, right, right, right, right, right),
    row-gutter: 0.1em,
    [代碼], [名稱], [類型], [期間], [全部（萬班）], [樣本（萬班）], [起點機場], [格子], [延誤中位數],
    ..cb.map(r => (
      r.at(0), r.at(1).replace(regex(" (Inc\.|Co\.|LLC)$"), ""), if r.at(2) == "true" [區域] else [主線],
      if r.at(3) == r.at(4) { r.at(3) } else { r.at(3).slice(2) + "–" + r.at(4).slice(2) },
      wan(r.at(5)), wan(r.at(6)), r.at(7), r.at(8), str(int(float(r.at(9)))),
    )).flatten(),
  )),
  caption: [#cb.len() 家執飛公司的基本盤。全部為六年年度檔班次；樣本、起點機場、格子（機場 × 年月）與延誤中位數（分鐘）皆取稽核樣本。期間為公司在稽核樣本中出現的年份。],
)

公司之間的規模差距跨越一個數量級以上，另有三個結構特徵影響後續設計：

- 主線與區域航空並存。區域航空以大型公司的品牌執飛，飛的機場多、每個
  機場的班次少。清單以執飛公司為單位，行銷品牌作為對照。
- 部分公司只在部分年度出現於資料中，時期比較只能在跨期皆存在的公司上
  進行。
- 少數公司的航網高度集中於天氣少變的地區。這類公司進入稽核樣本的比例
  偏低，格子數也少，係數最依賴收縮。

#figure(
  image("../outputs/figures/slides_concentration.png", width: 80%),
  caption: [機場與執飛公司的班次集中程度，取六年年度檔全部航班，非稽核樣本。],
)

班次高度集中，小公司在多數格子中班次少、係數不穩，這是#ref(<sec:ident>)排序前須先收縮的理由。

= 分析設計 <ch:design>

== 核心研究問題

#block(fill: rgb("#ebf8ff"), inset: 1em, radius: 3pt, width: 100%)[
  以實際觀測氣象為基準，哪些航空公司的天氣歸因行為系統性偏離同儕？偏離的
  方向與程度能否量化並排序，作為監理查核的優先順序？
]

資料沒有記錄真實成因（見#ref(<ch:limits>)），本研究不判定造假，只做相對比較：
同樣的天氣、機場與月份下，哪一家公司的歸因行為與同儕不同。交付物是以執飛
公司為單位的監理優先清單，供 DOT 與 FAA 使用，走「具體可行的建議」路線。

== 不預設偏誤方向 <sec:direction>

資料源與 DOT OIG 只說自報欄位「不可信」，未指方向。先定方向再找證據是先射箭
再畫靶，故偏高與偏低皆列入清單。

#figure(
  table(
    columns: (auto, 1fr),
    stroke: (x, y) => if y == 0 { (bottom: 0.7pt) },
    align: (left, left),
    row-gutter: 0.45em,
    [偏離方向], [需要解釋的問題],
    [過度歸因天氣], [是否把自身責任轉嫁給免責事由],
    [歸因顯著偏低], [延誤究竟記到了哪裡],
  ),
  caption: [雙向偏離皆列入清單，各自需要不同的解釋。],
)

== 描述層：三個比例及其限制

#figure(
  image("../outputs/figures/slides_outcome_dist.png", width: 95%),
  caption: [三個描述性比例。Y1 為二元，故以公司間的原始比例呈現；Y2 與 Y3 畫分布。歸因邊界的縱軸取對數，否則零的質量點會壓平其餘。此處的公司順序為未經控制的原始比例，不是本研究的清單排序；Y1 與前提檢驗同一口徑，樣本內班次過少的公司不繪出。],
)

#figure(
  table(
    columns: (auto, auto, 1fr, auto),
    stroke: (x, y) => if y == 0 { (bottom: 0.7pt) },
    align: (left, left, left, left),
    row-gutter: 0.42em,
    [短名], [定義], [回答的問題], [承擔],
    [是否歸因], [`weather > 0`], [有沒有把延誤掛上天氣], [描述],
    [歸因強度], [天氣延誤分鐘數 / 總延誤], [掛上時掛了多少], [描述],
    [歸因邊界], [天氣 / (天氣 + NAS)], [兩個免責事由之間偏向哪邊], [描述],
    [單體係數], [公司 × 成因交互項], [延誤被推向哪一欄], [識別],
  ),
  caption: [描述層的三個比例與推論層的單體係數，及其各自承擔的角色。],
)

設歸因邊界，是因為 `NAS` 本身包含天氣造成的流量管制，把天氣延誤記入 `NAS`
完全合法，對準點績效的影響也相同。只看是否歸因天氣，會把記帳習慣的差異誤判
為申報異常；以兩欄合為分母即消去這項差異。

三者形狀不同，不能共用一套模型。歸因強度在 1.0 處有質量點（認定天氣即整段
掛上），beta 迴歸在端點無定義而不可用；歸因邊界兩端皆有質量點，多數航班全記
`NAS` 或全記天氣。判斷依據是這些結構特徵，不用常態性檢定：十萬列以上的樣本
下檢定力趨近於一，「顯著故非常態」只反映樣本量大，也測不出質量點與分母約束。
模型假設以#ref(<sec:deliver>)的穩健性檢查驗證。

這三個比例只承擔描述，不承擔識別，問題出在定義：

- 歸因強度的分母是被約束的量。各成因分鐘數加總等於到達延誤，歸因強度因此
  是單體的一個邊際投影，卻被當成獨立比例建模。1.0 處的質量點正是單體的頂點。
- 歸因強度限於已歸因天氣的航班，構成 selection on the outcome。當公司間
  的歸因傾向不同，比較便落在不同的選擇樣本上。
- 歸因邊界的分母是內生的。「天氣加 `NAS`」這個組合取決於公司如何填報。
  且只取兩類會丟失 `carrier` 與 `late-aircraft` 的資訊，看不出延誤是從
  哪一個事由移轉而來。

描述層仍保留：原始差異是報告必須交代的內容，兩層若指向同一批公司即為穩健性
證據。清單的數字只從推論層產出。

== 推論層：歸因單體 <sec:simplex>

五欄加總等於到達延誤，此約束在樣本內嚴格成立，申報因此是一個單體。逐欄各自
建模會違反它：某一欄多記的分鐘必然是其他欄少記的。

推論層採組合資料框架，問的是「相對於同一格子的同儕，這家公司把延誤多推向
哪一欄」。以唯一可歸責的 `carrier` 欄為基準，其餘係數便有「相對於承認自身
責任」的方向。歸因邊界在此成為兩個係數的對比，內生分母的問題隨之消失。

估計法須滿足兩個條件：零是結構性的，不能用要求各維皆正的對數比轉換；格子
數以萬計，須能吸收高維固定效應。

#figure(
  table(
    columns: (auto, 1fr),
    stroke: (x, y) => if y == 0 { (bottom: 0.7pt) },
    align: (left, left),
    row-gutter: 0.45em,
    [設定], [理由],
    [基準取 `carrier` 欄], [唯一可歸責，係數即「相對於承認自身責任，多推向哪個外部事由」],
    [不用 ALR／ILR], [對數比要求每一維皆為正，而零是結構性的],
    [Poisson 對數線性], [與多項 logit 等價，且上萬格固定效應可以去均值吸收],
  ),
  caption: [推論層的三項設定與各自的理由。],
)

零是結構性的：該成因不成立即不填報，不是微量被捨去。要湊到各欄皆正必須刪去
小樣本單元，而小公司正是清單關心的對象，對數比路線會系統性刪去交付物的目標。
計數模型沒有這個問題，零是合法的觀測值。

不直接實作多項 logit，是因為非線性模型無法以去均值吸收上萬個固定效應。改用
兩者的等價關係：以單元 × 成因的分鐘數為計數，單元固定效應吸收該公司在該格子
的申報總量，格子 × 成因固定效應吸收當地的基準成因分配，公司 × 成因的交互項
即多項 logit 的係數。

氣象在單元內不隨成因變動，主效應會被單元固定效應完全吸收，故須與成因交互才
進得了模型；其係數讀作「天氣越惡劣時，延誤多記入該欄」。

== 識別策略 <sec:ident>

直接把公司放入固定效應模型，係數即公司效應，並附標準誤；先建預測模型再看
殘差則得不到不確定性的量度。

- *條件化於機場 × 年月。* 跑道配置、除冰能力與當地航管慣例因此被吸收。
- *樣本與特徵。* 樣本見#ref(<sec:sample>)，特徵限於 `wx_` 與排定時已知的
  欄位（#ref(<sec:oracle>)），結構性填零的欄位連同指示變數入模。
- *標準誤雙向聚類於機場 × 年月與機尾號。* 兩者在三個樣本口徑下皆非巢狀，
  依據見#ref(<sec:evidence>)。
- *排序前以經驗貝氏收縮。* 收縮量由公司間真實差異與抽樣誤差之比決定；若
  差異未超出雜訊，矩估計會把所有公司壓回均值，排序隨之消失。收縮因此同時
  檢驗差異是否為真。
- *清單是排序而非檢定。* 「幾項顯著」的粗排序用到門檻，故以
  Benjamini–Hochberg 控制偽發現率。

== 樣本範圍與時期切點 <sec:era>

#figure(
  image("../outputs/figures/slides_covid_volume.png", width: 82%),
  caption: [航班量相對疫情前同月的比例，與暫定的兩個切點。],
)

六年全數納入，以三段時段指示變數處理疫情斷裂。裁切會損失兩年資料，且仍須
交代切點，並無節省。切點依 BTS 航班量的實際崩跌與回復時點決定，不用行政上的
疫情日期；航班量未回到疫情前水準，故分三段。「回復」的門檻與敏感度分析於
下次報告確定，圖中切點為暫定。

申報指標的轉折未必與航班量對齊（#ref(<sec:trend>)），兩者是否同步亦待檢驗。
中段樣本少且航網重組，僅作控制，不單獨解讀。

== 前提檢驗 <sec:premise>

#figure(
  image("../outputs/figures/slides_cause_share.png", width: 88%),
  caption: [限定惡劣天氣子樣本下，各執飛公司的延誤成因分配。天氣段以顏色標示，其餘三段為背景。樣本內班次過少的公司不繪出，圖中 #rows("cause_share").len() 家涵蓋稽核樣本的 #pct(rows("cause_share").map(r => int(r.at(1))).sum() / int(pick(rows("sample_dims"), "航班數")), d: 1)。],
)

設計的前提是公司間的天氣歸因確有差異，且不單由航網結構造成。上圖為未經控制
的原始佔比；另以「控制機場 × 日期」檢驗，要求同一機場同一日有足夠家數的公司、
各有足夠班次，落差仍然存在。機隊、航線距離、班表緩衝等其餘競爭解釋此設定
排除不了（#ref(<ch:limits>)）。

本章是可行性證據。惡劣天氣門檻、收縮強度與時期切點尚未定案，一動排序即隨之
變動，故本提案不給出最終排序。

== 設計的統計依據 <sec:evidence>

圖上看得出的落差，不代表超出雜訊，也決定不了標準誤怎麼算。以下統計讓前述
設計選擇可以檢驗。

#let de = rows("design_evidence")
#let de-pick(kind, item) = float(de.find(r => r.at(2) == kind and r.at(0) == item).at(1))
#figure(
  table(
    columns: (auto, 1fr, auto),
    stroke: (x, y) => if y == 0 { (bottom: 0.7pt) },
    align: (left, left, right),
    row-gutter: 0.42em,
    [量], [它回答的問題], [值],
    [$I^2$（是否歸因）], [公司間的離散有多少是真的], [#pct(de-pick("I2", "y1"))],
    [$I^2$（歸因強度）], [同上], [#pct(de-pick("I2", "y2"))],
    [$I^2$（歸因邊界）], [同上], [#pct(de-pick("I2", "y3"))],
    [ICC（機場 × 年月）], [同一格子內的班次有多相關], [#pct(de-pick("ICC", "機場 × 年月格子"), d: 1)],
    [ICC（機尾號）], [同一架飛機的班次有多相關], [#pct(de-pick("ICC", "機尾號"), d: 1)],
  ),
  caption: [設計層面的三項統計。$I^2$ 為公司間變異佔「公司間變異加抽樣變異」的比例；ICC 為組內相關。ICC 取稽核樣本全量；$I^2$ 取固定效應可識別的公司，即扣除基準公司與僅出現在單一公司格子者。],
)

*公司間的差異超出抽樣雜訊。* 三個 $I^2$ 皆極接近 1，公司間的離散幾乎全部
是真的，這是前一節落差的正式陳述。百萬列資料上任何差異的 $p$ 值都極小，
$I^2$ 則不隨列數膨脹。

*標準誤須雙向聚類。* 兩個 ICC 皆明顯非零，名目標準誤會低估。同一架飛機跨越
數十至上百個格子，兩分組非巢狀，機尾號的相關吸收不進格子聚類。

*以效果量為判準。* 控制機場與年月後，公司能解釋的變異很低，二元結果通常如此。
監理關心的是系統性差異的幅度：只解釋少量變異的因子，公司間的水準差距仍可能
具有制度意義。

=== 為何不採用列聯表檢定

邊際列聯表把所有機場與月份混在一起，顯著性可以完全來自航網結構；卡方檢定
又假設觀測獨立，上表的 ICC 顯示並不成立，檢定量會被高估。固定效應模型是同
一族方法的條件化版本，無共變數時退化為對數線性列聯表分析，另外提供條件化、
聚類標準誤，以及清單需要的逐公司效果量與區間。

= 進度與預期目標

== 工作階段

#figure(
  table(
    columns: (auto, 1fr, auto),
    stroke: (x, y) => if y == 0 { (bottom: 0.7pt) },
    align: (left, left, left),
    row-gutter: 0.4em,
    [階段], [內容], [狀態],
    [資料稽核], [溯源、完整性驗證、缺失分類、樣本界定], [已完成],
    [前提檢驗], [公司效應是否存在、NAS 替代效應的解釋力], [已完成],
    [描述層], [三個比例的公司效應與分布形狀], [初版，口徑未定],
    [推論層], [單體的公司 × 成因係數、歸因邊界], [初版，口徑未定],
    [收縮與清單], [經驗貝氏收縮、小樣本公司的處理], [初版，口徑未定],
    [穩健性], [時期交互、hold-out、氣象基準敏感度], [進行中],
    [上游核對], [與 BTS、NCEI 原始檔逐班比對], [待進行],
    [報告撰寫], [方法說明、限制、監理建議], [待進行],
  ),
  caption: [工作階段依研究問題劃分，不依腳本執行順序。],
)

初版指流程已跑通、數字出得來，口徑則未定案（見@sec:premise）。

== 資料可信度的核對計畫 <sec:verify>

核對項目如下，尚未執行：

+ 自 BTS 官方取得若干月份的 On-Time Performance 原始檔，與年度檔逐班核對
  `reported_` 五欄、`arr_delay_min` 與 `operating_carrier`。
+ 自 NCEI 取得數個代表機場的 ISD 原始年檔，抽驗 `wx_` 欄位的解碼。

== 預期交付物 <sec:deliver>

+ 以 `operating_carrier` 為單位的排序清單，每列附信賴區間，並附該公司惡劣
  天氣下的延誤中位數作反向檢核：表現差但如實申報的公司不應上榜。
+ 描述層與推論層各自的排序，不合成單一分數，並標示兩層方向不一致的公司。
+ 近期排序及其相對前期的變化。知道誰正在變壞，比知道誰當前偏高更有助於決定
  先查誰。
+ 各對成因欄位之間的歸因邊界，附信賴區間。
+ 經驗貝氏收縮後的清單與收縮量。
+ 至少兩項穩健性檢查：時期交互與樣本外驗證。後者切分時間，以前段估計、後段
  驗證，檢視前段的高位者在後段是否仍居高位。排序不一致可能是行為改變或模型
  失敗，兩者的區分於後續報告處理。
+ 一節明列已知限制。

本提案不預告結論的方向。

= 已知限制 <ch:limits>

- *無法觀測真實成因。* 資料中沒有任何欄位記錄某一班延誤的真實成因。本分析
  測到的是歸因行為的公司間差異，推論到偏誤需假設：同一機場、同一月份、同樣
  實測氣象下，各公司面對的真實天氣延誤比例相同。
- *取消的航班不在樣本內。* 惡劣天氣下取消率高的公司，留在樣本中的班次是
  被篩選過的。已初步檢視各公司的取消率與其歸因傾向的關聯，作為下次報告
  正式處理此選擇性的依據。
- *三個競爭解釋無法完全排除*，皆能產生歸因差異而不涉及不實申報：機隊與航線
  特性（小型支線機對低雲強風的容忍度較低）、成因拆分的顆粒度（同一段延誤，
  有的公司全記 `late-aircraft`，有的拆開記）、NAS 的合法邊界。
- *基準的資訊量少於申報人*（#ref(<sec:oracle>)），會把一部分合理的申報誤判
  為偏離。
- *清單不是違規認定。* 認定需要航空公司的內部運行紀錄，那不是公開資料。

= 資料來源與授權

*資料集。* US Flights Joined to Observed Airport Weather by Arimancy LLC,
licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).

*上游來源。*

- US DOT Bureau of Transportation Statistics, Airline On-Time Performance
  Data（美國政府作品，公眾領域）
- NOAA NCEI Integrated Surface Database, ISD（美國政府作品，公眾領域）
- OurAirports（公眾領域）

*監理依據。* US DOT Office of Inspector General 的 BTS 延誤資料稽核報告。

此產品未經 US DOT、BTS、FAA 或 NOAA 背書，亦與之無隸屬關係。
