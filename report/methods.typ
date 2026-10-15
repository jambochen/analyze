#import "@preview/touying:0.6.1": *
#import themes.metropolis: *
#import "@preview/cetz:0.4.2"

#show: metropolis-theme.with(
  aspect-ratio: "16-9",
  config-info(
    title: [美國航班延誤成因申報的責任歸因稽核],
    subtitle: [方法與相關文獻],
    author: [統計研究所　學期分析報告],
  ),
  config-common(new-section-slide-fn: none),
  // 封面也算一頁，頁尾的頁碼才與 PDF 及逐字稿的頁次一致；附錄不顯示頁碼。
  footer-right: context {
    let n = utils.slide-counter.get().first()
    let last = utils.last-slide-counter.final().first()
    if n > last [附錄] else [#(n + 1) / #(last + 1)]
  },
)

#set text(font: ("Noto Sans CJK TC",), lang: "zh", region: "tw", size: 20pt)
#show raw: set text(font: ("DejaVu Sans Mono", "Noto Sans CJK TC"))
#set figure(numbering: none, gap: 0.5em)
#show figure.caption: set text(size: 14pt)
#show regex("[\\p{Han}，。、；：「」（）][ \\n]+[\\p{Han}，。、；：「」（）]"): it => it.text.replace(regex("[ \\n]+"), "")

#let tbl(cols, size: 17pt, ..cells) = text(size: size, table(
  columns: cols,
  stroke: (x, y) => if y == 0 { (bottom: 0.8pt) },
  row-gutter: 0.5em,
  align: left,
  ..cells.pos(),
))

#let step-names = ([界定可比對象], [衡量偏離], [排序與標記], [驗證])
#let sub-names = ([要量什麼], [零的處理], [預期份額], [每班的分數])
#let chip(body, on) = box(
  inset: (x: 0.6em, y: 0.25em),
  radius: 3pt,
  fill: if on { rgb("#d9e6ef") } else { none },
  text(size: 13pt, fill: if on { rgb("#23373b") } else { rgb("#6d797f") }, body),
)
#let steps(k, sub: none, tier: none) = {
  place(top + left, block(width: 100%, {
    grid(
      columns: (1fr,) * 4,
      gutter: 0.4em,
      ..step-names.enumerate().map(((i, n)) => block(
        width: 100%,
        inset: (y: 0.3em),
        radius: 3pt,
        fill: if i == k { rgb("#23373b") } else { rgb("#eef1f3") },
        align(center, text(size: 12pt, fill: if i == k { white } else { rgb("#7a868c") }, n)),
      )),
    )
    if sub != none or tier != none {
      v(-0.5em)
      grid(
        columns: (1fr, auto),
        if sub != none { sub-names.enumerate().map(((i, n)) => chip(n, i == sub)).join(h(0.2em)) },
        if tier != none { box(inset: (x: 0.6em, y: 0.25em), radius: 3pt, stroke: 0.6pt + rgb("#23373b"), text(size: 13pt)[定位：#tier]) },
      )
    }
  }))
  v(if sub != none or tier != none { 2.5em } else { 1.3em })
}

#let thou(x) = {
  let c = x.clusters()
  c.enumerate().map(((i, d)) => if i > 0 and calc.rem(c.len() - i, 3) == 0 { "," + d } else { d }).join()
}
#let funnel = csv("../outputs/tables/sample_funnel.csv").slice(1)
#let mix = csv("../outputs/tables/methods_cause_mix.csv").slice(1)
#let unit-zeros = csv("../outputs/tables/methods_unit_zeros.csv").slice(1)
#let mix-pct(name) = [#calc.round(float(mix.find(r => r.at(0) == name).at(1)) * 100, digits: 1)%]

#let ink = rgb("#4b6277")
#let cause-colors = (rgb("#4b6277"), rgb("#f2c57c"), rgb("#b8c4cc"), rgb("#e3ecf4"))

#title-slide()
#speaker-note[
  這次介紹打算採用的方法方向與相關論文；方法與題目都還可以更換。
  從兩篇不靠逐筆真值找異常的理賠稽核論文出發，把它們的流程拆成四步，再依本題資料的特徵逐步替換。
]

= 資料與目標

== 資料回顧

#tbl(
  (auto, 1fr, auto),
  size: 17pt,
  [申報欄], [記入的延誤分鐘], [占申報總分鐘],
  [`carrier`], [公司可控制的因素], mix-pct("carrier"),
  [`weather`], [極端天氣], mix-pct("weather"),
  [NAS], [非極端天氣、機場運作、航管], mix-pct("NAS"),
  [`late-aircraft`], [同一架飛機的前一航班晚到], mix-pct("late-aircraft"),
  [`security`], [安檢或保安事件], mix-pct("security"),
)

- 到達延誤達 15 分鐘才須申報；未達者為空，非零。
- `security` 極少：二類時併入非 `carrier`，四欄時不納入。
- 達門檻 #thou(funnel.at(1).at(2)) 班；其中 #thou(funnel.at(2).at(2)) 班遇惡劣飛行條件（IFR／LIFR）。

#speaker-note[
  上次介紹過資料；這頁只回顧兩件事：五個申報欄，以及誰需要申報。
  五欄是公司自己填的分類，不是查證過的真實成因。late-aircraft 只記前一班晚到，沒有交代原始成因。
  IFR、LIFR 是依能見度與雲冪界定的飛行規則分類；天氣好壞的區分在講驗證時會再用到。
]

== 目前的分析目標

- 五欄由公司自行申報，沒有例行查核（美國交通部監察長辦公室）。
- 公司有少記 `carrier` 的誘因，也有空間。

#align(center)[
  #block(fill: rgb("#ebf8ff"), inset: 0.7em, radius: 4pt, width: 95%)[
    目標：在同樣的機場、年月與氣象下，找出申報配置與其他航班明顯不同的航班。

    #text(size: 15pt)[申報配置：一班的延誤分鐘在五欄之間如何分配。]
  ]
]

- `carrier` 份額：`carrier` 分鐘占申報總分鐘的比例。
- 查核對象是航班；依公司彙總是第二層。

結果只供查核排序，不代表該航班確有問題。

#speaker-note[
  申報欄位平常沒有人逐一查核：美國交通部監察長辦公室的稽核報告指出，運輸統計局對成因申報沒有例行的一致性查核程序。
  動機是推卸責任，所以直覺上只要看 carrier 份額偏低的航班。但我們沒有真值，只能拿同條件的其他航班當基準；基準本身可能就是偏的，所以兩個方向都留著。
  這是目前在這份資料上找到的目標，之後可能換；後面多數方法換題後仍用得上。
]

== 偏離的兩種可能原因

#align(center)[
  #block(fill: rgb("#ebf8ff"), inset: 0.7em, radius: 4pt, width: 92%)[
    假設觀察：相同觀測條件下，某一班的 `carrier` 份額遠低於其他航班。
  ]
]

#tbl(
  (1fr, 1fr),
  size: 17pt,
  [可能原因一：真實成因不同], [可能原因二：申報不同],
  [公司可控制的延誤確實較少；填法相同], [真實成因相近；分鐘記入 `carrier` 以外的欄],
)

- 申報偏移：分鐘記入的欄位與真實成因不符，即原因二。
- 讀成申報偏移須假設：同條件下各航班真實成因的比例相近。

資料無法檢驗此假設。

#speaker-note[
  量到差異，不等於有人不實申報。
  第一種原因的例子：航路上的雷雨、航管指令或這架飛機的狀況沒有被觀測到，這一班真的較少遇到可歸責於公司的延誤。
  公司層有幾十萬班可以把這些看不到的因素平均掉，單一航班沒有；所以逐班清單裡，第一種原因造成的誤報會比公司層多。
]

== 資料特徵：無序、固定總量與零值

#figure(
  grid(
  columns: (0.7fr, 1.3fr),
  gutter: 0.8em,
  align: horizon,
  align(center, cetz.canvas(length: 1cm, {
    import cetz.draw: *
    let names = ([`carrier`], [`weather`], [NAS], [`late-aircraft`])
    let mins = (25, 0, 35, 0)
    let w = 0.1
    for (k, m) in mins.enumerate() {
      let y = -k * 1.0
      content((-0.3, y + 0.35), anchor: "east", text(size: 14pt, names.at(k)))
      line((0, y - 0.1), (0, y + 0.8), stroke: 0.6pt + ink)
      if m > 0 { rect((0, y), (m * w, y + 0.7), fill: cause-colors.at(k), stroke: none) }
      content((m * w + 0.25, y + 0.35), anchor: "west", text(size: 14pt, str(m)))
    }
    line((-3.4, -3.35), (6.6, -3.35), stroke: 0.6pt + ink)
    content((-0.3, -3.95), anchor: "east", text(size: 14pt)[合計])
    let x = 0
    for (k, m) in mins.enumerate() {
      if m > 0 { rect((x * w, -4.3), ((x + m) * w, -3.6), fill: cause-colors.at(k), stroke: white) }
      x += m
    }
    content((x * w + 0.25, -3.95), anchor: "west", text(size: 14pt, str(x)))
  })),
  image("../outputs/figures/methods_simplex_zeros.png", width: 100%),
  ),
  caption: [Left: a hypothetical flight. Middle: causes with positive minutes per flight. Right: share of #thou(unit-zeros.at(0).at(2)) route-month-carrier units with zero minutes per cause (grey: at least one zero). Security is excluded.],
)

- 無序、固定總量（左）：四欄沒有高低；合計為申報總分鐘。
- 零值（中、右）：單班的份額常恰為 0 或 1。

這三項特徵決定哪些既有做法可以照搬。

#speaker-note[
  這頁先看資料長什麼樣子，後面講論文時才知道哪裡搬不過來。
  左圖的分鐘數是假設的，只用來說明一班的申報長什麼樣子：兩欄有分鐘、兩欄是零，四欄加起來是這班的申報總分鐘。
  中、右兩圖是實際資料，只用四個主要欄位總分鐘為正、氣象資料完整的航班。灰色那條就是要求四欄皆正的做法必須丟掉的單元。零分鐘跟未達門檻的空白不同：前者是申報結果，後者是沒有申報義務。
]

= 兩篇稽核論文

== Crawford & Petela：診斷群內的尾端比例

- 資料：9,793 筆急診理賠，各有診斷代碼與嚴重度代碼 @crawford2022erclaims。
- 可比對象：1,951 個診斷代碼依平均嚴重度合成 10 群。
- 尾端比例：同群中，嚴重度與該筆相同或更高的比例。
- 例：同群 100 筆中只有 3 筆，尾端比例為 0.03。
- 本題四欄無序，須先化成一個有大小的數。

#v(0.5em)

示範前兩步：先定比較對象，再量差距。

#speaker-note[
  這是論文的第一種分析，示範前兩步：先定跟誰比，再量差多少。
  論文先用同一診斷代碼定義尾端比例，再指出代碼太細、同碼案件太少、比例不穩，於是以各代碼的平均嚴重度做階層式分群；最後的分數是在群內算的。
  本題沿用這兩步的想法，但分群的依據與比較的量都要換。
]

== Crawford & Petela：審核順序的預測

- 同篇的另一項分析使用已審核的理賠。

#align(center)[
  #block(fill: rgb("#ebf8ff"), inset: 0.7em, radius: 4pt, width: 90%)[
    審核省下的金額 ＝ 審核前的理賠金額 − 審核後的理賠金額
  ]
]

- 依預測的金額排審核順序。
- 審核一成案件時，比按請款金額排序多省約四成。
- 本題沒有逐班查核結果。

#v(0.5em)

示範後兩步：排序，再以已知答案驗證。

#speaker-note[
  這一項是監督式的：訓練答案是實際審核前後的金額差，預測模型用的是隨機森林。
  審核一成案件時，模型排序省下的金額約 2,160 萬美元，按請款金額排序約 1,540 萬美元；審核比例愈高，差距愈小。
  它示範了後兩步：依分數排序，再用已審核案件驗證排序有沒有用。
]

== Shekhar 等人：三類醫院異常訊號

- 比較單位換成醫院，對應本題的公司彙總。
- 資料：2017 年 Medicare 住院理賠 1,120 萬筆、7,661 家醫院 @shekhar2022fraud。

#tbl(
  (0.8fr, 1.4fr, 1.1fr),
  size: 17pt,
  [訊號], [醫院層級的做法], [與本題的關係],
  [一、支出迴歸], [每家醫院一個截距，找支出偏高者], [附錄的公司層模型],
  [二、診斷代碼], [在大量代碼中找局部異常], [本題僅五欄，不適用],
  [三、相似醫院比較], [先找相似的醫院，再比給付分類], [公司層可比性的檢查],
)

#v(0.3em)

第一、三種訊號用於公司層，第二種不適用。

#speaker-note[
  Medicare 是美國聯邦醫療保險。畫面由上到下就是講的順序：比較單位、資料、兩項要求、三種訊號。
  第一種訊號把給付額對病人病史與地區回歸，再看每家醫院的截距；把醫院換成航空公司、把支出換成申報配置，就是本題估計公司效果的形式；簡報正文的公司層只做彙總，公司效果放在附錄。
  第三種在下一頁說明。
]

== Shekhar 等人：相似醫院的界定

- 依服務類型或病人慢性病的分布找相似的醫院 @shekhar2022fraud。
- 相似度：Hellinger 距離，0 為分布相同，愈近 1 愈不相似。
- 驗證：司法部的反詐欺名單約每 20 家有 1 家；排名前 50 的醫院有 21 家。

#v(0.5em)

相似度只能用條件計算，不能用待稽核的申報分布。

#speaker-note[
  Hellinger 距離量的是「條件像不像」，不是「申報像不像」。
  本題可以用它檢查兩家公司遇到的氣象分布是否相近；但氣象相近不代表航線相同，距離本身也不會告訴我們同樣條件下誰記得比較多。
  所以它留在可比性檢查，不作排序依據。
]

== 論文流程與本題的替換

#tbl(
  (0.75fr, 1.1fr, 1.25fr, 1.25fr),
  size: 17pt,
  [步驟], [論文的做法], [本題為何不能照搬], [本題的方向],
  [界定可比對象], [同診斷群；相似醫院], [條件多維且連續，沒有現成的分群], [同機場、同年月內比較，並控制氣象],
  [衡量偏離], [有序嚴重度的尾端比例], [無序、多欄、總量固定、常為零], [對份額建模，再算 `carrier` 份額的尾端比例],
  [排序與標記], [依分數排序，審核前幾成], [同分者多；未觀測因素多], [分級；再彙總到公司層],
  [驗證], [已審核或已公布的案件], [沒有逐班查核結果], [植入已知的申報偏移],
)

#v(0.3em)

每頁上方標出所在步驟與定位：主要做法、檢查、備選、第二層。

#speaker-note[
  這張表是整份報告的主線，只舉第二列當例子：論文用嚴重度的高低算尾端比例；我的成因沒有高低、一班可填多欄、加起來固定又常是零，所以先對各欄的份額建模，把四欄化成 carrier 份額一個數，再算尾端比例。
  其他三列同樣讀法，具體的方法名稱到最後的總表才列出來。
]

= 界定可比對象

== 可比對象的做法

#steps(0)

#tbl(
  (0.9fr, 1.1fr, 1.4fr),
  size: 17pt,
  [做法], [出處], [定位],
  [分群比較], [#cite(<crawford2022erclaims>, form: "prose")], [改造後採用：依預期份額分群],
  [固定效應加氣象與航班條件], [#cite(<shekhar2022fraud>, form: "prose")；機構評比 @normand2007profiling @kalbfleisch2013monitoring], [主要做法：機場 × 年月（機場月份）各一個截距],
  [限同一航線], [—], [檢查],
  [相似度配對], [#cite(<shekhar2022fraud>, form: "prose")], [公司層的檢查],
)

#text(size: 15pt)[機構評比（profiling）：調整條件後比較一批機構、找出偏離者。]

#text(size: 15pt)[預期份額：同條件的航班平均記在各欄的比例。]

- 公司不放入模型。
- 氣象兩組並列：起飛前的 `wx_` 與事件時刻的 `oracle_`（附錄）。

#speaker-note[
  固定效應在這裡的意思是每個機場月份給一個自己的截距，把跑道、除冰能力、當地航管慣例這些共同條件吸收掉，航班之間的比較就落在同一個機場月份之內。
  氣象與航班條件包含能見度、雲冪、風、降水、天氣現象、航程與到達延誤長度。
  公司不放入：如果基準只取同公司的航班，整家公司的填法都偏時，它的航班反而顯不出來。
  兩組氣象並列：申報是事後填寫，申報人知道實際遇到的天氣，只用起飛前的觀測等於用較少的資訊評斷。
]

= 衡量偏離

== 二類與四欄的假設例子

#steps(1, sub: 0)

- 二類：`carrier` 與非 `carrier`。

#align(center, cetz.canvas(length: 1cm, {
  import cetz.draw: *
  let names = ([`carrier`], [`weather`], [NAS], [`late-aircraft`])
  let rows = (([起始], (40, 30, 20, 10)), ([情況 A], (20, 50, 20, 10)), ([情況 B], (40, 50, 0, 10)))
  let w = 0.16
  for (k, n) in names.enumerate() {
    rect((k * 4.2, 1.55), (k * 4.2 + 0.5, 1.95), fill: cause-colors.at(k), stroke: none)
    content((k * 4.2 + 0.7, 1.75), anchor: "west", text(size: 13pt, n))
  }
  for (r, row) in rows.enumerate() {
    let (label, mins) = row
    let y = -r * 1.2
    content((-0.3, y + 0.45), anchor: "east", text(size: 15pt, label))
    let x = 0
    for (k, m) in mins.enumerate() {
      if m > 0 {
        rect((x * w, y), ((x + m) * w, y + 0.9), fill: cause-colors.at(k), stroke: white)
        content(((x + m / 2) * w, y + 0.45), text(size: 13pt, fill: if k == 0 { white } else { black }, str(m)))
      }
      x += m
    }
    content((16.4, y + 0.45), anchor: "west", text(size: 13pt)[`carrier` 份額：#(mins.at(0) / 100)])
  }
}))

- 情況 A：20 分鐘由 `carrier` 移到 `weather`。
- 情況 B：20 分鐘由 NAS 移到 `weather`。

二類用於排序；四欄用於解讀。

#speaker-note[
  為什麼先分兩類：五欄裡只有 carrier 算公司可控制，誘因是把分鐘移出 carrier，所以先問分鐘有沒有離開 carrier；而且兩類合成 carrier 份額一個數，才有高低可以算尾端比例。四欄指圖中的四個欄位。
  再用例子說明要量什麼；這個 100 分鐘的例子後面講公式時還會用到。
  兩種情況的天氣欄都從 30 變成 50。情況 A 的分鐘離開了 carrier，這正是目前目標關心的責任邊界。
  情況 B 只是非 carrier 欄位之間互換，二類份額不變，要看四欄才知道。
  這是假設的例子，不表示實際有人改寫分鐘。
]

== 零的處理路線

#steps(1, sub: 1)

#align(center, cetz.canvas(length: 1cm, {
  import cetz.draw: *
  let grey = rgb("#a3adb3")
  let node(x0, x1, y, body, fill: rgb("#f1f4f8"), off: false) = {
    rect((x0, y - 0.52), (x1, y + 0.52), fill: if off { white } else { fill }, stroke: if off { grey } else { ink }, radius: 0.12)
    content(((x0 + x1) / 2, y), text(size: 14pt, fill: if off { grey } else { black }, body))
  }
  node(0, 3.4, 0, [欄位常為零], fill: rgb("#e7f1fb"))
  let mids = ((1.4, [只對平均數建模], false), (-1.4, [另設零的產生機制], false))
  for (y, body, off) in mids {
    line((3.4, 0), (4.4, y), stroke: if off { grey } else { ink })
    node(4.4, 9.8, y, body, off: off)
  }
  let leaves = (
    (1.4, 1.4, [PPML（主要做法）、fractional logit（檢查）], false),
    (-1.4, 0, [two-part model（檢查）], false),
    (-1.4, -1.4, [混合模型：不採用], true),
    (-1.4, -2.8, [零膨脹 Dirichlet-multinomial：不採用], true),
  )
  for (my, y, body, off) in leaves {
    line((9.8, my), (10.8, y), stroke: if off { grey } else { ink })
    node(10.8, 24.6, y, body, fill: white, off: off)
  }
}))

#text(size: 14pt)[出處：PPML @santossilva2006log；fractional logit @papke1996fractional；two-part model @ospina2011inflated；混合模型 @grun2024mixture；零膨脹 @koslovsky2023zidm。]

#text(size: 14pt)[灰色的做法不採用。]

主要做法只對平均數建模；零直接放入。

#speaker-note[
  同一個問題有兩條路。
  備詢，組成資料常用的對數比轉換：把兩欄份額相除再取對數後做迴歸，要求每欄都是正的；要湊到都為正就得丟掉大多數航班，所以不列入。
  備詢，為什麼文獻要取對數：份額加總為一、又限在零到一之間，直接做最小平方時預測值可能超出範圍，各欄的預測加總也不保證為一；對數比把份額轉成沒有範圍限制的數，才能套一般的線性模型。
  備詢，不取對數直接處理為什麼可行：每班截距的 PPML 得出的預期份額是 multinomial logistic regression 的形式，本身落在零到一之間且加總為一；fractional logit 的平均數也落在零到一之間。
  只設定平均數的方法不回答零是怎麼來的，只要平均數形式對，零照樣放進去。
  另外三條會替零設定機制：two-part model 把零當成看得到的決策；混合模型假設資料來自幾個看不到的群體，包含混合二項；零膨脹 Dirichlet-multinomial 把結構性的零與抽樣造成的零分開。
  備詢：混合模型須先設定群體數，沒有真值時難以解釋估出的各群代表什麼；零膨脹模型區分「該成因不成立」與「成立但剛好為零」，計算量大。
]

== 含零分鐘的平均數模型：PPML

#steps(1, sub: 2, tier: [主要做法])

- 原文處理貿易額：常為零，取對數須丟掉零 @santossilva2006log。
- 做法：不取對數，直接對平均數建模。
- PPML（Poisson pseudo maximum likelihood）：借用 Poisson 的式子估計，不假設 Poisson 分布 @gourieroux1984pml。

$ mu_i = exp(bold(z)_i^top bold(beta)), quad sum_(i in "同一機場月份") hat(mu)_i = sum_(i in "同一機場月份") y_i $

- $i$：航班；$y_i$：該班某一欄申報的分鐘；$hat(mu)_i$：fit 的分鐘。

一次看一欄的分鐘；個別欄常為零。

#speaker-note[
  PPML 是 Poisson pseudo maximum likelihood。y 是某一欄的分鐘，z 是機場月份、氣象與航班條件。
  備詢：估計準則是 Poisson 形式的準概似，各班的 y log mu 減 mu 加總；它極大的條件是殘差與每個條件變數的乘積加總為零，對機場月份的指標變數就是畫面右式。
  備詢：取對數再做最小平方，在誤差變異數不相等時估計量不具一致性。
  平均數設對時，估計方程式的期望為零，所以對連續的非負結果也成立。
  真實變異數未必等於平均數，標準誤要另外用穩健的方式估。
  它不回答零是哪種機制產生的。
]

== 各欄加總固定：Baker 的轉換

#steps(1, sub: 2, tier: [主要做法])

- 要估的是總分鐘給定時的分配，即 multinomial logistic regression。
- 改用 Poisson 形式、每班各一個截距，估計結果相同 @baker1994multinomial。

#tbl(
  (1fr, 1.3fr, 1.1fr),
  size: 17pt,
  [截距], [fit 的總和等於申報總和], [係數反映],
  [全體共用一個], [只在全體成立；申報 100 分鐘，fit 的總和可能為 130], [總量與分配混在一起],
  [每班各一個], [每班皆成立], [只剩分配],
)

機場月份的截距極多；multinomial logistic regression 須把截距展開成虛擬變數，放不進記憶體。

#speaker-note[
  上一頁一次只看一欄；這頁把同一班的各欄放在一起。
  Baker 原本要估的是 multinomial 模型：總量給定，問各類別的份額怎麼隨條件變。他指出可以改配 Poisson 形式的模型，只要替每一筆多項觀測加一個自己的截距，估出來的份額係數相同。
  共用一個截距時，某一班申報一百分鐘，fit 出來可能是一百三十分鐘；係數同時在解釋總量多少和怎麼分。每班一個截距之後，總量被截距吸收，係數只剩分配。
  截距本身不解讀，只是用來消掉總量。
  機場乘年月的固定效應有上萬個，Poisson 形式可以用去均值的方式吸收，直接估 multinomial logistic regression 做不到；換的是估計的寫法，模型相同。
  備詢，為什麼 multinomial 算不動：兩組機場月份的截距各有上萬個，每個截距對每個類別都要一個係數；一般軟體只能展開成虛擬變數，幾百萬列乘幾萬欄的矩陣放不進記憶體。線性模型可以先去均值消掉截距，非線性的 multinomial 沒有這個捷徑。
  備詢，為什麼 Poisson 算得動：給定其他係數時，Poisson 的每個截距有公式解，就是上一頁右邊那條「該組觀測總和等於 fit 的總和」；所以可以在迭代中直接吸收截距，不必展開。演算法出自 Correia 等人的 ppmlhdfe。
  「算不動」指現成軟體在這個規模下做不到，不是數學上不可能。
  Baker 原文假設計數服從 Poisson；這裡只用它的估計方程式，不假設分鐘的分布。
  逐班估計的列數很多；算不動時，條件相同的航班可以先加總，見附錄的估計風險。
]

== 每班的截距與預期份額

#steps(1, sub: 2, tier: [主要做法])

- $i$：航班；$k$：成因類別（二類或四欄）。
- $M_i$：總分鐘；$mu_(i k)$：第 $k$ 欄 fit 的分鐘；$eta_i$：每班的截距。

$ mu_(i k) = exp(eta_i + zeta_(i k)) quad arrow.r.double quad mu_(i k) = M_i pi_(i k), quad pi_(i k) = exp(zeta_(i k)) / (sum_l exp(zeta_(i l))) $

- $pi_(i k)$：預期份額，同條件的航班平均記在第 $k$ 欄的比例。
- 例：一班 100 分鐘、預期份額 0.6，fit 的分鐘為 60。

總分鐘已知，模型要估的只有份額。

#speaker-note[
  k 是成因類別。
  左式是 PPML 的平均數，截距換成每班各一個。截距的估計方程式要求各欄的 fit 的分鐘加總等於申報總分鐘；把截距解出來就得到右式：各欄 fit 的分鐘等於總分鐘乘上該欄的預期份額，份額是 multinomial logistic regression 的形式。這一步不需要分布假設。
  各類別的 zeta 同加一個常數會被截距吸收，所以只有 zeta 之間的差決定份額。
  備詢：只分兩類時，點估計等同以分鐘加權的 logistic regression；變異數不採 binomial 的，因為一分鐘不是一次獨立試驗。與 PPML 相同，只設定平均數，不假設分鐘的分布。
  總分鐘大的航班權重比較大。
]

== 二類模型：預期份額與差額

#steps(1, sub: 2, tier: [主要做法])

- $"N"$：非 `carrier`；$"C"$：`carrier`。

$ zeta_(i,"N") - zeta_(i,"C") = log (pi_(i,"N") / pi_(i,"C")) = lambda_(o t) + kappa_(d t) + bold(x)_i^top bold(psi) $

- 右邊：起點機場月份、目的地機場月份、氣象與航班條件；不含公司。
- 差額分鐘 $M_i (pi_(i,"C") - s_(i,"C"))$，$s_(i,"C")$ 是實際份額；正值即 `carrier` 偏低。
- 例：100 分鐘、預期 0.4、實際 0.2，差額 20 分鐘。

差額是要量的偏離；它多少見，由尾端比例回答。

#speaker-note[
  o 是起點、d 是目的地、t 是年月；x 是這一班的氣象與航班條件。
  模型裡沒有公司：預期份額是同條件下所有航班的平均，不分公司。
  例：一百分鐘的航班，同條件的航班平均把四成記在 carrier，也就是四十分鐘；情況 A 只記了二十分鐘，差額是二十分鐘。
  零分鐘照常保留；未達門檻的空白不能當成零。
]

== 四欄模型：差異落在哪一欄

#steps(1, sub: 2, tier: [主要做法])

- 類別改為四欄，形式不變；`security` 不納入。
- 實際減預期：分鐘多記在哪一欄、少記在哪一欄。
- 例：20 分鐘由 NAS 移到 `weather`，二類的差額是零，四欄才看得出。

#v(0.5em)

`weather` 與 NAS 之間的差額是分類習慣，不涉及責任歸屬。

#speaker-note[
  這頁沒有新方法，只是把類別集合從兩類換成四類。
  NAS 本身就包含非極端天氣，把天氣造成的延誤記在 NAS 是合法的；所以天氣對 NAS 的差額描述的是記帳習慣，不是責任移轉。
]

== 每班等權的檢查：fractional logit

#steps(1, sub: 2, tier: [檢查])

- 二類模型按分鐘加權：100 分鐘的航班是 20 分鐘的五倍。
- 原文：可恰為 0 或 1 的比率 @papke1996fractional。
- 形式同 logistic regression；0 與 1 不必刪除。

$ bb(E)[s_i | bold(z)_i] = exp(bold(z)_i^top bold(beta)) / (1 + exp(bold(z)_i^top bold(beta))) $

- $s_i$：每班非 `carrier` 的份額；每班等權。

檢查：換一組預期份額後，清單前段是否大致相同。

#speaker-note[
  它跟一般的 logistic regression 最大化的式子形式相同。結果是 0 或 1 時那是 Bernoulli 概似；這裡的結果是份額，同一個式子只當作估計準則。
  若改成按總分鐘加權、用相同的預測式，它的估計方程式就跟二類模型一樣，所以真正提供不同資訊的是「每班等權」與「逐班的氣象」。
  兩種權重回答不同問題，不預期係數逐一相等；要看的是清單前段有沒有換一批航班。
]

== 每班的分數：同群航班的尾端比例

#steps(1, sub: 3, tier: [主要做法])

- 分群：依預期 `carrier` 份額 @crawford2022erclaims。
- 尾端比例：同群中 `carrier` 份額與該班相同或更低的比例；兩側取較小者。
- 例：同群 100 班中只有 3 班，尾端比例為 0.03。
- 同分者多：同分時依差額分鐘排先後。
- 群數是設定，須檢查。

假設：預期份額相近的航班，實際份額的分布也相近。

#speaker-note[
  Crawford 與 Petela 的問題是診斷代碼太細，所以用平均嚴重度把代碼合成十群。我的問題是條件多維而且連續；模型把所有條件濃縮成預期份額一個數，預期份額相近的航班就當成同一群。
  四欄沒有高低，但 carrier 份額是零到一之間的一個數，有大小，所以尾端比例算得出來。
  兩側都算：一側看 carrier 份額是不是低得少見，另一側看是不是高得少見。
  同分的問題：同一群裡 carrier 份額是零的航班，尾端比例都一樣，只能靠差額分鐘再分先後；所以清單比較適合分級，不適合逐名次。
]

== 同分的處理：two-part model

#steps(1, sub: 3, tier: [檢查])

#figure(
  image("../outputs/figures/methods_weather_share.png", width: 62%),
  caption: [Weather cause, by flight. Right panel: flights with positive weather minutes only.],
)

#text(size: 18pt)[
  - 份額大量恰為 0 或 1（point mass）。
  - 三個問題：是否為 0、是否為 1、介於其間是多少 @ramalho2011fractional @ospina2011inflated @chen2023lpm。
]

每班得到出現這種填法的機率，不必分群；同分太多時改以此為主。

#speaker-note[
  圖是實際資料，逐班計算。左邊：絕大多數航班的天氣欄是零。右邊：有填的航班裡，份額在零到一之間散開，但有一大群剛好是一，也就是整段延誤都記天氣。這兩處就是 point mass。carrier 份額在 0 與 1 兩個端點也是同樣的形狀。
  可以分開的理由：零、一、中間三種情形直接看得到，不必推斷哪一班屬於哪一種；整體的平均份額等於各部的乘積，所以各部可以分別設定。
  Ramalho 等人整理了比例資料的 two-part 做法：先用二元模型處理是否為正，再用 fractional regression 處理正的部分。Ospina 和 Ferrari 處理在 0 或 1 有 point mass 的比例資料，把端點與內部分開。
  用法：一班的 carrier 份額是零，就看模型說在它的條件下份額為零的機率多大；機率愈小，這種填法愈少見。這個機率隨每班的條件連續變動，不像分群的尾端比例那樣整群同分。
  線性機率模型就是對 0／1 結果做最小平方；若真實關係不是線性的，它只是線性近似，fit 的值也可能超出零到一。
  這也是跟混合模型的差別：零是看得到的，不需要假設看不到的群體。
]

= 排序與標記

== 由分數到清單

#steps(2)

#tbl(
  (1fr, 1.5fr, 0.6fr),
  size: 17pt,
  [需求], [做法], [定位],
  [同分者多], [依尾端比例分級，不逐名次], [主要做法],
  [查核量有限], [切點取預計查核的比例；#cite(<crawford2022erclaims>, form: "prose") 取一成], [主要做法],
  [清單隨設定改變], [看各設定下清單前段的重疊], [檢查],
  [集中於少數公司], [依公司彙總], [第二層],
)

#v(0.3em)

尾端比例只供排序；逐班不作顯著性的標記。

#speaker-note[
  這一步的需求是算出分數之後才出現的：幾百萬班各有一個分數，同分的很多。
  為什麼逐班不做檢定：尾端比例的參照就是同群的其他航班，所以不管有沒有人不實申報，落在百分之一以下的航班都大約是百分之一。它只能排先後，不能說有幾班顯著。
  標記留在公司層：公司只有十幾家，各有估計誤差，那裡才需要檢定與多重檢定的校正；做法見附錄。
]

== 航班清單的呈現形式

#steps(2, tier: [主要做法])

#tbl(
  (0.8fr, 0.45fr, 0.9fr, 0.7fr, 0.7fr, 1.05fr, 0.9fr),
  size: 15.5pt,
  [級別], [航班], [方向], [尾端比例], [差額分鐘], [差異所在欄位], [各設定下],
  [優先查核], [1], [`carrier` 偏低], [0.01], [120], [`weather`／`carrier`], [皆在前段],
  [優先查核], [2], [`carrier` 偏高], [0.01], [45], [NAS／`carrier`], [皆在前段],
  [待觀察], [3], [`carrier` 偏低], [0.04], [60], [`weather`／`carrier`], [部分在前段],
  [不列入], [4], [`carrier` 偏低], [0.30], [10], [—], [—],
)

#text(size: 16pt)[假設的例子，非估計結果。]

級別依尾端比例與各設定下的一致性。

#speaker-note[
  級別來自尾端比例：比例小、而且換設定之後仍在前段的列為優先；只在部分設定下排在前段的放在待觀察。
  差異所在欄位來自四欄模型，告訴查核的人先看哪兩欄。
]

== 公司層的彙總

#steps(2, tier: [第二層])

- 列入率：該公司的航班中被列入的比例；偏低與偏高分開算。
- 差額分鐘加總：正值即整體 `carrier` 偏低。

#tbl(
  (0.6fr, 0.8fr, 1fr, 1.2fr),
  size: 17pt,
  [公司], [列入率], [對全體的倍數], [差額分鐘加總],
  [全體], [0.10], [1.0], [—],
  [A], [0.18], [1.8], [正，`carrier` 偏低],
  [B], [0.10], [1.0], [接近零],
)

#text(size: 16pt)[假設的例子，非估計結果。]

彙總只說明列入的航班集中在哪些公司；查核對象仍是航班。

#speaker-note[
  這一頁沒有新模型，只是把前面每班的分數與差額分鐘依公司加起來。
  預期份額已經放入機場月份與氣象，所以列入率的差別是調整這些條件之後的差別；集中的原因同樣可能是兩種原因的任一種。
  例：全體的列入率是一成，公司 A 是一成八，它的航班被列入的機會是全體的一點八倍。
  兩個量各看一件事：列入率看被列入的班數多不多，差額分鐘加總看整體往哪個方向偏、偏多少分鐘。
  備詢：差額分鐘加總明顯不為零，等同於在二類模型加入公司項時公司效果不為零；公司效果的式子、公司之間怎麼比先後、要不要標記，都在附錄。
]

= 驗證

== 驗證的做法

#steps(3)

#tbl(
  (0.85fr, 1.4fr, 0.6fr),
  size: 17pt,
  [做法], [回答的問題], [定位],
  [已知的答案（兩篇論文）], [排在前面的對象是否真有問題], [本題沒有],
  [plasmode 模擬 @schreck2023plasmode], [植入申報偏移後，清單能否找回], [主要做法],
  [negative control @lipsitch2010negative], [非惡劣天氣的航班是否仍動用 `weather` 欄], [檢查],
  [形狀限制的檢定 @chetverikov2012monotonicity], [天氣愈差，`weather` 欄的分鐘是否不減少], [備選],
  [敏感度分析 @cinelli2020sensitivity], [未觀測因素多強才改變公司層的結論], [備選],
)

#v(0.3em)

後四項皆不需逐班真值。

#speaker-note[
  沒有答案時怎麼知道方法有沒有用，有幾條路。
  形狀限制是檢查「天氣愈差愈該歸因天氣」這種應該成立的單調關係。
  敏感度分析不產生新證據，只回答結論有多脆弱。
]

== 沒有真值時的模擬：plasmode

#steps(3, tier: [主要做法])

- 一般的模擬：重複生成資料、重估模型。
- plasmode：保留觀測到的條件，只生成結果 @schreck2023plasmode。

#align(center)[
  #cetz.canvas(length: 1.5cm, {
    import cetz.draw: *
    let a = (stroke: (paint: ink, thickness: 0.9pt), mark: (end: ">"))
    let boxes = (
      (0.0, rgb("#e7f1fb"), [保留航班], [公司、航線、氣象]),
      (2.55, rgb("#f1f4f8"), [生成成因], [五欄分鐘]),
      (5.10, rgb("#fff0d5"), [植入申報偏移], [總分鐘不變]),
      (7.65, rgb("#e7f1fb"), [重算分數], [規則相同]),
      (10.20, rgb("#e9f4e9"), [評估清單], [前段找回多少]),
    )
    for (x, fill, title, detail) in boxes {
      rect((x, 0.3), (x + 2.25, 1.9), fill: fill, stroke: ink)
      content((x + 1.125, 1.4), text(size: 13pt, title))
      content((x + 1.125, 0.8), text(size: 11pt, detail))
    }
    for x in (2.25, 4.80, 7.35, 9.90) {
      line((x, 1.1), (x + 0.30, 1.1), ..a)
    }
  })
]

- 植入：把一部分 `carrier` 分鐘移到 `weather`，例如兩成；隨機或集中於指定公司。

模擬中的真值只存在於生成的資料，不能反推實際航班的成因。

#speaker-note[
  plasmode 的好處是公司的航線分布、遇到的氣象、機尾號這些複雜的結構都是真的，不用自己編；只有成因分鐘和申報過程是設定的。
  植入的例子：對指定的、天氣欄已有正分鐘的航班，把一定比例的 carrier 分鐘移到 weather；總分鐘不變，各欄不為負。
  每次生成後重跑同一套選樣、模型、分數與清單，看被植入的航班排在哪裡；集中於指定公司時，另看那幾家公司的列入率有沒有升高。
]

== 模擬情境與評估指標

#steps(3, tier: [主要做法])

#tbl(
  (0.8fr, 1.3fr, 1.3fr),
  size: 17pt,
  [情境], [設定], [要看的指標],
  [零偏移], [全部航班共用申報規則], [清單前段的重疊；各公司的列入率],
  [植入申報偏移], [`carrier` 移到其他欄（情況 A）], [前幾成命中多少，與隨機抽查相比],
  [非 `carrier` 欄互換], [非 `carrier` 欄之間移動（情況 B）], [二類應看不出；四欄能否看出],
  [真實成因不同], [成因不同，填法相同], [多少被排進清單前段],
)

#text(size: 15pt)[命中：被植入申報偏移的航班落在清單前段。]

#v(0.2em)

結果只適用於設定過的情境 @morris2017simulation。

#speaker-note[
  第一種情境沒有任何人偏移，看清單本身穩不穩，以及各公司的列入率會不會只因航線不同就偏離全體。
  第二、三種對應一開始 100 分鐘的例子：第二種是分鐘由 carrier 移到 weather，第三種是由 NAS 移到 weather。第二種的指標跟 Crawford 與 Petela 的驗證同形：只查前一成時，比隨機抽查多抓到多少。
  第四種是刻意違反「真實成因比例相近」的假設：有些航班的真實成因就是不同，但照同一套規則填，看它們有多少被誤排到前段。
  植入的比例不等於模型係數的真值，各指標的目標值要另外算。
]

== 非惡劣天氣下的天氣申報：negative control

#steps(3, tier: [檢查])

- 找一個不該出現效果的情境當對照 @lipsitch2010negative。

#align(center)[
  #cetz.canvas(length: 1cm, {
    import cetz.draw: *
    let a = (stroke: (paint: ink, thickness: 0.9pt), mark: (end: ">"))
    let boxes = (
      (0.0, rgb("#e7f1fb"), [對象], [兩端皆非惡劣天氣的航班]),
      (6.3, rgb("#f1f4f8"), [基準], [`weather` 欄的分鐘應接近零]),
      (12.6, rgb("#fff0d5"), [限制], [航路上的天氣看不到，單班不能認定有誤]),
      (18.9, rgb("#e9f4e9"), [做法], [列出動用 `weather` 欄的航班，與清單對照]),
    )
    for (x, fill, title, detail) in boxes {
      rect((x, 0), (x + 5.7, 2.8), fill: fill, stroke: ink, radius: 0.12)
      content((x + 2.85, 2.25), text(size: 15pt, title))
      content((x + 2.85, 1.0), block(width: 5.1cm, align(center, text(size: 13.5pt, detail))))
    }
    for x in (5.7, 12.0, 18.3) {
      line((x, 1.4), (x + 0.6, 1.4), ..a)
    }
  })
]

- 例：兩端皆非惡劣天氣，`weather` 欄卻有 40 分鐘。
- 只適用於 `weather` 欄。

基準來自氣象觀測，故能指出方向：`weather` 欄多記。

#speaker-note[
  negative control 是找一個理論上不該出現效果的情境當對照。
  它的價值在基準：前面的方法都只能說哪一班跟別人不一樣；這裡有一個接近零的外部基準，所以能說方向。
  但航路天氣看不到，單班不能認定申報有誤，只能列入查核；它跟尾端比例是兩種獨立的依據，重疊的航班證據較強。
  公司層的比較需要一個假設：同一機場月份之內，各公司遇到的未觀測天氣沒有系統性的差別。各公司的樞紐不同，遇到雷雨的機會也不同，所以比較必須限在同一機場月份之內。
]

= 總結

== 資料特徵與方法總表

#tbl(
  (0.75fr, 1.1fr, 1.3fr, 1.25fr),
  size: 17pt,
  [步驟], [本題的資料特徵或需求], [主要做法], [檢查、備選與第二層],
  [界定可比對象], [條件多維且連續], [機場月份的截距加氣象，得出預期份額], [同航線；兩組氣象],
  [衡量偏離], [無序、多欄、總量固定、常為零], [每班截距的 PPML；尾端比例], [fractional logit；two-part model],
  [排序與標記], [同分者多；未觀測因素多], [依尾端比例分級], [設定之間的重疊；公司層的彙總],
  [驗證], [沒有逐班查核結果], [plasmode 模擬], [negative control；形狀限制；敏感度分析],
)

#v(0.3em)

換題時主要須改欄位的分類；其餘方法多可沿用。

#speaker-note[
  回到一開始的四步，這次把方法名稱填進去。每一列由左到右就是前因後果：資料有這個特徵，所以用這個方法，再用右邊的方法檢查。
  如果之後目標換了，要改的主要是怎麼分類；估計、排序與驗證的方法大多不必換。
]

== 共同的限制

- 觀測氣象是比較的條件，不能證明單班的成因。
- 航路天氣與航管指令未觀測；清單會有誤報。
- 範圍限於已執飛且達申報門檻的航班。
- 模擬只能檢查設定過的情境。

#v(0.6em)

清單提供查核的優先順序，不構成不實申報的認定。

#speaker-note[
  要認定個案是否不實申報，仍需要航空公司內部的運行、簽派與航管紀錄。
  取消的航班與未達十五分鐘門檻的航班不在樣本內。
]

#show: appendix

= 附錄

== 參考文獻

#text(size: 7pt)[
  #columns(3, bibliography("refs.yml", style: "apa", title: none))
]

== 公司效果：加入公司項的二類模型

- 問題：某公司的填法是否整體與其他公司不同。
- 做法：二類模型加入公司項 $b_c$，即 Shekhar 等人的醫院截距 @shekhar2022fraud。

$ log (pi_(i,"N") / pi_(i,"C")) = lambda_(o t) + kappa_(d t) + b_c + bold(x)_i^top bold(psi) $

- 公司效果 $b_c$：其他條件相同時，該公司把分鐘記在非 `carrier` 的傾向；$b_c > 0$ 即 `carrier` 份額偏低。兩家之差是 log odds ratio，不受基準公司影響。
- 與航班層的關係：不含公司項時，某公司各班的差額分鐘加總若明顯不為零，$b_c$ 即不為零。

#speaker-note[
  c 是公司。估計時要選一家基準公司，但任意兩家公司之差不受基準影響。
  與航班層的關係來自 PPML 的估計條件：放入公司項之後，每家公司觀測與 fit 的分鐘總和必須相等。不含公司項的模型下，某公司的差額分鐘加總若不是零，放入公司項時就要靠 b_c 把它補平。
  例：非 carrier 六十分鐘對 carrier 四十分鐘，odds 是一點五；情況 A 之後是八十對二十，odds 是四；odds ratio 約二點七。odds ratio 是比值之比，不是機率差，也不是實際移轉了多少分鐘。
  公司、機場與年月相同的航班可以先加總成單元再估，計算較省；公司之間能不能比，見附錄的比較機會。
  排序用的偏離是公司效果與各公司平均的差距。
]

== 公司名次的不確定性：confidence set for ranks

#grid(
  columns: (1.5fr, 1fr),
  gutter: 0.8em,
  text(size: 17.5pt)[
    - 排序用的偏離是公司效果與各公司平均的差距 $b_c - overline(b)$；依絕對值由大到小排，正負（`carrier` 偏低或偏高）另列。
    - 機構名次有抽樣誤差，點估計的名次可能互換 @goldstein1996league。
    - 做法：先取各公司效果的聯合信賴區域，即同時涵蓋各係數的區域（Bonferroni 或 F 分布的橢圓體）；對區域內每一組可能的公司效果重排名次，記下各公司最前與最後的名次 @chetverikov2024csranks。
    - 真實效果落在區域內時，所有公司的真實名次同時落在各自的範圍內。
  ],
  align(center, cetz.canvas(length: 1cm, {
    import cetz.draw: *
    let ranges = ((1, 1, 2), (2, 1, 3), (3, 2, 6), (4, 2, 7), (5, 3, 7), (6, 3, 8), (7, 6, 8), (8, 7, 8))
    let s = 0.8
    line((0.4, -6.9), (7.0, -6.9), stroke: ink)
    for r in range(1, 9) {
      content((r * s, -7.25), text(size: 10pt, str(r)))
    }
    content((3.7, -7.9), text(size: 11pt)[可能名次])
    for (p, lo, hi) in ranges {
      let y = -p * 0.78
      line((lo * s, y), (hi * s, y), stroke: (paint: ink, thickness: 2.2pt))
      circle((p * s, y), radius: 0.11, fill: white, stroke: ink)
    }
    content((3.7, 0.1), text(size: 11pt)[示意，非估計結果])
  })),
)

#text(size: 16pt)[例：偏離為 0.5、0.45、0.1 的三家，可能名次依序是 1 至 2、1 至 2、3。]

範圍寬表示目前分不出先後；中段若普遍偏寬，公司層同樣宜分級而非逐名次。

#speaker-note[
  偏高偏低都可能值得查，所以用絕對值排，正負另外列。
  聯合信賴區域就是迴歸課講的多個係數的聯合信賴區域：各自的 95% 區間相交，同時成立的機率不到 95%，所以要用 Bonferroni 放寬每個區間，或用 F 分布得到的橢圓體。這裡多做的一步，只是把區域裡每一組係數都拿去排名次。
  個別公司的區間有沒有重疊，不能取代整組效果一起變動的計算。
  右圖是示意：圓點是點估計的名次，橫線是可能的名次範圍。
  小例子：三家公司的偏離是 0.5、0.45、0.1；前兩家接近，名次範圍都是第一到第二，第三家固定在第三。
  絕對值在零點不平滑，名次範圍的近似與涵蓋率要在模擬裡檢查。
]

== 多家公司的多重檢定：BH 與 empirical null

- FDR（錯誤發現率）：被標記的公司中，其實 $b_c = overline(b)$ 者所占比例的期望。
- Benjamini-Hochberg（BH）procedure @benjamini1995fdr：$n$ 個 $p$ 值由小到大排，標記前 $m^*$ 家；$q$ 是目標水準。

$ m^* = max{m : p_((m)) <= m q \/ n} $

- 弱點：樣本量大時幾乎任何公司差異都顯著，多數公司都會被標記。
- empirical null：改由全體機構的檢定統計量估計「無異常」的參照分布 @efron2007size @xia2019profiling；本題公司數少，這個分布估不準。

兩種標記並列，不決定排序；用途是認出證據最弱的一群：BH 未標記的公司，資料連「與平均有差異」都看不出。

#speaker-note[
  這裡的「錯誤」是統計上的：公司效果其實等於平均卻被標記，不是指航空公司有沒有不實申報。
  各公司的效果共用資料和平均，檢定彼此相依，獨立條件下的 FDR 保證不能直接引用，實際的誤標比例要在模擬裡看。
  Xia 等人在醫療機構評比遇到同樣的問題：機構一大，任何偏離都顯著。他們把調整不完全造成的機構間變異算進參照分布。醫療評比有上千家機構可以估這個分布，我們只有十幾家。
  兩個工具各有問題，所以不拿任何一個當入列的門檻，只當成並列的兩欄證據。
]

== 公司層的比較機會

#figure(
  grid(
  columns: (1fr, 1.15fr),
  gutter: 1em,
  align: horizon,
  align(center, cetz.canvas(length: 1cm, {
    import cetz.draw: *
    rect((0, 0), (4.4, 3), stroke: ink, fill: rgb("#f1f4f8"), radius: 0.15)
    rect((5.4, 0), (9.8, 3), stroke: ink, fill: rgb("#f1f4f8"), radius: 0.15)
    content((2.2, 2.55), text(size: 12pt)[X 機場・一月])
    content((7.6, 2.55), text(size: 12pt)[Y 機場・二月])
    for (x, n) in ((1.1, [A]), (3.3, [C]), (6.5, [C]), (8.7, [B])) {
      circle((x, 1.2), radius: 0.5, fill: white, stroke: ink)
      content((x, 1.2), text(size: 14pt, n))
    }
    line((1.6, 1.2), (2.8, 1.2), stroke: ink)
    line((7.0, 1.2), (8.2, 1.2), stroke: ink)
    line((3.8, 1.2), (6.0, 1.2), stroke: (paint: ink, dash: "dotted"))
    bezier((1.1, 0.7), (8.7, 0.7), (4.9, -1.9), stroke: (paint: ink, dash: "dashed"))
    content((4.9, -0.95), text(size: 12pt)[A、B 經 C 間接比較])
  })),
  image("../outputs/figures/methods_overlap_levels.png", width: 100%),
  ),
  caption: [Right: share of flights in groups with at least two carriers (orange), of carrier pairs sharing a group (blue), and of carrier pairs linked directly or through other carriers (grey). Connectedness: @eccleston1974connected, @weeks1964connectedness, @abowd2002connected.],
)

#text(size: 18pt)[
  - 直接比較（同場）：兩家公司在同一組出現；組內只有一家則無從比較。
  - 間接比較：不同場兩家經第三家比較，即連通（公司為處理、機場月份為區集）。
]

機場 × 年月下幾乎每對公司都直接同場，公司效果在此分組下估計；同航線只作檢查。

#speaker-note[
  右圖兩排用同一批航班，只是分組方式不同：上排是起點機場加年月，下排再加目的地。
  橘柱回答有多少航班用得上；只有一家公司的組合不會因為間接比較而變得可用。藍柱和灰柱的單位是公司對，18 家公司共 153 對；兩種分組下全部公司都連成一個連通成分。
  這張圖只回答能不能比，不回答比得準不準：兩家只同場過幾班也算直接同場。
  左圖是間接比較：A、C 在 X 機場同月，C、B 在 Y 機場同月；A、B 從未同場，但都跟 C 比較過。這在實驗設計裡叫連通性：線性加法模型下，設計連通是所有處理對比可估計的條件（Eccleston 與 Hedayat）。分類超過兩個方向時，連通的判定見 Weeks 與 Williams；雇主與員工的固定效應模型有同樣的討論（Abowd 等人）。這些都是線性模型的結果；本題用 PPML，連通只是必要條件，仍須檢查公司效果能否估計。
  即使同場同月，兩家遇到的氣象也可能不同，所以還要看氣象分布是否重疊。
]

== 氣象資訊的三種設定

#tbl(
  (auto, 1.25fr, 1.5fr),
  [設定], [氣象欄位], [比較條件],
  [起飛前], [`wx_`], [給定事前的天氣],
  [事件時刻], [`oracle_origin_dep_*` 與 `oracle_dest_scharr_*`], [給定申報時已發生的天氣],
  [兩組並用], [以上兩組], [同時使用兩種資訊],
)

#v(0.5em)

- `oracle_dest_arr_*` 的觀測時刻取決於實際落地時間，與延誤機械相關，排除。
- `reported_*` 是被比較的申報結果，不放入條件。

三種設定使用同一批航班。

== 估計的風險

- 逐班估計的列數與固定效應都多；算不動時須把條件相同的航班加總，氣象取平均會掩蓋逐班的非線性關係。
- 某機場月份或某公司從不使用某欄時可能出現 separation：相應係數沒有有限的估計值 @correia2019existence。
- 大量固定效應與少數含大量航班的機場月份，可能影響係數與區間的準確性 @weidner2019bias。

#v(0.5em)

穩健標準誤只處理變異數，不能修正平均數設錯。

== 設定的檢查項目

#tbl(
  (1.25fr, 2fr),
  [檢查], [針對的風險],
  [限同航線、同年月的多公司航班重估], [兩端機場的固定效應未吸收航線特有的差異],
  [改變 `late-aircraft` 的二類歸屬後重估], [前班晚到的原始成因不明 @bilotkach2022causes],
  [逐班等權的 fractional logit], [長延誤航班主導分鐘加權的結果],
  [公司層逐班重估], [單元聚合後，同一機尾號的相關是否使區間失準],
  [依到達延誤長度分段後重估], [單元只留平均延誤長度，短延誤與長延誤的分配差異被平均掉],
  [依天氣惡劣與否分開重估公司效果], [天氣惡劣時各欄的用法不同，公司之間的差異可能只在這時出現],
)

== 延誤長度：放入條件，不作分組

- 已經控制：航班條件包含到達延誤長度，預期份額是延誤長度相近時的份額。
- 不能取代機場月份：同樣延誤 60 分鐘，暴雪與晴天下的真實成因不同。延誤長度是延誤的規模，不是造成延誤的條件。
- 不另外切成組：在同一機場月份內再依長度切分，每組的航班變少，預期份額估不準。
- 限制：延誤長度決定航班是否達申報門檻；各公司達門檻的比例若不同，進入比較的航班組成也不同。

#v(0.4em)

檢查：把延誤長度分成數段後重估，看清單前段是否改變。

== 模擬的單次生成

- 先生成共同申報規則下的申報 $R^0$。
- 對指定的、天氣欄已有正分鐘的航班，設定移轉比例 $rho in [0, 1]$。

$ R_(i,"C") = (1 - rho) R^0_(i,"C"), quad R_(i,"W") = R^0_(i,"W") + rho R^0_(i,"C") $

- $"C"$ 為 `carrier`，$"W"$ 為 `weather`；其餘欄位維持 $R^0$，故各欄非負、總分鐘不變。
- $rho$ 是模擬的設定，不是由實際資料估得的係數。

以生成的申報取代 `reported_*`，按相同規則選樣、重估並重算分數。
